"""Async execution helpers for running scenarios with LangChain and MCP tools."""

from __future__ import annotations

import asyncio
import json
import textwrap
from typing import Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.tools import BaseTool
from rich.console import Console
from rich.table import Table

from .run_logging import ConsoleRunLogger, RunLogger
from .verifier import evaluate_verifiers

from .prompts import Scenario, scenario_summary


def _serialize_tool_output(result: object) -> object:
    """Render tool outputs in a consistent structure."""

    if isinstance(result, str):
        stripped = result.strip()
        if stripped:
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return result
        return result
    if isinstance(result, (int, float, bool)) or result is None:
        return result
    return result


def _format_json(value: object) -> str:
    """Best-effort JSON formatting for debug output."""

    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except TypeError:
        text = str(value)
        max_len = 1200
        if len(text) > max_len:
            return text[:max_len] + "…"
        return text


def _collect_reasoning_chunks(reasoning_block: object, chunks: list[str]) -> None:
    """Normalize reasoning data from various providers into text snippets."""

    if reasoning_block is None:
        return
    if isinstance(reasoning_block, str):
        if reasoning_block:
            chunks.append(reasoning_block)
        return
    if isinstance(reasoning_block, dict):
        text = reasoning_block.get("text")
        if isinstance(text, str) and text:
            chunks.append(text)
        summary = reasoning_block.get("summary")
        if isinstance(summary, list):
            for entry in summary:
                _collect_reasoning_chunks(entry, chunks)
        elif isinstance(summary, dict):
            _collect_reasoning_chunks(summary, chunks)
        steps = reasoning_block.get("steps")
        if isinstance(steps, list):
            for step in steps:
                _collect_reasoning_chunks(step, chunks)
        return
    if isinstance(reasoning_block, list):
        for entry in reasoning_block:
            _collect_reasoning_chunks(entry, chunks)


def extract_ai_message_content(message: AIMessage) -> tuple[str, list[str], list[str]]:
    """Return primary text plus any reasoning traces from an AI message."""

    primary_chunks: list[str] = []
    reasoning_chunks: list[str] = []
    raw_reasoning: list[str] = []
    content = message.content

    if isinstance(content, str):
        if content:
            primary_chunks.append(content)
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type in {"output_text", "text", None}:
                text = block.get("text")
                if text:
                    primary_chunks.append(text)
            elif block_type == "reasoning":
                summary = block.get("summary")
                if summary:
                    raw_reasoning.append(_format_json({"summary": summary}))
                _collect_reasoning_chunks(block, reasoning_chunks)
            elif block_type == "message":
                # Anthropic style message blocks may embed text directly.
                text = block.get("text")
                if text:
                    primary_chunks.append(text)
            elif block_type == "tool_result":
                text = block.get("content")
                if isinstance(text, str):
                    primary_chunks.append(text)
            else:
                text = block.get("text")
                if text:
                    primary_chunks.append(text)

    extra_reasoning = message.additional_kwargs.get("reasoning") if hasattr(message, "additional_kwargs") else None
    if extra_reasoning:
        summary = None
        if isinstance(extra_reasoning, dict):
            summary = extra_reasoning.get("summary")
        if summary:
            raw_reasoning.append(_format_json({"summary": summary}))
    _collect_reasoning_chunks(extra_reasoning, reasoning_chunks)

    primary_text = "\n".join(chunk for chunk in primary_chunks if chunk)
    return primary_text, reasoning_chunks, raw_reasoning


async def _invoke_tool(tool: BaseTool, arguments: dict) -> object:
    """Invoke the MCP tool with the provided arguments."""

    if hasattr(tool, "ainvoke"):
        return await tool.ainvoke(arguments)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, tool.invoke, arguments)


async def _model_step(
    *,
    llm_with_tools: BaseChatModel,
    messages: List[BaseMessage],
    tool_map: Dict[str, BaseTool],
    logger: RunLogger,
    remaining_tool_calls: Optional[int],
    verifier_hook: Optional[Callable[[], Awaitable[None]]] = None,
) -> Tuple[AIMessage, Optional[int]]:
    """Iteratively invoke the model while handling tool calls."""

    while True:
        ai_message: AIMessage = await llm_with_tools.ainvoke(messages)
        primary_text, reasoning_chunks, _ = extract_ai_message_content(ai_message)
        if reasoning_chunks:
            label_base = "Reasoning" if len(reasoning_chunks) == 1 else "Reasoning {}"
            for idx, chunk in enumerate(reasoning_chunks, start=1):
                label = "Reasoning" if len(reasoning_chunks) == 1 else label_base.format(idx)
                logger.print(f"[yellow]{label}[/yellow]: {chunk}")
        if primary_text:
            logger.print(f"[bold cyan]AI[/bold cyan]: {primary_text}")
        logger.log_message(
            "assistant",
            primary_text or "",
            reasoning=reasoning_chunks if reasoning_chunks else None,
        )
        messages.append(ai_message)

        tool_calls = ai_message.tool_calls or []
        if not tool_calls:
            return ai_message, remaining_tool_calls

        for tool_call in tool_calls:
            if remaining_tool_calls is not None:
                if remaining_tool_calls <= 0:
                    raise RuntimeError("Tool call budget exhausted.")
                remaining_tool_calls -= 1

            tool_name = tool_call["name"]
            tool = tool_map.get(tool_name)
            if tool is None:
                error_text = f"Requested unknown tool '{tool_name}'."
                logger.print(f"[bold red]{error_text}[/bold red]")
                messages.append(
                    ToolMessage(
                        content=error_text,
                        tool_call_id=tool_call.get("id", tool_name),
                        name=tool_name,
                    )
                )
                continue

            raw_args = tool_call.get("args", {})
            logger.print(
                "[green]→ Invoking tool[/green] [bold]{name}[/bold] with args [italic]{args}[/italic]".format(
                    name=tool_name,
                    args=_format_json(raw_args),
                )
            )
            serialized_content: Optional[str] = None
            raw_response: Optional[object] = None
            tool_error = False
            try:
                output = await _invoke_tool(tool, raw_args)
                serialized_obj = _serialize_tool_output(output)
                logger.print(f"[magenta]← Tool response[/magenta]: {_format_json(serialized_obj)}")
                raw_response = serialized_obj
                serialized_content = (
                    _format_json(serialized_obj)
                    if isinstance(serialized_obj, (dict, list))
                    else str(serialized_obj)
                )
            except Exception as exc:  # noqa: BLE001
                serialized_content = f"Tool '{tool_name}' failed with error: {exc!r}"
                logger.print(f"[bold red]{serialized_content}[/bold red]")
                raw_response = serialized_content
                tool_error = True

            messages.append(
                ToolMessage(
                    content=serialized_content,
                    tool_call_id=tool_call.get("id", tool_name),
                    name=tool_name,
                )
            )

            logger.log_tool_call(tool_name, raw_args, raw_response, error=tool_error)

            if verifier_hook is not None:
                await verifier_hook()


async def execute_scenario(
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    scenario: Scenario,
    *,
    tool_call_limit: Optional[int] = 1000,
    logger: RunLogger | None = None,
    sql_runner_url: Optional[str] = None,
    database_id: Optional[str] = None,
    verifier_client: Optional[httpx.AsyncClient] = None,
) -> list[AIMessage]:
    """Run the prompts in a scenario and return the final assistant messages."""

    if logger is None:
        logger = ConsoleRunLogger(Console(stderr=True))

    tool_map: Dict[str, BaseTool] = {tool.name: tool for tool in tools}

    intro = Table(title=f"Scenario {scenario.scenario_id}")
    intro.add_column("Key", style="bold")
    intro.add_column("Value")
    display_summary = scenario_summary(scenario, include_expected_tools=True)
    for line in display_summary.splitlines():
        key, _, value = line.partition(": ")
        intro.add_row(key, value)
    logger.print(intro)

    instructions = textwrap.dedent(
        """
        You are an autonomous project-management agent operating inside an MCP server.
        You receive: (a) a user request and (b) a set of tool definitions (schemas, params, return types).
        Your goal is to complete the tasks assigned to you by using these tools effectively and efficiently.
        Your only way to read or modify project state is via these tools.

        Operating constraints
        - Non-interactive: The user will not answer follow-ups. Do not ask questions. Do not halt for clarification.
        - Obligation: Treat the user request as correct and feasible with the provided context. Execute to completion with best effort.
        - Tools-first: Treat tool definitions as the single source of truth. Do not fabricate data, IDs, or results. Never assume hidden state.

        Core behavior
        - Objective-first: Extract the core objective succinctly and decompose into the minimal set of steps to achieve it.
        - Read-before-write: When safe and efficient, fetch current state to avoid duplicates, race conditions, or destructive updates.
        - Preconditions: Check the parameters before calling tools.

        Tool usage policy
        - Error Handling: Incase a tool call results in an error, retry the tool calling adjusting the parameters based on the error message, retrying with same parameters will only result in the same error.
        """
    )

    scenario_details = scenario_summary(scenario)
    messages: List[BaseMessage] = [
        SystemMessage(content=f"{instructions.strip()}")
    ]
    logger.log_message("system", instructions.strip())
    logger.log_message("system", scenario_details)
    llm_with_tools = llm.bind_tools(tools)
    final_ai_messages: List[AIMessage] = []
    remaining_tool_calls = tool_call_limit

    for index, prompt in enumerate(scenario.prompts, start=1):
        logger.rule(f"Prompt {index}")
        user_message = HumanMessage(content=prompt.prompt_text)
        messages.append(user_message)

        logger.print(f"[bold magenta]User prompt[/bold magenta]: {prompt.prompt_text}")
        logger.log_message("user", prompt.prompt_text)

        verifier_hook: Optional[Callable[[], Awaitable[None]]] = None

        if sql_runner_url and database_id:
            async def incremental_verifiers() -> None:
                results = await evaluate_verifiers(
                    scenario,
                    sql_runner_url=sql_runner_url,
                    database_id=database_id,
                    client=verifier_client,
                )
                if results:
                    logger.update_verifier_status(results)

            verifier_hook = incremental_verifiers

        ai_message, remaining_tool_calls = await _model_step(
            llm_with_tools=llm_with_tools,
            messages=messages,
            tool_map=tool_map,
            logger=logger,
            remaining_tool_calls=remaining_tool_calls,
            verifier_hook=verifier_hook,
        )
        final_ai_messages.append(ai_message)

    return final_ai_messages


__all__ = ["execute_scenario", "extract_ai_message_content"]
