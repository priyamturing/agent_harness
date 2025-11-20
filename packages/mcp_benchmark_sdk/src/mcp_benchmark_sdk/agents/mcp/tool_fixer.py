"""Tool schema fixer for Python reserved keywords."""

from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, create_model


RESERVED_KEYWORDS = {
    "self", "class", "def", "return", "if", "else", "elif", "while", "for",
    "in", "is", "and", "or", "not", "None", "True", "False", "import",
    "from", "as", "with", "try", "except", "finally", "raise", "yield",
    "lambda", "pass", "break", "continue", "global", "nonlocal", "del",
    "assert", "async", "await", "match", "case"
}


def fix_tool_schemas(tools: list[BaseTool]) -> list[BaseTool]:
    """Fix MCP tools that have Python reserved keywords as parameter names.

    Creates wrapper tools with safe parameter names (e.g., 'self' → 'self_')
    that translate parameters back to original names when calling MCP server.

    Args:
        tools: List of LangChain BaseTool instances from MCP servers

    Returns:
        List of tools where conflicting parameters have been wrapped with safe names
    """
    fixed_tools = []
    
    for tool in tools:
        if not hasattr(tool, "args_schema") or tool.args_schema is None:
            fixed_tools.append(tool)
            continue

        schema = tool.args_schema
        
        # StructuredTool can have dict or Pydantic schemas
        if isinstance(schema, dict):
            properties = schema.get("properties", {})
            
            has_reserved = any(field_name in RESERVED_KEYWORDS for field_name in properties)
            if not has_reserved:
                fixed_tools.append(tool)
                continue
            
            new_properties = {}
            field_mapping = {}
            
            for field_name, field_def in properties.items():
                if field_name in RESERVED_KEYWORDS:
                    new_name = f"{field_name}_"
                    field_mapping[field_name] = new_name
                    new_properties[new_name] = field_def
                else:
                    new_properties[field_name] = field_def
            
            new_schema_dict = {**schema, "properties": new_properties}
            
            if "required" in schema:
                new_required = [
                    field_mapping.get(f, f) for f in schema["required"]
                ]
                new_schema_dict["required"] = new_required
            
            tool.args_schema = new_schema_dict
            
            # Create wrapper for dict-schema tools
            original_tool_ref = tool
            reverse_mapping = {v: k for k, v in field_mapping.items()}
            
            if hasattr(original_tool_ref, "coroutine") and original_tool_ref.coroutine:  # type: ignore[attr-defined]
                original_coroutine = original_tool_ref.coroutine  # type: ignore[attr-defined]
                
                async def wrapper_func(**kwargs: Any) -> Any:
                    mcp_args = {}
                    for k, v in kwargs.items():
                        original_name = reverse_mapping.get(k, k)
                        mcp_args[original_name] = v
                    return await original_coroutine(**mcp_args)
                
                tool.coroutine = wrapper_func  # type: ignore[attr-defined]
            
            fixed_tools.append(tool)
            continue
        
        if not isinstance(schema, type) or not issubclass(schema, BaseModel):
            fixed_tools.append(tool)
            continue

        schema_fields = schema.model_fields
        has_reserved = any(field_name in RESERVED_KEYWORDS for field_name in schema_fields)

        if not has_reserved:
            fixed_tools.append(tool)
            continue

        new_fields = {}
        field_mapping = {}

        for field_name, field_info in schema_fields.items():
            if field_name in RESERVED_KEYWORDS:
                new_name = f"{field_name}_"
                field_mapping[field_name] = new_name
                new_fields[new_name] = (field_info.annotation, field_info)
            else:
                new_fields[field_name] = (field_info.annotation, field_info)

        new_schema = create_model(
            f"{schema.__name__}_Fixed",
            **new_fields,
        )

        original_tool_ref = tool
        reverse_mapping = {v: k for k, v in field_mapping.items()}

        if hasattr(original_tool_ref, "coroutine") and original_tool_ref.coroutine:  # type: ignore[attr-defined]
            original_coroutine = original_tool_ref.coroutine  # type: ignore[attr-defined]
            
            async def wrapper_func(**kwargs: Any) -> Any:
                """Wrapper that translates safe parameter names back to original names."""
                mcp_args = {}
                for k, v in kwargs.items():
                    original_name = reverse_mapping.get(k, k)
                    mcp_args[original_name] = v
                
                return await original_coroutine(**mcp_args)
        else:
            async def wrapper_func(**kwargs: Any) -> Any:
                """Wrapper for tools without coroutine."""
                mcp_args = {}
                for k, v in kwargs.items():
                    original_name = reverse_mapping.get(k, k)
                    mcp_args[original_name] = v
                
                return await original_tool_ref.ainvoke(mcp_args)

        wrapped_tool = StructuredTool.from_function(
            coroutine=wrapper_func,
            name=tool.name,
            description=tool.description,
            args_schema=new_schema,
            return_direct=tool.return_direct,
            verbose=tool.verbose,
            handle_tool_error=tool.handle_tool_error,
        )

        fixed_tools.append(wrapped_tool)

    return fixed_tools
