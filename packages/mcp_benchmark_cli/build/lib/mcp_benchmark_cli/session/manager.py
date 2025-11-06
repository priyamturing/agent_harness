"""Session persistence for benchmark runs."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from mcp_benchmark_sdk import Result


class SessionManager:
    """Manages saving and loading benchmark run sessions."""

    def __init__(self, results_root: Union[str, Path] = "results"):
        """Initialize session manager.

        Args:
            results_root: Root directory for results
        """
        self.results_root = Path(results_root)
        self.results_root.mkdir(exist_ok=True)

    def create_session_dir(self, name: str) -> Path:
        """Create a new session directory.

        Args:
            name: Session name (will be sanitized)

        Returns:
            Path to session directory
        """
        # Sanitize name
        safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
        safe_name = safe_name.strip("_") or "session"

        # Find unique directory
        counter = 1
        while True:
            if counter == 1:
                dir_name = safe_name
            else:
                dir_name = f"{safe_name}_{counter}"

            session_dir = self.results_root / dir_name
            if not session_dir.exists():
                session_dir.mkdir(parents=True)
                return session_dir

            counter += 1

    def _extract_conversation_history(self, messages: list[Any]) -> list[dict[str, Any]]:
        """Extract conversation history from result messages.
        
        Args:
            messages: List of message objects from Result
            
        Returns:
            List of conversation entries with role and content
        """
        if not messages:
            return []
        
        conversation = []
        for msg in messages:
            entry: dict[str, Any] = {
                "role": msg.type if hasattr(msg, "type") else "unknown",
                "content": msg.content if hasattr(msg, "content") else str(msg),
            }
            
            # Add additional metadata if available
            if hasattr(msg, "additional_kwargs") and msg.additional_kwargs:
                entry["additional_kwargs"] = msg.additional_kwargs
            
            # Add tool calls if present (AIMessage)
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None),
                        "name": tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None),
                        "args": tc.get("args") if isinstance(tc, dict) else getattr(tc, "arguments", None),
                    }
                    for tc in msg.tool_calls
                ]
            
            # Add tool call ID if present (ToolMessage)
            if hasattr(msg, "tool_call_id") and msg.tool_call_id:
                entry["tool_call_id"] = msg.tool_call_id
            
            # Add name if present (ToolMessage)
            if hasattr(msg, "name") and msg.name:
                entry["name"] = msg.name
            
            conversation.append(entry)
        
        return conversation

    def save_result(
        self,
        session_dir: Path,
        result: Result,
        model_name: str,
        scenario_id: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Path:
        """Save a single result to the session directory.

        Args:
            session_dir: Session directory
            result: Result object
            model_name: Model name
            scenario_id: Scenario ID
            metadata: Additional metadata

        Returns:
            Path to saved file
        """
        # Extract conversation history from messages
        conversation_history = self._extract_conversation_history(result.messages)
        
        # Build artifact
        artifact = {
            "model": model_name,
            "scenario_id": scenario_id,
            "success": result.success,
            "database_id": result.database_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                **(result.metadata or {}),
                **(metadata or {}),
            },
            "reasoning_traces": result.reasoning_traces,
            "conversation_history": conversation_history,
            "verifier_results": [
                {
                    "name": vr.name,
                    "success": vr.success,
                    "expected": vr.expected_value,
                    "actual": vr.actual_value,
                    "comparison": vr.comparison_type,
                    "error": vr.error,
                }
                for vr in result.verifier_results
            ]
            if result.verifier_results
            else [],
            "error": result.error,
        }

        # Save to file
        filename = f"{scenario_id}_{model_name}.json"
        safe_filename = "".join(c if c.isalnum() or c in "-_." else "_" for c in filename)
        filepath = session_dir / safe_filename

        with filepath.open("w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2, ensure_ascii=False)

        return filepath

    def save_session_manifest(
        self,
        session_dir: Path,
        runs: list[dict[str, Any]],
        name: Optional[str] = None,
    ) -> Path:
        """Save session manifest.

        Args:
            session_dir: Session directory
            runs: List of run summaries
            name: Session name

        Returns:
            Path to manifest file
        """
        manifest = {
            "session_name": name or session_dir.name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "runs": runs,
        }

        manifest_path = session_dir / "session.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        return manifest_path

