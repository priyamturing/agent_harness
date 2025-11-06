"""Session persistence for benchmark runs."""

from __future__ import annotations

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

    def _extract_provider_from_model(self, model_name: str) -> str:
        """Extract provider name from model name.
        
        Args:
            model_name: Full model name
            
        Returns:
            Provider name (anthropic, openai, google, xai, etc.)
        """
        model_lower = model_name.lower()
        if "claude" in model_lower or "anthropic" in model_lower:
            return "anthropic"
        elif "gpt" in model_lower or "openai" in model_lower or "o1" in model_lower:
            return "openai"
        elif "gemini" in model_lower or "google" in model_lower:
            return "google"
        elif "grok" in model_lower or "xai" in model_lower:
            return "xai"
        elif "qwen" in model_lower:
            return "qwen"
        else:
            return "unknown"

    def save_result(
        self,
        session_dir: Path,
        result: Result,
        model_name: str,
        scenario_id: str,
        metadata: Optional[dict[str, Any]] = None,
        verifier_results: Optional[list[Any]] = None,
    ) -> Path:
        """Save a single result to the session directory.

        Args:
            session_dir: Session directory
            result: Result object (agent execution result)
            model_name: Model name
            scenario_id: Scenario ID
            metadata: Additional metadata
            verifier_results: List of VerifierResult objects from harness

        Returns:
            Path to saved file
        """
        # Extract conversation history using SDK method
        conversation = result.get_conversation_history()
        
        # Build verifier results
        verifier_results_json = [
            {
                "name": vr.name,
                "success": vr.success,
                "expected": vr.expected_value,
                "actual": vr.actual_value,
                "comparison": vr.comparison_type,
                "error": vr.error,
            }
            for vr in verifier_results
        ] if verifier_results else []
        
        # Build artifact in OLD jira_mcp_benchmark format
        # conversation must be FIRST!
        artifact = {
            "conversation": conversation,
            "run_label": f"{model_name}-{scenario_id}",
            "database_id": result.database_id,
            "provider": self._extract_provider_from_model(model_name),
            "model": model_name,
            "prompt_alias": metadata.get("scenario_name", scenario_id) if metadata else scenario_id,
            "status": "completed" if result.success else "failed",
            "scenarios": [
                {
                    "scenario_id": metadata.get("scenario_name", scenario_id) if metadata else scenario_id,
                    "verifiers": verifier_results_json,
                }
            ],
        }
        
        # Add optional fields
        if result.reasoning_traces:
            artifact["reasoning_traces"] = result.reasoning_traces
        if result.error:
            artifact["error"] = result.error
        if result.langsmith_url:
            artifact["langsmith_url"] = result.langsmith_url
        if metadata:
            artifact["metadata"] = metadata
        artifact["timestamp"] = datetime.now(timezone.utc).isoformat()

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

