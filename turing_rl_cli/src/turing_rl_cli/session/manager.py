"""Session persistence for benchmark runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from turing_rl_sdk.harness.orchestrator import RunResult, ModelResultFile


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
        run_result: RunResult,
        scenario_id: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Path:
        """Save a single result to the session directory.

        Args:
            session_dir: Session directory
            run_result: RunResult from harness containing agent result and verifiers
            scenario_id: Scenario ID for filename
            metadata: Additional metadata (merged with run_result metadata)

        Returns:
            Path to saved file
        """
        result_dict = run_result.to_dict()
        
        scenario_name = result_dict.get("scenario_name", scenario_id)
        
        artifact = {
            "conversation": result_dict["conversation"],
            "run_label": f"{run_result.model}-{scenario_id}",
            "database_id": result_dict.get("database_id"),
            "provider": self._extract_provider_from_model(run_result.model),
            "model": run_result.model,
            "prompt_alias": scenario_name,
            "status": "completed" if run_result.success else "failed",
            "scenarios": [
                {
                    "scenario_id": scenario_name,
                    "verifiers": result_dict["verifier_results"],
                }
            ],
        }
        
        if result_dict.get("reasoning_traces"):
            artifact["reasoning_traces"] = result_dict["reasoning_traces"]
        if run_result.error:
            artifact["error"] = run_result.error
        if run_result.result and run_result.result.langsmith_url:
            artifact["langsmith_url"] = run_result.result.langsmith_url
        if run_result.result and run_result.result.langfuse_url:
            artifact["langfuse_url"] = run_result.result.langfuse_url
        
        combined_metadata = {**result_dict.get("metadata", {})}
        if metadata:
            combined_metadata.update(metadata)
        if combined_metadata:
            artifact["metadata"] = combined_metadata
            
        artifact["timestamp"] = datetime.now(timezone.utc).isoformat()

        filename = f"{scenario_id}_{run_result.model}.json"
        safe_filename = "".join(c if c.isalnum() or c in "-_." else "_" for c in filename)
        filepath = session_dir / safe_filename

        with filepath.open("w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2, ensure_ascii=False)

        return filepath

    def save_model_result(
        self,
        session_dir: Path,
        model_result: ModelResultFile,
    ) -> Path:
        """Persist aggregated model results for a harness file."""
        filename = f"{model_result.harness_name}_{model_result.model_name}.json"
        safe_filename = "".join(
            c if c.isalnum() or c in "-_." else "_" for c in filename
        ).strip("_")
        safe_filename = safe_filename or "results.json"
        filepath = session_dir / safe_filename

        with filepath.open("w", encoding="utf-8") as f:
            json.dump(model_result.payload, f, indent=2, ensure_ascii=False)

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
