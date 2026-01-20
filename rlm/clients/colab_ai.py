"""
Colab AI Client for RLM.

This client uses Google Colab's built-in AI library (google.colab.ai) to access
Gemini models without requiring an API key. This is only available when running
inside Google Colab notebooks.

Usage in Colab:
    from rlm import RLM

    rlm = RLM(
        backend="colab_ai",
        backend_kwargs={"model_name": "gemini-2.5-flash-lite"},
        environment="local",
        verbose=True,
    )

    result = rlm.completion("Your prompt here")
"""

from collections import defaultdict
from typing import Any

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary


class ColabAIClient(BaseLM):
    """
    LM Client for running models with Google Colab's built-in AI library.
    No API key required - only works within Google Colab environment.

    Available models (as of 2025):
    - gemini-2.5-flash-lite (default, free tier)
    - gemini-2.5-flash (free tier)
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash-lite",
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)

        # Try to import google.colab.ai
        try:
            from google.colab import ai

            self.ai = ai
        except ImportError as e:
            raise ImportError(
                "google.colab.ai is not available. "
                "This client only works within Google Colab notebooks. "
                "Please use 'gemini' backend with an API key for non-Colab environments."
            ) from e

        self.model_name = model_name

        # Per-model usage tracking
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)

        # Last call tracking (Colab AI doesn't provide token counts, so we estimate)
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0

    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        """Generate a completion using Colab AI."""
        model = model or self.model_name
        prompt_text = self._prepare_prompt(prompt)

        # Call Colab AI
        response = self.ai.generate_text(
            prompt_text,
            model_name=f"google/{model}",
        )

        # Handle None response (API error or rate limit)
        if response is None:
            raise RuntimeError(
                "Colab AI returned None. This may be due to rate limiting, "
                "content filtering, or a temporary API issue. Please try again."
            )

        self._track_usage(prompt_text, response, model)
        return response

    async def acompletion(
        self, prompt: str | list[dict[str, Any]], model: str | None = None
    ) -> str:
        """
        Async completion using Colab AI.
        Note: Colab AI doesn't have native async support, so this wraps the sync call.
        """
        # Colab AI doesn't have async support, use sync version
        return self.completion(prompt, model)

    def _prepare_prompt(self, prompt: str | list[dict[str, Any]]) -> str:
        """Convert prompt to string format for Colab AI."""
        if isinstance(prompt, str):
            return prompt

        if isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
            # Convert OpenAI-style messages to a single string
            parts = []
            for msg in prompt:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                if role == "system":
                    parts.append(f"System: {content}")
                elif role == "user":
                    parts.append(f"User: {content}")
                elif role == "assistant":
                    parts.append(f"Assistant: {content}")
                else:
                    parts.append(content)

            return "\n\n".join(parts)

        raise ValueError(f"Invalid prompt type: {type(prompt)}")

    def _track_usage(self, prompt: str, response: str, model: str):
        """Track usage statistics (estimated, as Colab AI doesn't provide token counts)."""
        self.model_call_counts[model] += 1

        # Estimate tokens (rough approximation: ~4 chars per token)
        estimated_input = len(prompt) // 4
        estimated_output = len(response) // 4

        self.model_input_tokens[model] += estimated_input
        self.model_output_tokens[model] += estimated_output

        self.last_prompt_tokens = estimated_input
        self.last_completion_tokens = estimated_output

    def get_usage_summary(self) -> UsageSummary:
        """Get usage summary for all model calls."""
        model_summaries = {}
        for model in self.model_call_counts:
            model_summaries[model] = ModelUsageSummary(
                total_calls=self.model_call_counts[model],
                total_input_tokens=self.model_input_tokens[model],
                total_output_tokens=self.model_output_tokens[model],
            )
        return UsageSummary(model_usage_summaries=model_summaries)

    def get_last_usage(self) -> ModelUsageSummary:
        """Get usage summary for the last model call."""
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=self.last_prompt_tokens,
            total_output_tokens=self.last_completion_tokens,
        )
