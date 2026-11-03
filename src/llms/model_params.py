"""
Generation parameters per model family.

Reasoning models (gpt-5*, gpt-6*, o1/o3/o4) don't accept `max_tokens` nor a custom
`temperature`: they use `max_completion_tokens` (which also includes reasoning tokens).
"""

REASONING_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4")

# Low effort keeps reasoning tokens from eating the completion budget
DEFAULT_REASONING_EFFORT = "low"


def is_reasoning_model(model: str) -> bool:
    """True if the model (or Azure deployment name) belongs to a reasoning family."""
    return (model or "").lower().startswith(REASONING_PREFIXES)


def generation_params(*, model: str, max_tokens: int, temperature: float) -> dict:
    """Returns the token/temperature params accepted by the given model."""
    if is_reasoning_model(model):
        return {
            "max_completion_tokens": max_tokens,
            "reasoning_effort": DEFAULT_REASONING_EFFORT,
        }
    return {"max_tokens": max_tokens, "temperature": temperature}
