"""LiteReason: latent reasoning via `<implicit_thought>N</implicit_thought>`.

LiteReason implements a latent reasoning mechanism for language models.
Instead of emitting explicit chain-of-thought text, the model learns to insert
latent "reasoning embeddings" at runtime when it encounters an
`<implicit_thought>N</implicit_thought>` marker.

The API is architecture-agnostic:
`AutoModelForCausalLMWithReasoning.from_pretrained(...)` patches any HF causal LM
instance in-place (Mistral/Gemma/Llama/etc.) while keeping `.generate()` intact.
"""

__version__ = "0.1.0"

# Token utilities
from .token_utils import (
    ImplicitThoughtDetector,
    get_implicit_thought_patterns,
)


# Lazy imports for modules that require transformers
def __getattr__(name):
    """Lazy import for model classes that require transformers."""
    if name == "AutoModelForCausalLMWithReasoning":
        from .causal_lm_with_reasoning import AutoModelForCausalLMWithReasoning
        return AutoModelForCausalLMWithReasoning
    elif name == "attach_litereason_to_causal_lm":
        from .causal_lm_with_reasoning import attach_litereason_to_causal_lm
        return attach_litereason_to_causal_lm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Generic API
    "AutoModelForCausalLMWithReasoning",
    "attach_litereason_to_causal_lm",
    # Token utilities
    "ImplicitThoughtDetector",
    "get_implicit_thought_patterns",
]
