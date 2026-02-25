"""Simple cost estimation for LLM API calls."""

from typing import Any

import tiktoken

# Estimated output tokens for structured responses
OUTPUT_TOKENS = 150

# Pricing per 1M tokens: (input_price, output_price)
MODEL_PRICING = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.150, 0.600),
    "gpt-3.5-turbo": (0.50, 1.50),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "deepseek-chat": (0.14, 0.28),
    "default": (1.00, 3.00),
}


def estimate_cost(prompt: str, llm_config: dict[str, Any]) -> float:
    """
    Estimate and log the cost of an LLM API call.

    Args:
        prompt: The prompt text to be sent
        llm_config: LLM configuration dict with 'provider' and 'model'

    Returns:
        Estimated cost in USD
    """
    model = llm_config.get("model", "unknown")

    # Count input tokens
    try:
        encoding = tiktoken.encoding_for_model(model)
        input_tokens = len(encoding.encode(prompt))
    except KeyError:
        # Fallback: ~4 characters per token
        input_tokens = len(prompt) // 4

    # Get pricing
    input_price, output_price = MODEL_PRICING.get(model, MODEL_PRICING["default"])

    # Calculate cost
    input_cost = (input_tokens / 1_000_000) * input_price
    output_cost = (OUTPUT_TOKENS / 1_000_000) * output_price
    total_cost = input_cost + output_cost

    return total_cost
