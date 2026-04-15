"""Tests for gateway pre-agent session hygiene thresholds."""

from gateway.run import (
    _GATEWAY_HYGIENE_HARD_MSG_LIMIT,
    _GATEWAY_HYGIENE_HARD_PROMPT_TOKEN_LIMIT,
    _gateway_hygiene_token_threshold,
    _should_gateway_hygiene_compress,
)


def test_large_context_models_use_practical_hard_prompt_cap():
    """Very large model windows still compress before replay becomes pathological."""
    threshold = _gateway_hygiene_token_threshold(1_050_000, 0.85)
    assert threshold == _GATEWAY_HYGIENE_HARD_PROMPT_TOKEN_LIMIT


def test_smaller_context_models_keep_relative_threshold():
    """Normal-size context windows should still use the relative threshold."""
    threshold = _gateway_hygiene_token_threshold(200_000, 0.85)
    assert threshold == 170_000


def test_prompt_tokens_over_practical_cap_trigger_hygiene_compression():
    """A long-lived large-context session should compress before 85% of 1M context."""
    needs_compress, effective_threshold = _should_gateway_hygiene_compress(
        approx_tokens=310_600,
        msg_count=231,
        context_length=1_050_000,
        threshold_pct=0.85,
    )

    assert effective_threshold == _GATEWAY_HYGIENE_HARD_PROMPT_TOKEN_LIMIT
    assert needs_compress is True


def test_hard_message_limit_still_forces_compression():
    """Runaway message growth remains an independent safety valve."""
    needs_compress, effective_threshold = _should_gateway_hygiene_compress(
        approx_tokens=10_000,
        msg_count=_GATEWAY_HYGIENE_HARD_MSG_LIMIT,
        context_length=1_050_000,
        threshold_pct=0.85,
    )

    assert effective_threshold == _GATEWAY_HYGIENE_HARD_PROMPT_TOKEN_LIMIT
    assert needs_compress is True
