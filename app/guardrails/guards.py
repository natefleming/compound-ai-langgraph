from typing import Tuple
from guardrails import Guard, OnFailAction
from app.guardrails.validators import HallucinationValidator, TopicValidator, PIIValidator

def pii_guard() -> Guard:
    """Creates a Guard instance for PII validation.

    Returns:
        Guard: A Guard instance configured with PIIValidator.
    """
    guard = Guard(name='pii_guard').use(
        PIIValidator(
            on_fail=OnFailAction.FIX
        ),
    )
    return guard

def topic_guard(banned_topics: Tuple[str, ...]) -> Guard:
    """Creates a Guard instance for topic validation.

    Args:
        banned_topics (Tuple[str, ...]): A tuple of banned topics.

    Returns:
        Guard: A Guard instance configured with TopicValidator.

    Raises:
        ValueError: If banned_topics is empty.
    """
    if not banned_topics:
        raise ValueError("banned_topics must not be empty")
        
    guard = Guard(name='topic_guard').use(
        TopicValidator(
            banned_topics=banned_topics,
            on_fail=OnFailAction.FIX,
        ),
    )
    return guard

def hallucination_guard() -> Guard:
    """Creates a Guard instance for hallucination validation.

    Returns:
        Guard: A Guard instance configured with HallucinationValidator.
    """
    guard = Guard().use(
        HallucinationValidator(
            name="hallucination_guard",
            embedding_model='all-MiniLM-L6-v2',
            entailment_model='GuardrailsAI/finetuned_nli_provenance',
            sources=[],
            on_fail=OnFailAction.FIX
        )
    )
    return guard