from typing import Tuple

from guardrails import Guard, OnFailAction

from app.guardrails.validators import HallucinationValidator, TopicValidator, PIIValidator


def pii_guard() -> Guard:
    guard = Guard(name='pii_guard').use(
        PIIValidator(
            on_fail=OnFailAction.EXCEPTION
        ),
    )
    return guard


def topic_guard(banned_topics: Tuple[str, ...]) -> Guard:
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
    guard = Guard().use(
        HallucinationValidator(
            name="hallucination_guard",
            embedding_model='all-MiniLM-L6-v2',
            entailment_model='GuardrailsAI/finetuned_nli_provenance',
            sources=['The sun rises in the east and sets in the west.', 'The sun is hot.'],
            on_fail=OnFailAction.EXCEPTION
        )
    )
    return guard