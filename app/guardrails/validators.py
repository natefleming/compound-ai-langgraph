from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from presidio_analyzer import AnalyzerEngine

from guardrails import Guard, OnFailAction
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)

@register_validator(name="pii_detector", data_type="string")
class PIIValidator(Validator):

    def __init__(
        self, 
        entities: List[str] = [], 
        language: str = "en", 
        **kwargs
    ) -> None:
        self.pii_analyzer = AnalyzerEngine()
        self.entities = entities or ["PERSON", "PHONE_NUMBER"]
        self.language = language
        super().__init__(**kwargs)

    def _validate(
        self,
        value: Any,
        metadata: Dict[str, Any] = {}
    ) -> ValidationResult:
        detected_pii = self.detect_pii(value)
        if detected_pii:
            return FailResult(
                error_message=f"PII detected: {', '.join(detected_pii)}",
                metadata={"detected_pii": detected_pii},
            )
        return PassResult(message="No PII detected")
    
    def detect_pii(
        self,
        text: str
    ) -> list[str]:
        result = self.pii_analyzer.analyze(
            text,
            language=self.language,
            entities=self.entities
        )
        return [entity.entity_type for entity in result]

CLASSIFIER = pipeline(
    "zero-shot-classification",
    model='facebook/bart-large-mnli',
    hypothesis_template="This sentence above contains discussions of the folllowing topics: {}.",
    multi_label=True,
)

@register_validator(name="constrain_topic", data_type="string")
class TopicValidator(Validator):
    def __init__(
        self,
        banned_topics: Optional[list[str]] = ["politics"],
        threshold: float = 0.8,
        **kwargs
    ):
        self.topics = banned_topics
        self.threshold = threshold
        super().__init__(**kwargs)

    def _validate(
        self, value: str, metadata: Optional[dict[str, str]] = None
    ) -> ValidationResult:
        detected_topics = self.detect_topics(value, self.topics, self.threshold)
        if detected_topics:
            return FailResult(error_message="The text contains the following banned topics: "
                        f"{detected_topics}",
            )

        return PassResult()

    def detect_topics(
        self,
        text: str,
        topics: list[str],
        threshold: float = 0.8
    ) -> list[str]:
        result = CLASSIFIER(text, topics)
        return [topic
                for topic, score in zip(result["labels"], result["scores"])
                if score > threshold]
    
@register_validator(name="hallucination_detector", data_type="string")
class HallucinationValidator(Validator):
    def __init__(
            self, 
            embedding_model: Optional[str] = None,
            entailment_model: Optional[str] = None,
            sources: Optional[List[str]] = None,
            **kwargs
        ):
        if embedding_model is None:
            embedding_model = 'all-MiniLM-L6-v2'
        self.embedding_model = SentenceTransformer(embedding_model)

        self.sources = sources
        
        if entailment_model is None:
            entailment_model = 'GuardrailsAI/finetuned_nli_provenance'
        self.nli_pipeline = pipeline("text-classification", model=entailment_model)

        super().__init__(**kwargs)

    def validate(
        self, value: str, metadata: Optional[Dict[str, str]] = None
    ) -> ValidationResult:
        # Split the text into sentences
        sentences = self.split_sentences(value)

        # Find the relevant sources for each sentence
        relevant_sources = self.find_relevant_sources(sentences, self.sources)

        entailed_sentences = []
        hallucinated_sentences = []
        for sentence in sentences:
            # Check if the sentence is entailed by the sources
            is_entailed = self.check_entailment(sentence, relevant_sources)
            if not is_entailed:
                hallucinated_sentences.append(sentence)
            else:
                entailed_sentences.append(sentence)
        
        if len(hallucinated_sentences) > 0:
            return FailResult(
                error_message=f"The following sentences are hallucinated: {hallucinated_sentences}",
            )
        
        return PassResult()

    def split_sentences(self, text: str) -> List[str]:
        import nltk
        if nltk is None:
            raise ImportError(
                "This validator requires the `nltk` package. "
                "Install it with `pip install nltk`, and try again."
            )
        return nltk.sent_tokenize(text)

    def find_relevant_sources(self, sentences: str, sources: List[str]) -> List[str]:
        source_embeds = self.embedding_model.encode(sources)
        sentence_embeds = self.embedding_model.encode(sentences)

        relevant_sources = []

        for sentence_idx in range(len(sentences)):
            # Find the cosine similarity between the sentence and the sources
            sentence_embed = sentence_embeds[sentence_idx, :].reshape(1, -1)
            cos_similarities = np.sum(np.multiply(source_embeds, sentence_embed), axis=1)
            # Find the top 5 sources that are most relevant to the sentence that have a cosine similarity greater than 0.8
            top_sources = np.argsort(cos_similarities)[::-1][:5]
            top_sources = [i for i in top_sources if cos_similarities[i] > 0.8]

            # Return the sources that are most relevant to the sentence
            relevant_sources.extend([sources[i] for i in top_sources])

        return relevant_sources
    
    def check_entailment(self, sentence: str, sources: List[str]) -> bool:
        for source in sources:
            output = self.nli_pipeline({'text': source, 'text_pair': sentence})
            if output['label'] == 'entailment':
                return True
        return False