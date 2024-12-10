from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from guardrails import Guard, OnFailAction
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)

@register_validator(name="pii_validator", data_type="string")
class PIIValidator(Validator):
    """Validator to detect Personally Identifiable Information (PII) in text."""

    DEFAULT_ENTITIES: List[str] = [
        "PERSON",
        "CREDIT_CARD", 
        "PHONE_NUMBER", 
        "IP_ADDRESS", 
        "US_SSN", 
        "US_PASSPORT", 
        "US_BANK_NUMBER",
        "US_ITIN",
        "US_DRIVER_LICENSE",
        "EMAIL_ADDRESS",
        "CRYPTO",
    ]
    
    def __init__(
        self, 
        entities: List[str] = [], 
        language: str = "en", 
        **kwargs
    ) -> None:
        """
        Initializes the PIIValidator.

        Args:
            entities (List[str]): List of entities to detect.
            language (str): Language of the text.
            **kwargs: Additional arguments.
        """
        from presidio_analyzer import AnalyzerEngine
        
        self.pii_analyzer = AnalyzerEngine()
        self.entities = entities or PIIValidator.DEFAULT_ENTITIES
        self.language = language
        super().__init__(**kwargs)

    def _validate(
        self,
        value: Any,
        metadata: Dict[str, Any] = {}
    ) -> ValidationResult:
        """
        Validates the text for PII.

        Args:
            value (Any): The text to validate.
            metadata (Dict[str, Any]): Additional metadata.

        Returns:
            ValidationResult: Result of the validation.
        """
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
    ) -> List[str]:
        """
        Detects PII in the text.

        Args:
            text (str): The text to analyze.

        Returns:
            List[str]: List of detected PII entities.
        """
        result = self.pii_analyzer.analyze(
            text,
            language=self.language,
            entities=self.entities
        )
        return [entity.entity_type for entity in result]



@register_validator(name="topic_validator", data_type="string")
class TopicValidator(Validator):
    """Validator to detect banned topics in text."""
    
    CLASSIFIER = pipeline(
        "zero-shot-classification",
        model='facebook/bart-large-mnli',
        hypothesis_template="This sentence above contains discussions of the following topics: {}.",
        multi_label=True,
    )

    def __init__(
        self,
        banned_topics: Optional[List[str]] = ["politics"],
        threshold: float = 0.8,
        **kwargs
    ):
        """
        Initializes the TopicValidator.

        Args:
            banned_topics (Optional[List[str]]): List of banned topics.
            threshold (float): Threshold for topic detection.
            **kwargs: Additional arguments.
        """
        self.topics = banned_topics
        self.threshold = threshold
        super().__init__(**kwargs)

    def _validate(
        self, value: str, metadata: Optional[Dict[str, str]] = None
    ) -> ValidationResult:
        """
        Validates the text for banned topics.

        Args:
            value (str): The text to validate.
            metadata (Optional[Dict[str, str]]): Additional metadata.

        Returns:
            ValidationResult: Result of the validation.
        """
        detected_topics = self.detect_topics(value, self.topics, self.threshold)
        if detected_topics:
            return FailResult(
                error_message="The text contains the following banned topics: "
                              f"{detected_topics}",
            )
        return PassResult()

    def detect_topics(
        self,
        text: str,
        topics: List[str],
        threshold: float = 0.8
    ) -> List[str]:
        """
        Detects topics in the text.

        Args:
            text (str): The text to analyze.
            topics (List[str]): List of topics to detect.
            threshold (float): Threshold for topic detection.

        Returns:
            List[str]: List of detected topics.
        """
        result = TopicValidator.CLASSIFIER(text, topics)
        return [topic
                for topic, score in zip(result["labels"], result["scores"])
                if score > threshold]
    
@register_validator(name="hallucination_validator", data_type="string")
class HallucinationValidator(Validator):
    """Validator to detect hallucinated content in text."""
    
    def __init__(
            self, 
            embedding_model: Optional[str] = None,
            entailment_model: Optional[str] = None,
            sources: Optional[List[str]] = None,
            **kwargs
        ):
        """
        Initializes the HallucinationValidator.

        Args:
            embedding_model (Optional[str]): Embedding model for sentence embeddings.
            entailment_model (Optional[str]): Model for entailment checking.
            sources (Optional[List[str]]): List of source texts.
            **kwargs: Additional arguments.
        """
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
        """
        Validates the text for hallucinated content.

        Args:
            value (str): The text to validate.
            metadata (Optional[Dict[str, str]]): Additional metadata.

        Returns:
            ValidationResult: Result of the validation.
        """
        sentences = self.split_sentences(value)
        relevant_sources = self.find_relevant_sources(sentences, self.sources)

        entailed_sentences = []
        hallucinated_sentences = []
        for sentence in sentences:
            is_entailed = self.check_entailment(sentence, relevant_sources)
            if not is_entailed:
                hallucinated_sentences.append(sentence)
            else:
                entailed_sentences.append(sentence)
        
        if hallucinated_sentences:
            return FailResult(
                error_message=f"The following sentences are hallucinated: {hallucinated_sentences}",
            )
        
        return PassResult()

    def split_sentences(self, text: str) -> List[str]:
        """
        Splits the text into sentences.

        Args:
            text (str): The text to split.

        Returns:
            List[str]: List of sentences.
        """
        import nltk
        if nltk is None:
            raise ImportError(
                "This validator requires the `nltk` package. "
                "Install it with `pip install nltk`, and try again."
            )
        return nltk.sent_tokenize(text)

    def find_relevant_sources(self, sentences: List[str], sources: List[str]) -> List[str]:
        """
        Finds relevant sources for the given sentences.

        Args:
            sentences (List[str]): List of sentences.
            sources (List[str]): List of source texts.

        Returns:
            List[str]: List of relevant sources.
        """
        source_embeds = self.embedding_model.encode(sources)
        sentence_embeds = self.embedding_model.encode(sentences)

        relevant_sources = []

        for sentence_idx in range(len(sentences)):
            sentence_embed = sentence_embeds[sentence_idx, :].reshape(1, -1)
            cos_similarities = np.sum(np.multiply(source_embeds, sentence_embed), axis=1)
            top_sources = np.argsort(cos_similarities)[::-1][:5]
            top_sources = [i for i in top_sources if cos_similarities[i] > 0.8]
            relevant_sources.extend([sources[i] for i in top_sources])

        return relevant_sources
    
    def check_entailment(self, sentence: str, sources: List[str]) -> bool:
        """
        Checks if the sentence is entailed by any of the sources.

        Args:
            sentence (str): The sentence to check.
            sources (List[str]): List of source texts.

        Returns:
            bool: True if the sentence is entailed, False otherwise.
        """
        for source in sources:
            output = self.nli_pipeline({'text': source, 'text_pair': sentence})
            if output['label'] == 'entailment':
                return True
        return False