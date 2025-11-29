"""Core chatbot utilities shared between the GUI and CLI interfaces."""

from dataclasses import dataclass
import json
import pickle
import random
from pathlib import Path
from typing import List, Sequence

import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from keras.models import load_model


@dataclass
class IntentPrediction:
    """Represents a single intent prediction with its probability."""

    intent: str
    probability: float


class Chatbot:
    """Encapsulates the model and helper utilities for generating responses."""

    def __init__(
        self,
        model_path: str | Path = "chatbot_model.h5",
        intents_path: str | Path = "intents.json",
        words_path: str | Path = "words.pkl",
        classes_path: str | Path = "classes.pkl",
        error_threshold: float = 0.25,
        default_response: str | None = "I did not understand. Could you rephrase?",
    ) -> None:
        self.model = load_model(model_path)
        self.intents = json.loads(Path(intents_path).read_text())
        self.words: List[str] = pickle.loads(Path(words_path).read_bytes())
        self.classes: List[str] = pickle.loads(Path(classes_path).read_bytes())
        self.error_threshold = error_threshold
        self.default_response = default_response
        self.lemmatizer = WordNetLemmatizer()

    def clean_up_sentence(self, sentence: str) -> List[str]:
        """Tokenize and lemmatize the input sentence."""

        sentence_words = nltk.word_tokenize(sentence)
        return [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]

    def bow(self, sentence: str) -> np.ndarray:
        """Create a bag-of-words vector for the provided sentence."""

        sentence_words = self.clean_up_sentence(sentence)
        bag = [1 if word in sentence_words else 0 for word in self.words]
        return np.array(bag)

    def predict_class(self, sentence: str) -> List[IntentPrediction]:
        """Predict the intent of the provided sentence."""

        bag = self.bow(sentence)
        probabilities: Sequence[float] = self.model.predict(np.array([bag]), verbose=0)[0]
        results = [
            IntentPrediction(intent=self.classes[index], probability=probability)
            for index, probability in enumerate(probabilities)
            if probability > self.error_threshold
        ]
        return sorted(results, key=lambda prediction: prediction.probability, reverse=True)

    def _response_for_intent(self, intent_tag: str) -> str | None:
        for intent in self.intents.get("intents", []):
            if intent.get("tag") == intent_tag:
                responses = intent.get("responses", [])
                return random.choice(responses) if responses else None
        return None

    def chatbot_response(self, message: str) -> str:
        """Generate a chatbot response for the given user message."""

        predictions = self.predict_class(message)
        if not predictions:
            return self.default_response or ""

        top_intent = predictions[0].intent
        response = self._response_for_intent(top_intent)
        return response if response is not None else self.default_response or ""
