import re
from unidecode import unidecode
from pandas import DataFrame
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from abc import ABC, abstractmethod
from typing import List

class AbstractPreprocessor(ABC):
    data: DataFrame
    clean_data: DataFrame

    def set_data(self, data: DataFrame) -> "AbstractPreprocessor":
        self.data = data
        return self

    def get_data(self) -> DataFrame:
        return self.clean_data

    @abstractmethod
    def preprocess(self) -> "AbstractPreprocessor":
        ...

class AbstractTextStep(ABC):
    def call(self, x: str) -> str:
        ...


class LowerStep(AbstractTextStep):
    def call(self, x: str) -> str:
        return x.lower()

class UnicodeStep(AbstractTextStep):
    def call(self, x: str) -> str:
        return unidecode(x)

class WordTokenStep(AbstractTextStep):
    def __init__(self, lang: str="english"):
        self.lang = lang

    def call(self, x: str) -> str:
        tokens = word_tokenize(x, language=self.lang)
        return " ".join(tokens)

class StopwordStep(AbstractTextStep):
    def __init__(self, lang: str="english"):
        self.sw = stopwords.words(lang)

    def call(self, x: str) -> str:
        tokens = x.split(" ")
        filtered_tokens = filter(lambda token: token not in self.sw, tokens)
        return " ".join(filtered_tokens)

class LenFilterStep(AbstractTextStep):
    def __init__(self, min_len: int, max_len: int):
        self.min_len = min_len
        self.max_len = max_len

    def call(self, x: str) -> str:
        tokens = x.split(" ")
        filtered_tokens = filter(
                lambda token: len(token) >= self.min_len and len(token) <= self.max_len,
                tokens
                )
        return " ".join(filtered_tokens)

class AbstractRegexStep(AbstractTextStep, ABC):

    @abstractmethod
    def get_pattern(self) -> re.Pattern:
        ...

    def call(self, x: str) -> str:
        pat = self.get_pattern()
        return re.sub(pat, ' ', x)

class UrlRemovalStep(AbstractRegexStep):
    def get_pattern(self) -> re.Pattern:
        return re.compile(r"https?://[^\s]+ ")

class SpRemovalStep(AbstractRegexStep):
    def get_pattern(self) -> re.Pattern:
        return re.compile(r"[^a-z ]")

class DupSpacesStep(AbstractRegexStep):
    def get_pattern(self) -> re.Pattern:
        return re.compile(r"\s+")

class TextPreprocessor(AbstractPreprocessor):
    def __init__(self, column: str, steps: List[AbstractTextStep]):
        self.column = column
        self.steps = steps

    def preprocess(self):
        def clean_fn(x: str) -> str:
            for step in self.steps:
                x = step.call(x)
            return x
        self.clean_data = self.data.assign(**{self.column: lambda df: df[self.column].apply(clean_fn)})
        return self
