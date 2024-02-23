from abc import ABC, abstractmethod
import joblib
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sentan.eval import ModelEvaluator
from pathlib import Path
from typing import Dict

class AbstractModelBuilder(ABC):
    model: Pipeline

    def get_model(self) -> Pipeline:
        return self.model

    @abstractmethod
    def build(self) -> "AbstractModelBuilder":
        ...

class BowLogisticBuilder(AbstractModelBuilder):

    def build(self) -> "BowLogisticBuilder":
        self.model = Pipeline([
            ("vect", CountVectorizer()),
            ("clf", LogisticRegression())
            ])
        return self

class TfidfLogisticBuilder(AbstractModelBuilder):

    def build(self) -> "TfidfLogisticBuilder":
        self.model = Pipeline([
            ("vect", TfidfVectorizer()),
            ("clf", LogisticRegression())
            ])
        return self

class TfidfRandomForestBuilder(AbstractModelBuilder):
    def __init__(self, n_estimators: int, max_depth: int):
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def build(self) -> "TfidfLogisticBuilder":
        self.model = Pipeline([
            ("vect", TfidfVectorizer()),
            ("clf", RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                ))
            ])
        return self

class AbstractModelProcessor(ABC):
    model: Pipeline
    data: DataFrame

    def set_elements(self, model: Pipeline, data: DataFrame) -> "AbstractModelProcessor":
        self.model = model
        self.data = data
        return self

    def get_model(self) -> Pipeline:
        return self.model

    @abstractmethod
    def process(self) -> "AbstractModelProcessor":
        ...

class TrainModelProcessor(AbstractModelProcessor):
    def __init__(
        self,
        text_column: str,
        label_column: str,
        partition_column: str,
        model_path: Path,
        evaluator: ModelEvaluator,
    ):
        self.text_column = text_column
        self.label_column = label_column
        self.partition_column = partition_column
        self.model_path = model_path
        self.evaluator = evaluator

    def process(self) -> "AbstractModelProcessor":
        train_data = self.data.query(f"{self.partition_column} == 'train'")
        test_data = self.data.query(f"{self.partition_column} == 'test'")

        train_corpus, train_labels = train_data[self.text_column], train_data[self.label_column]
        test_corpus, test_labels = test_data[self.text_column], test_data[self.label_column]

        self.model.fit(train_corpus, train_labels)
        self.evaluator.evaluate(test_corpus, test_labels, self.model)

        joblib.dump(self.model, filename=self.model_path)
        return self
