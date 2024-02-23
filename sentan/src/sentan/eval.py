from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from pandas import Series, DataFrame
from typing import Dict

class AbstractMetric(ABC):
    model: Pipeline

    def set_model(self, model: Pipeline) -> "AbstractMetric":
        self.model = model
        return self

    @abstractmethod
    def call(self, data: Series, label: Series) -> float:
        ...

class AccuracyMetric(AbstractMetric):
    def call(self, data: Series, label: Series) -> float:
        y_pred = self.model.predict(data)
        return accuracy_score(label, y_pred)

class F1Metric(AbstractMetric):
    def __init__(self, average: str):
        self.average = average

    def call(self, data: Series, label: Series) -> float:
        y_pred = self.model.predict(data)
        return f1_score(label, y_pred, average=self.average)

class ModelEvaluator:
    def __init__(self, metrics: Dict[str, AbstractMetric]):
        self.metrics = metrics

    def evaluate(self, data: Series, label: Series, model: Pipeline):
        for name, metric in self.metrics.items():
            res = metric.set_model(model).call(data, label)
            print(f"{name}: {res:.4f}")
