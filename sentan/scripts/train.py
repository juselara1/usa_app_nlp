import nltk
from pathlib import Path
from sentan.data import PartitionedCsvLoader
from sentan.preprocessing import (
    TextPreprocessor,
    AbstractTextStep,
    LowerStep,
    UnicodeStep,
    WordTokenStep,
    StopwordStep,
    UrlRemovalStep,
    SpRemovalStep,
    DupSpacesStep,
    LenFilterStep
)
from sentan.model import (
    BowLogisticBuilder,
    TfidfLogisticBuilder,
    TfidfRandomForestBuilder,
    TrainModelProcessor
)
from sentan.eval import ModelEvaluator, AccuracyMetric, F1Metric

nltk.download("stopwords")
nltk.download("punkt")

data = PartitionedCsvLoader(raw_path=Path("data/raw/")).load().get_data()
clean_data = (
    TextPreprocessor(column="text", steps=[
        UnicodeStep(),
        LowerStep(),
        UrlRemovalStep(),
        WordTokenStep(),
        LenFilterStep(4, 15),
        StopwordStep(),
        SpRemovalStep(),
        DupSpacesStep()
    ])
    .set_data(data)
    .preprocess()
    .get_data()
)

model = TfidfLogisticBuilder().build().get_model()
processor = (
    TrainModelProcessor(
        text_column="text",
        label_column="sentiment",
        partition_column="partition",
        model_path=Path("./model.joblib"),
        evaluator=ModelEvaluator(
            metrics={
                "acc": AccuracyMetric(),
                "f1_micro": F1Metric(average="micro"),
                "f1_macro": F1Metric(average="macro"),
                }
            ),
        label_maps={"negative": 0, "neutral": 1, "positive": 2}
        )
    .set_elements(data=clean_data, model=model)
    .process()
)
