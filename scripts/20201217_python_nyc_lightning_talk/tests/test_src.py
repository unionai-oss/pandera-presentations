import hypothesis

from src.schemas import RawData, ProcessedData
from src.process_data import process_data
from src.train_model import train_model


@hypothesis.given(RawData.strategy(size=10))
@hypothesis.settings(max_examples=100)
def test_process_data(raw_data):
    process_data(raw_data)


@hypothesis.given(ProcessedData.strategy(size=10))
@hypothesis.settings(max_examples=100)
def test_train_model(processed_data):
    estimator = train_model(processed_data)
    preds = estimator.predict(processed_data.drop("price", axis=1))
    assert len(preds) == processed_data.shape[0]
