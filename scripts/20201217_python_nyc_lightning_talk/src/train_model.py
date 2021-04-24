import pandera as pa
from pandera.typing import DataFrame
from sklearn.linear_model import LinearRegression

from .schemas import RawData, ProcessedData


@pa.check_types
def train_model(processed_data: DataFrame[ProcessedData]):
    estimator = LinearRegression()
    targets = processed_data["price"]
    features = processed_data.drop("price", axis=1)
    estimator.fit(features, targets)
    return estimator
