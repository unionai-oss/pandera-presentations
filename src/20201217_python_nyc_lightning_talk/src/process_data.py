from io import StringIO

import pandas as pd
import pandera as pa
from pandera.typing import DataFrame

from .schemas import RawData, ProcessedData, PROPERTY_TYPES


raw_data = """
square_footage,n_bedrooms,property_type,price
750,1,condo,200000
900,2,condo,400000
1200,2,house,500000
1100,3,house,450000
1000,2,condo,300000
1000,2,townhouse,300000
1200,2,townhouse,350000
"""


@pa.check_types
def process_data(raw_data: DataFrame[RawData]) -> DataFrame[ProcessedData]:
    return pd.get_dummies(
        raw_data.astype({"property_type": pd.CategoricalDtype(PROPERTY_TYPES)})
    )
