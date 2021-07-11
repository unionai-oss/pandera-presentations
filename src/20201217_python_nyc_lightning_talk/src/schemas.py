import pandera as pa
from pandera.typing import Series


PROPERTY_TYPES = ["condo", "townhouse", "house"]


class BaseSchema(pa.SchemaModel):
    square_footage: Series[int] = pa.Field(in_range={"min_value": 0, "max_value": 3000})
    n_bedrooms: Series[int] = pa.Field(in_range={"min_value": 0, "max_value": 10})
    price: Series[float] = pa.Field(in_range={"min_value": 0, "max_value": 1000000})

    class Config:
        coerce = True


class RawData(BaseSchema):
    property_type: Series[str] = pa.Field(isin=PROPERTY_TYPES)


class ProcessedData(BaseSchema):
    property_type_condo: Series[int] = pa.Field(isin=[0, 1])
    property_type_house: Series[int] = pa.Field(isin=[0, 1])
    property_type_townhouse: Series[int] = pa.Field(isin=[0, 1])
