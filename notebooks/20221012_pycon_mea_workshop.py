# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: pandera-presentations
#     language: python
#     name: pandera-presentations
# ---

# %% [markdown] slideshow={"slide_type": "slide"}
# # Pandera Workshop: Making Data Processing Pipelines more Readable and Robust
#
# *Global DevSlam x Pycon MEA 2022*

# %% [markdown] slideshow={"slide_type": "slide"}
# # Outline 📝
#
# - 🤷‍♂️ Why Should I Validate Data?
# - 🤔 What's Data Testing, and How Can I Put it Into Practice?
# - ✅ Pandera Quickstart: create statistical types for your DataFrames
# - ⌨️ Statistical types: A typing paradigm for DS/ML
# - 📊 Example 1: Validate your Data analysis
# - 🤖 Example 2: Validate your Machine Learning Pipeline
# - ⭐️ Conclusion: How can I start using Pandera in my work?

# %% [markdown]
# ### Setup

# %%
# %%capture
# !pip install 'pandera[all]'

# %%
import warnings
from hypothesis.errors import HypothesisWarning

from IPython.display import display, Markdown

warnings.filterwarnings("ignore", category=HypothesisWarning)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Where's the Code?
#
# 📓 **Notebook**: https://github.com/pandera-dev/pandera-presentations/blob/master/notebooks/20221012_pycon_mea_workshop.ipynb

# %% [markdown] slideshow={"slide_type": "slide"}
# # What's Data Validation?

# %% [markdown] slideshow={"slide_type": "slide"}
# Data validation is the act of _falsifying_ data against explicit assumptions
# for some downstream purpose, like analysis, modeling, and visualization.

# %% [markdown] slideshow={"slide_type": "slide"}
# # 🤷‍♂️ Why Should I Validate Data?
#
# - 🐞 It can be difficult to reason about and debug data processing pipelines.
# - ⚠️ It's critical to ensuring data quality in many contexts especially when the end product informs business decisions, supports scientific findings, or generates predictions in a production setting.
# - 📊 Everyone has a personal relationship with their dataframes.

# %% [markdown] slideshow={"slide_type": "slide"}
# # 🤔 What's Data Testing

# %% [markdown] slideshow={"slide_type": "slide"}
# > **Data validation:** The act of falsifying data against explicit assumptions for some downstream purpose, like
# > analysis, modeling, and visualization.
#
# > **Data Testing:** Validating not only real data, but also the functions that produce them.

# %% [markdown] slideshow={"slide_type": "slide"}
# # ✅ Pandera
#
# #### A data testing and statistical typing library for DS/ML-oriented data containers
#
# - Check the types and properties of dataframes
# - Easily integrate with existing data pipelines via function decorators
# - Synthesize data from schema objects for property-based testing

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Pandera Comes in Two Flavors
#
# #### Object-based API
#
# Defining a schema looks and feels like defining a pandas dataframe

# %%
import pandera as pa

schema = pa.DataFrameSchema(
    columns={
        "item": pa.Column(str, pa.Check.isin(["apple", "orange"])),
        "price": pa.Column(float, pa.Check.ge(0)),
    },
    coerce=True,
)

# %% [markdown] slideshow={"slide_type": "fragment"}
# #### Class-based API
#
# Define complex types with modern Python, inspired by [pydantic](https://pydantic-docs.helpmanual.io/) and `dataclasses`

# %%
from pandera.typing import DataFrame, Series

class Schema(pa.SchemaModel):
    item: Series[str] = pa.Field(isin=["apple", "orange"])
    price: Series[float] = pa.Field(gt=0)

    class Config:
        coerce = True

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Pandera Raises Informative Errors
#
# With valid data, pandera simply passes the data through:

# %%
valid_data = pd.DataFrame.from_records([
    {"item": "apple", "price": 0.5},
    {"item": "orange", "price": 0.75}
])

Schema.validate(valid_data)

# %% [markdown]
# But with invalid data, it'll raise an error, and you'll
# know exactly what went wrong with your data.

# %% tags=[]
invalid_data = pd.DataFrame.from_records([
    {"item": "applee", "price": 0.5},
    {"item": "orange", "price": -1000}
])

try:
    Schema.validate(invalid_data, lazy=True)
except pa.errors.SchemaErrors as exc:
    display(exc.failure_cases)


# %% [markdown]
# ### Schema Components
#
# ### Object-based API
#
# In the Pandera schemas consist of composable components that themselves can be used in isolation

# %%
column = pa.Column(str, pa.Check.isin(["apple", "orange"]), name="item")
column.validate(valid_data)

# %% [markdown]
# #### Class-based API
#
# The class-based API doesn't quite offer this same reusability, but you can still define `Field`s in isolation and reuse them across `SchemaModel` classes.

# %%
field = pa.Field(isin=["apple", "orange"])
field

# %%
field.to_column(str, name="item").validate(valid_data)

# %% [markdown]
# ### Checks
#
# `Check` objects are the validation workhorse of pandera. The `Check` class ships with some built-in checks for common use cases, but they are designed to make customizability extremely easy.
#
# As with schema components, we can define and use these by themselves.

# %%
gt_check = pa.Check.gt(0)
result = gt_check(pd.Series([1, 2, 3, -1]))
print(f"check_passed:\n{result.check_passed}")
print(f"\ncheck_output:\n{result.check_output}")
print(f"\nfailure_cases:\n{result.failure_cases}")

# %% [markdown]
# #### Custom Checks
#
# Checks have a particular contract, for example, here's a custom check that's equivalent to the built-in `Check.gt(0)` that we defined above:

# %%
gt_check = pa.Check(lambda x: x > 0, name="gt_zero")
gt_check(pd.Series([1, 2, 3, -1]))


# %% [markdown]
# Alternatively, we can use functions if you prefer:

# %%
def gt_zero(x):
    return x > 0

gt_check = pa.Check(gt_zero)
gt_check(pd.Series([1, 2, 3, -1]))

# %% [markdown]
# #### Dataframe-level Checks
#
# The signature of the function is flexible in that you can pass into the `Check` object also supports dataframe-level checks. In fact, the pandas API makes it such that the custom checks we defined can be used to validate an entire dataframe:

# %%
gt_check(pd.DataFrame([[1, 2, 3]]))

# %% [markdown]
# You can use these in your schemas like so:

# %%
df_schema = pa.DataFrameSchema(checks=gt_check)
df_schema(pd.DataFrame([[1, 2, 3]]))


# %% [markdown]
# Dataframe-level checks enable you to apply validation rules in a conditional manner.
#
# Suppose you want to check that column `A` is positive but only if column `B` has a specific value:

# %%
def conditional_check(df: pd.DataFrame) -> Series[bool]:
    check_output = pd.Series(True, index=df.index)
    return check_output.mask(df["B"] == "x", df["A"] > 0)


df_schema = pa.DataFrameSchema(checks=pa.Check(conditional_check))
df_schema.validate(pd.DataFrame({"A": [1, 2, -3], "B": ["x", "x", "y"]}))


# %% [markdown]
# #### Class-based API
#
# Custom checks in the class-based API are defined via methods:

# %%
class SchemaWithCustomChecks(pa.SchemaModel):
    @pa.check
    def gt_zero(cls, x: pd.Series) -> Series[bool]:
        return x > 0

SchemaWithCustomChecks.validate(pd.DataFrame([[1, 2, 3]]))


# %% [markdown]
# And similarly, we can define dataframe-level checks like so:

# %%
class SchemaWithCustomDFChecks(pa.SchemaModel):
    @pa.dataframe_check
    def conditional_check(cls, df: pd.DataFrame) -> Series[bool]:
        check_output = pd.Series(True, index=df.index)
        return check_output.mask(df["B"] == "x", df["A"] > 0)

SchemaWithCustomChecks.validate(pd.DataFrame({"A": [1, 2, -3], "B": ["x", "x", "y"]}))

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Pandera Supports Schema Transformations/Inheritence
#
# #### Object-based API
#
# Dynamically transform schema objects on the fly

# %%
schema = pa.DataFrameSchema(
    columns={
        "item": pa.Column(str, pa.Check.isin(["apple", "orange"])),
        "price": pa.Column(float, pa.Check.ge(0)),
    },
    coerce=True,
)

transformed_schema = schema.add_columns({"expiry": pa.Column(pd.Timestamp)})
print(transformed_schema)

# %% [markdown] slideshow={"slide_type": "fragment"}
# #### Class-based API
#
# Inherit from `pandera.SchemaModel` to Define Type Hierarchies

# %%
from pandera.typing import DataFrame, Series

class Schema(pa.SchemaModel):
    item: Series[str] = pa.Field(isin=["apple", "orange"])
    price: Series[float] = pa.Field(gt=0)

    class Config:
        coerce = True

class TransformedSchema(Schema):
    expiry: Series[pd.Timestamp]


print(TransformedSchema.to_schema())

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Integrate Seamlessly with your Pipeline
#
# Use decorators to add IO checkpoints to the critical functions in your pipeline

# %% [markdown]
# #### Object-based API
#
# Use the `pandera.check_io` decorator to validate input and output dataframes at runtime using `DataFrameSchema` objects.

# %%
from typing import List
from datetime import datetime

@pa.check_io(data=schema, out=transformed_schema)
def transform_data(data, expiry: List[datetime]):
    return data.assign(expiry=expiry)

transform_data(valid_data, [datetime.now()] * valid_data.shape[0])

# %% [markdown]
# #### Class-based API
#
# Use the `pandera.check_types` decorator to validate input/output dataframes at runtime using `SchemaModel` classes.

# %%
from typing import List
from datetime import datetime


@pa.check_types
def transform_data(data: DataFrame[Schema], expiry: List[datetime]) -> DataFrame[TransformedSchema]:
    return data.assign(expiry=expiry)


transform_data(valid_data, [datetime.now()] * valid_data.shape[0])


# %% [markdown] slideshow={"slide_type": "slide"}
# ### Generative Schemas
#
# Schemas that synthesize valid data under its constraints

# %%
Schema.example(size=5)

# %% [markdown] slideshow={"slide_type": "fragment"}
# **Data Testing:** Test the functions that produce clean data

# %%
from hypothesis import given


@given(Schema.strategy(size=5))
def test_fn(data):
    transform_data(data, [datetime.now()] * data.shape[0])


def run_test_suite():
    test_fn()
    print("tests passed ✅")


run_test_suite()


# %% [markdown] slideshow={"slide_type": "slide"}
# ## Scaling Pandera
#
# Pandera supports `dask`, `modin`, and `pyspark.pandas` dataframes to scale
# data validation to big data.

# %%
display(invalid_data)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Apply a single schema to a suite of dataframe-like objects
#
# #### `dask`

# %%
import dask.dataframe as dd

dask_dataframe = dd.from_pandas(invalid_data, npartitions=1)

try:
    Schema(dask_dataframe, lazy=True).compute()
except pa.errors.SchemaErrors as exc:
    display(exc.failure_cases.sort_index())

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Apply a single schema to a suite of dataframe-like objects
#
# #### `modin`

# %%
import modin.pandas as mpd
import ray

ray.init()
modin_dataframe = mpd.DataFrame(invalid_data)

try:
    Schema(modin_dataframe, lazy=True)
except pa.errors.SchemaErrors as exc:
    display(exc.failure_cases.sort_index())


# %% [markdown] slideshow={"slide_type": "slide"}
# ### Apply a single schema to a suite of dataframe-like objects
#
# #### `pyspark.pandas`

# %%
import pyspark.pandas as ps

pyspark_pd_dataframe = ps.DataFrame(invalid_data)

try:
    Schema(pyspark_pd_dataframe, lazy=True)
except pa.errors.SchemaErrors as exc:
    display(exc.failure_cases.sort_index())


# %% [markdown]
# ### Schema Inference
#
# You can bootstrap a schema if you want to quickly create a schema that you can refine further:

# %%
schema = pa.infer_schema(valid_data)

# %%
print(schema.to_script())

# %% [markdown] slideshow={"slide_type": "slide"}
# ## ⌨️ Statistical Typing

# %% [markdown]
# #### Type systems help programmers reason about and write more robust code

# %% slideshow={"slide_type": "fragment"}
from typing import Union

Number = Union[int, float]

def add_and_double(x: Number, y: Number) -> Number:
    ...


# %% [markdown] slideshow={"slide_type": "fragment"}
# #### Can you predict the outcome of these function calls?

# %%
add_and_double(5, 2)
add_and_double(5, "hello")
add_and_double(11.5, -1.5)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Similarly...

# %%
import pandera as pa
from pandera.typing import DataFrame, Series

class Inputs(pa.SchemaModel):
    x: Series[int]
    y: Series[int]

    class Config:
        coerce = True


class Outputs(Inputs):
    z: Series[int]
        
    @pa.dataframe_check
    def custom_check(cls, df: DataFrame) -> Series:
        return df["z"] == (df["x"] + df["y"]) * 2
    
    
@pa.check_types
def add_and_double(raw_data: DataFrame[Inputs]) -> DataFrame[Outputs]:
    ...

# %% [markdown] slideshow={"slide_type": "slide"}
# ### 🤔 What's Statistical Typing?
#
# > **Statistical typing** extends primitive data types with additional semantics
# > about the _properties held by a collection of data points_.

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Consider a single data point

# %%
data_point = {"square_footage": 700, "nbedrooms": 1, "price": 500_000}

# %% [markdown] slideshow={"slide_type": "fragment"}
# - Primitive datatypes
# - Value range
# - Allowable values
# - Regex string match
# - Nullability

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Now consider a collection data point

# %%
data_points = [
    {"square_footage": 700, "nbedrooms": 1, "price": 500_000},
    {"square_footage": 1000, "nbedrooms": 2, "price": 750_000},
    {"square_footage": 3000, "nbedrooms": 4, "price": 1_000_000},
    ...
]

# %% [markdown] slideshow={"slide_type": "fragment"}
# - Apply atomic checks at scale
# - Uniqueness
# - Monotonicity
# - Mean, median, standard deviation
# - Fractional checks, e.g. 90% of data points are not null

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Pandera is a Statistical Type System Geared Towards Data Science
#
# Statistical types are defined with multiple layers 🧅

# %% [markdown] slideshow={"slide_type": "fragment"}
# > **primitive data types**: `int`, `float`, `bool`, `str`, etc.

# %% [markdown] slideshow={"slide_type": "fragment"}
# > **deterministic properties**: domain of possible values, e.g. `x >= 0`

# %% [markdown] slideshow={"slide_type": "fragment"}
# > **probabilistic properties**: distributions that apply to the variable and their sufficient statistics, e.g. `mean`,
#   `standard deviation`

# %% [markdown] slideshow={"slide_type": "slide"}
# # 📊 Example 1: Validate your Data analysis

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Dataset: California Housing
#
# A dataset containing ~20,000 samples where each row is a California district and
# each column is an aggregate statistic about that district.

# %%
from sklearn.datasets import fetch_california_housing

housing_data = fetch_california_housing(as_frame=True).frame
housing_data.describe()

# %% [markdown] slideshow={"slide_type": "slide"}
# #### With a cursory glance at the data...

# %%
housing_data.head(5)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### We can start defining a basic schema

# %%
class HousingData(pa.SchemaModel):

    # features
    MedInc: Series[float] = pa.Field(in_range={"min_value": 0, "max_value": 100})
    HouseAge: Series[float] = pa.Field(in_range={"min_value": 0, "max_value": 100})
    AveRooms: Series[float] = pa.Field(in_range={"min_value": 0, "max_value": 1_000})
    AveBedrms: Series[float] = pa.Field(in_range={"min_value": 0, "max_value": 100})
    Population: Series[float] = pa.Field(in_range={"min_value": 0, "max_value": 100_000})
    AveOccup: Series[float] = pa.Field(in_range={"min_value": 0, "max_value": 10_000})
    Latitude: Series[float] = pa.Field(in_range={"min_value": -90, "max_value": 90})
    Longitude: Series[float] = pa.Field(in_range={"min_value": -180, "max_value": 180})

    # target variable! 🎯
    MedHouseVal: Series[float] = pa.Field(in_range={"min_value": 0, "max_value": 100})

    class Config:
        coerce = True


@pa.check_types
def read_data() -> DataFrame[HousingData]:
    return fetch_california_housing(as_frame=True).frame


housing_data = read_data()
print("validation passed ✅")


# %% [markdown] slideshow={"slide_type": "slide"}
# ### Analysis Pipeline
#
# Hypothesis: Median income is positively correlated with Median House Value

# %%
def analyze_data(housing_data, var1, var2):
    correlation_coef = housing_data[[var1, var2]].corr().at[var1, var2]
    display(Markdown(f"Pearson correlation coefficient = {correlation_coef:0.06f}"))
    housing_data.plot.scatter(var1, var2, s=1, alpha=0.5)

analyze_data(housing_data, "MedInc", "MedHouseVal")


# %% [markdown] slideshow={"slide_type": "slide"}
# #### Bake in statistical hypothesis testing into your pipeline
#
# Easily create re-usable custom checks

# %%
from scipy.stats import pearsonr
import pandera.extensions as extensions

@extensions.register_check_method(
    statistics=["var1", "var2", "alpha"],
    supported_types=[pd.DataFrame]
)
def is_positively_correlated(
    df: pd.DataFrame,
    *,
    var1: str,
    var2: str,
    alpha: float = 0.01,
):
    """Perform Pearson correlation hypothesis test."""

    r, pvalue = pearsonr(df[var1], df[var2])
    passed = r > 0 and pvalue <= alpha

    pretty_pvalue = np.format_float_scientific(pvalue)
    if passed:
        print(f"✅ {var1} is positively correlated with {var2} with r = {r:0.04f}; pvalue = {pretty_pvalue}")
    else:
        print(f"❌ {var1} not correlated with {var2} with with r = {r:0.04f}; pvalue = {pretty_pvalue}")

    return passed


# %% [markdown] slideshow={"slide_type": "slide"}
# #### Dynamically create schemas as statistical hypothesis validators

# %%
def analyze_data(housing_data, var1: str, var2: str):

    class HousingDataHypothesis(HousingData):
        class Config:
            coerce = True
            is_positively_correlated = {
                "var1": var1,
                "var2": var2,
                "alpha": 0.01,
            }

    housing_data = HousingDataHypothesis.validate(housing_data)
    correlation_coef = housing_data[[var1, var2]].corr().at[var1, var2]
    display(Markdown(f"Pearson correlation coefficient = {correlation_coef:0.06f}"))
    housing_data.plot.scatter(var1, var2, s=1, alpha=0.5)


analyze_data(housing_data, "MedInc", "MedHouseVal")

# %% [markdown] slideshow={"slide_type": "slide"}
# #### The Analysis Pipeline
#
# Every time this runs, pandera makes sure all the assumptions encoded in the schemas
# hold true.

# %%
def run_analysis_pipeline(var1: str, var2: str):
    data = read_data()
    analyze_data(data, var1, var2)


run_analysis_pipeline("MedInc", "MedHouseVal")


# %% [markdown] slideshow={"slide_type": "slide"}
# # 🤖 Example 2: Validate your Machine Learning Pipeline

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Prediction Task:
#
# From all the features, predict the median house value target `MedHouseVal`.

# %%
from typing import Tuple

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# ⚠️ This is the most critical part to check
@pa.check_types
def split_data(
    data: DataFrame[HousingData],
    test_size: float = 0.2,
) -> Tuple[DataFrame[HousingData], DataFrame[HousingData]]:
    return train_test_split(data, test_size=test_size)


# 👉 Notice that I don't use @pa.check_types here
def parse_data(data: DataFrame[HousingData], target: str) -> Tuple[DataFrame[HousingData], pd.Series]:
    features = [column for column in data if column != target]
    return data[features], data[target]


# 🔽 At this point onward the type annotations are for type linters like mypy
def train(features: pd.DataFrame, target: pd.Series) -> LinearRegression:
    model = LinearRegression()
    return model.fit(features, target)


def evaluate(model: LinearRegression, features: pd.DataFrame, target: pd.Series) -> float:
    prediction = model.predict(features)
    return r2_score(target, prediction)


# %% [markdown] slideshow={"slide_type": "slide"}
# #### Running a validated training pipeline

# %%
def run_training_pipeline(data: pd.DataFrame, target: str):
    train_data, test_data = split_data(data)
    train_features, train_target = parse_data(train_data, target)

    # train a model
    model = train(train_features, train_target)

    # evaluate
    train_r2 = evaluate(model, train_features, train_target)
    test_r2 = evaluate(model, *parse_data(test_data, target))

    return model, train_r2, test_r2


model, train_r2, test_r2 = run_training_pipeline(read_data(), "MedHouseVal")
print(f"🏋️‍♂️ Train R^2 score: {train_r2:0.6f}")
print(f"📝 Test R^2 score: {test_r2:0.6f}")
model


# %% [markdown] slideshow={"slide_type": "slide"}
# #### Unit testing a training pipeline
#
# Synthesize mock training data so that you don't have to hand-craft dataframes 🤯

# %%
from hypothesis import settings

prediction_schema = pa.SeriesSchema(
    float,
    # in-line custom checks
    pa.Check(lambda s: (s >= 0).mean() > 0.05, name="predictions are mostly positive"),
    nullable=False,
)


@given(HousingData.strategy(size=20))
@settings(max_examples=3)
def test_run_training_pipeline(data):
    target = "MedHouseVal"
    model, *_ = run_training_pipeline(data, target)
    features, _ = parse_data(data, target)
    predictions = pd.Series(model.predict(features))

    # validate predictions
    prediction_schema(predictions)


def run_test_suite():
    test_run_training_pipeline()
    print("✅ training pipeline test suite passed!")


run_test_suite()


# %% [markdown] slideshow={"slide_type": "slide"}
# # ⭐️ Conclusion: How can I start using Pandera in my work?

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Incrementally adopt `pandera` into your workflow

# %% [markdown] slideshow={"slide_type": "fragement"}
# > 🧠 → 📝 Encode the domain knowledge that you build up during the development and exploration process into schemas.

# %% [markdown] slideshow={"slide_type": "fragment"}
# > 🥾✨ If you're in a hurry, use [`pandera.infer_schema`](https://pandera.readthedocs.io/en/stable/schema_inference.html)
# > to bootstrap a schema and refine it over time.

# %% [markdown] slideshow={"slide_type": "fragment"}
# > ❗️ Identify the critical functions in your data processing pipeline and add `@pa.check_types` decorators as
# checkpoints.

# %% [markdown] slideshow={"slide_type": "fragment"}
# > 🔩 Codify data quality checks that are specific to your problem domain by creating reusable custom validation rules
# > via `@pandera.extensions.register_check_method`.

# %% [markdown] slideshow={"slide_type": "fragment"}
# > 🔄 Reuse schemas for runtime validation or test-time validation.

# %% [markdown] slideshow={"slide_type": "fragment"}
# > 🤩 Be more confident in the correctness of your analysis/model with programmatically enforced, self-documenting code.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 🛣 Future Roadmap
#
# - 📏 **Extensibility:** getting support for other schema formats and data container objects e.g.
#   `xarray`, `jsonschema`, `cudf`, `pyarrow`, and an extension API for arbitrary data containers.
# - 💻 **UX:** better error-reporting, more built-in checks, statistical hypothesis checks, conditional validation, and more!
# - 🤝 **Interoperability:** tighter integrations with the python ecosystem, e.g. `fastapi`, `pydantic`, `pytest`

# %% [markdown] slideshow={"slide_type": "slide"}
# # Join the Community!
#
# ![badge](https://img.shields.io/github/stars/pandera-dev/pandera?style=social)
# [![badge](https://img.shields.io/pypi/pyversions/pandera.svg)](https://pypi.python.org/pypi/pandera/)
# [![badge](https://img.shields.io/pypi/v/pandera.svg)](https://pypi.org/project/pandera/)
# ![badge](https://img.shields.io/github/contributors/pandera-dev/pandera)
# [![badge](https://pepy.tech/badge/pandera)](https://pepy.tech/project/pandera)
# [![badge](https://pepy.tech/badge/pandera/month)](https://pepy.tech/project/pandera)
# [![badge](https://img.shields.io/badge/discord-chat-purple?color=%235765F2&label=discord&logo=discord)](https://discord.gg/vyanhWuaKB)
#
#
# - **Twitter**: [@cosmicbboy](https://twitter.com/cosmicBboy)
# - **Discord**: https://discord.gg/vyanhWuaKB
# - **Email**: [niels@union.ai](mailto:niels@union.ai)
# - **Repo**: https://github.com/unionai-oss/pandera
# - **Docs**: https://pandera.readthedocs.io
# - **Contributing Guide**: https://pandera.readthedocs.io/en/stable/CONTRIBUTING.html
# - **Become a Sponsor**: https://github.com/sponsors/cosmicBboy
