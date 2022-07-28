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
#
# # Statistical Types for 🐼 Pandas DataFrames and Friends 🌈✨
#
# ### Parse, Validate, and Synthesize DataFrames with Generative Schemas.
#
# **Niels Bantilan**, Chief ML Engineer @ Union.ai
#
# *Python Live Webinars - Amicus, July 28th 2022*

# %% [markdown] slideshow={"slide_type": "slide"}
# # Background 🏞
#
# - 📜 B.A. in Biology and Dance
# - 📜 M.P.H. in Sociomedical Science and Public Health Informatics
# - 🤖 Chief Machine Learning Engineer @ Union.ai
# - 🛩 Flytekit OSS Maintainer
# - ✅ Author and Maintainer of Pandera
# - 🦾 Author of UnionML
# - 🛠 Make DS/ML practitioners more productive

# %% [markdown] slideshow={"slide_type": "slide"}
# # Outline 📝
#
# - 🤷‍♂️ Why Should I Validate Data?
# - 🤔 What's Data Testing, and How Can I Put it Into Practice?
# - ✅ Pandera Quickstart: create statistical types for your DataFrames
# - 📊 Example 1: Validate your Data analysis
# - 🤖 Example 2: Validate your Machine Learning Pipeline
# - ⭐️ Conclusion: How can I start using Pandera in my work?

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Where's the Code?
#
# 🖼 **Slides**: https://pandera-dev.github.io/pandera-presentations/slides/20220728_python_live_webinar_amicus.slides.html
#
# 📓 **Notebook**: https://github.com/pandera-dev/pandera-presentations/blob/master/notebooks/20220728_python_live_webinar_amicus.ipynb

# %% tags=["hide_input", "hide_output"] jupyter={"source_hidden": true}
import warnings
import pyspark

from IPython.display import display, Markdown

warnings.simplefilter("ignore")
pyspark.SparkContext().setLogLevel("OFF")

# %% [markdown] slideshow={"slide_type": "slide"}
# # 🤷‍♂️ Why Should I Validate Data?

# %% [markdown] slideshow={"slide_type": "slide"}
# ## What's a `DataFrame`?

# %% slideshow={"slide_type": "skip"}
import uuid

import numpy as np
import pandas as pd

dataframe = pd.DataFrame({
    "person_id": [str(uuid.uuid4())[:7] for _ in range(6)],
    "hours_worked": [38.5, 41.25, "35.0", 27.75, 22.25, -20.5],
    "wage_per_hour": [15.1, 15, 21.30, 17.5, 19.50, 25.50],
}).set_index("person_id")

df = dataframe

# %%
dataframe.head()


# %% [markdown] slideshow={"slide_type": "slide"}
# ## What's Data Validation?
#
# Data validation is the act of _falsifying_ data against explicit assumptions
# for some downstream purpose, like analysis, modeling, and visualization.

# %% [markdown] slideshow={"slide_type": "fragment"}
# > "All swans are white"

# %% tags=["hide_input"] slideshow={"slide_type": "fragment"} language="html"
# <p>
#     <a href="https://commons.wikimedia.org/wiki/File:Black_Swans.jpg#/media/File:Black_Swans.jpg">
#     <img src="https://upload.wikimedia.org/wikipedia/commons/6/60/Black_Swans.jpg" alt="Pair of black swans swimming" height="480" width="275"
#      style="display: block; margin-left: auto; margin-right: auto;"/>
#     </a>
#     <p style="font-size: x-small; text-align: center;">
#     <a href="http://creativecommons.org/licenses/by-sa/3.0/" title="Creative Commons Attribution-Share Alike 3.0">CC BY-SA 3.0</a>,
#     <a href="https://commons.wikimedia.org/w/index.php?curid=1243220">Link</a>
#     </p>
# </p>

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Why Do I Need it?

# %% [markdown] slideshow={"slide_type": "fragment"}
# #### 🐞 It can be difficult to reason about and debug data processing pipelines.

# %% [markdown] slideshow={"slide_type": "fragment"}
# #### ⚠️ It's critical to ensuring data quality in many contexts especially when the end product informs business decisions, supports scientific findings, or generates predictions in a production setting.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Everyone has a personal relationship with their dataframes

# %% [markdown] slideshow={"slide_type": "fragment"}
# ### Story Time 📖
#
# ##### Imagine that you're a data scientist maintaining an existing data processing pipeline 👩‍💻👨‍💻...

# %% slideshow={"slide_type": "skip"}
def process_data(df):
    return df.assign(weekly_income=lambda x: x.hours_worked * x.wage_per_hour)

try:
    process_data(dataframe)
except TypeError as exc:
    print(exc)


# %% [markdown] slideshow={"slide_type": "slide"}
# ### One day, you encounter an error log trail and decide to follow it...

# %% [markdown]
# ```python
# /usr/local/miniconda3/envs/pandera-presentations/lib/python3.7/site-packages/pandas/core/ops/__init__.py in masked_arith_op(x, y, op)
#     445         if mask.any():
#     446             with np.errstate(all="ignore"):
# --> 447                 result[mask] = op(xrav[mask], com.values_from_object(yrav[mask]))
#     448 
#     449     else:
#
# TypeError: can't multiply sequence by non-int of type 'float'
# ```

# %% [markdown] slideshow={"slide_type": "slide"}
# ### And you find yourself at the top of a function...

# %%
def process_data(df):
    ...

# %% [markdown] slideshow={"slide_type": "slide"}
# ### You look around, and see some hints of what had happened...

# %%
def process_data(df):
    return df.assign(weekly_income=lambda x: x.hours_worked * x.wage_per_hour)


# %% [markdown] slideshow={"slide_type": "slide"}
# ### You sort of know what's going on, but you want to take a closer look!

# %%
def process_data(df):
    import pdb; pdb.set_trace()  # <- insert breakpoint
    return df.assign(weekly_income=lambda x: x.hours_worked * x.wage_per_hour)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### And you find some funny business going on...

# %%
print(df)

# %%
df.dtypes

# %%
df.hours_worked.map(type)


# %% [markdown] slideshow={"slide_type": "slide"}
# ### You squash the bug and add documentation for the next weary traveler who happens upon this code.

# %%
def process_data(df):
    return (
        df
        # make sure columns are floats
        .astype({"hours_worked": float, "wage_per_hour": float})
        # replace negative values with nans
        .assign(hours_worked=lambda x: x.hours_worked.where(x.hours_worked >= 0, np.nan))
        # compute weekly income
        .assign(weekly_income=lambda x: x.hours_worked * x.wage_per_hour)
    )


# %%
process_data(df)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### ⏱ A few months later...

# %% [markdown] slideshow={"slide_type": "slide"}
# ### You find yourself at a familiar function, but it looks a little different from when you left it...

# %% slideshow={"slide_type": "skip"}
# This needs to be here, but skipped for story-telling effect in the slides
import pandera as pa
from pandera.typing import DataFrame, Series

class RawData(pa.SchemaModel):
    hours_worked: Series[float] = pa.Field(coerce=True, nullable=True)
    wage_per_hour: Series[float] = pa.Field(coerce=True, nullable=True)

class ProcessedData(RawData):
    hours_worked: Series[float] = pa.Field(ge=0, coerce=True, nullable=True)
    weekly_income: Series[float] = pa.Field(nullable=True)


# %%
@pa.check_types
def process_data(df: DataFrame[RawData]) -> DataFrame[ProcessedData]:
    return (
        # replace negative values with nans
        df.assign(hours_worked=lambda x: x.hours_worked.where(x.hours_worked >= 0, np.nan))
        # compute weekly income
        .assign(weekly_income=lambda x: x.hours_worked * x.wage_per_hour)
    )


# %% [markdown] slideshow={"slide_type": "slide"}
# ### You look above and see what `RawData` and `ProcessedData` are, finding a `NOTE` that a fellow traveler has left for you.

# %%
import pandera as pa

# NOTE: this is what's supposed to be in `df` going into `process_data`
class RawData(pa.SchemaModel):
    hours_worked: Series[float] = pa.Field(coerce=True, nullable=True)
    wage_per_hour: Series[float] = pa.Field(coerce=True, nullable=True)


# ... and this is what `process_data` is supposed to return.
class ProcessedData(RawData):
    hours_worked: Series[float] = pa.Field(ge=0, coerce=True, nullable=True)
    weekly_income: Series[float] = pa.Field(nullable=True)


@pa.check_types
def process_data(df: DataFrame[RawData]) -> DataFrame[ProcessedData]:
    ...

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Moral of the Story

# %% [markdown] slideshow={"slide_type": "fragment"}
# #### The better you can reason about the contents of a dataframe, the faster you can debug.

# %% [markdown] slideshow={"slide_type": "fragment"}
# #### The faster you can debug, the sooner you can focus on downstream tasks that you care about.

# %% [markdown] slideshow={"slide_type": "fragment"}
# #### By validating data through explicit contracts, you're also creating documentation for the rest of your team.


# %% [markdown] slideshow={"slide_type": "slide"}
# # 🤔 What's Data Testing
#
# ### And How Can I Put it Into Practice?

# %% [markdown] slideshow={"slide_type": "slide"}
# > **Data validation:** The act of falsifying data against explicit assumptions for some downstream purpose, like
# > analysis, modeling, and visualization.

# %% [markdown] slideshow={"slide_type": "fragment"}
# > **Data Testing:** Validating not only real data, but also the functions that produce them.

# %% [markdown] slideshow={"slide_type": "slide"}
#
# # In the Real World 🌍
#
# #### Validate real data in production

# %% [raw]
# <div class="mermaid">
# graph LR
#     R[(raw data)] --> RS([raw schema])
#     RS --> TF[transform function]
#     TF --> TS([transformed schema])
#     TS --> T[(transformed data)]
#
#     style RS fill:#8bedc6,stroke:#333
#     style TS fill:#8bedc6,stroke:#333
# </div>

# %% [markdown] slideshow={"slide_type": "slide"}
#
# # In the Test Suite 🧪
#
# #### Validate functions that produce data, given some test cases

# %% [raw]
# <div class="mermaid">
# graph LR
#     G([raw schema]) --> T1[(test case 1)]
#     G --> T2[(test case 2)]
#     G --> TN[(test case n)]
#
#     T1 --> TF[transform function]
#     T2 --> TF
#     TN --> TF
#     TF --> TS([transformed schema])
#     TS --> T[(transformed data)]
#
#     style G fill:#8bedc6,stroke:#333
#     style TS fill:#8bedc6,stroke:#333
# </div>

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Data Testing in Practice
#
# #### Data testing is an iterative process of:
#
# - Building domain knowledge about the data at hand with respect to the stated goal.
# - Implementing data transforms and defining schemas/tests in parallel.
# - Verifying that the output of the data transform is what you expected.

# %% [raw]
# <div class="mermaid">
# graph LR
#     
#     D[Define Goal]
#
#     subgraph Development Process
#         E[Explore Data]
#         S[Define Schema and Tests]
#         I[Implement Data Transforms]
#         V[Verify Data]
#     end
#
#     P{Checks Pass?}
#     A[Continue Analysis...]
#
#     D --> E
#     E --> S
#     E --> I
#     I --> V
#     S --> V
#     V --> P
#     P -- No --> E
#     P -- Yes --> A
#
#     style P fill:#aaffd6,stroke:#333
#     style A stroke-dasharray: 5 5
# </div>

# %% [markdown] slideshow={"slide_type": "slide"}
# # ✅ Pandera Quickstart
#
# ### Create statistical types for your DataFrames

# %% [markdown] slideshow={"slide_type": "slide"}
# <img src="https://raw.githubusercontent.com/pandera-dev/pandera/master/docs/source/_static/pandera-logo.png" width="125px" style="margin: 0;"/>
#
# <h2 style="margin-top: 0;">Pandera</h2>
#
# #### An expressive and light-weight statistical typing tool for dataframe-like containers
#
# - Check the types and properties of dataframes
# - Easily integrate with existing data pipelines via function decorators
# - Synthesize data from schema objects for property-based testing

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Object-based API
#
# Defining a schema looks and feels like defining a pandas dataframe

# %%
import pandera as pa

clean_data_schema = pa.DataFrameSchema(
    columns={
        "continuous": pa.Column(float, pa.Check.ge(0), nullable=True),
        "categorical": pa.Column(str, pa.Check.isin(["A", "B", "C"]), nullable=True),
    },
    coerce=True,
)

# %% [markdown] slideshow={"slide_type": "fragment"}
# #### Class-based API
#
# Define complex types with modern Python, inspired by [pydantic](https://pydantic-docs.helpmanual.io/) and `dataclasses`

# %%
from pandera.typing import DataFrame, Series

class CleanData(pa.SchemaModel):
    continuous: Series[float] = pa.Field(ge=0, nullable=True)
    categorical: Series[str] = pa.Field(isin=["A", "B", "C"], nullable=True)

    class Config:
        coerce = True

# %% [markdown] slideshow={"slide_type": "notes"}
# Pandera comes in two flavors

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Pandera Raises Informative Errors
#
# Know Exactly What Went Wrong with Your Data

# %% tags=[]
raw_data = pd.DataFrame({
    "continuous": ["-1.1", "4.0", "10.25", "-0.1", "5.2"],
    "categorical": ["A", "B", "C", "Z", "X"],
})

try:
    CleanData.validate(raw_data, lazy=True)
except pa.errors.SchemaErrors as exc:
    display(exc.failure_cases)


# %% [markdown] slideshow={"slide_type": "slide"}
# ### Pandera Supports Schema Transformations/Inheritence
#
# #### Object-based API
#
# Dynamically transform schema objects on the fly

# %%
raw_data_schema = pa.DataFrameSchema(
    columns={
        "continuous": pa.Column(float),
        "categorical": pa.Column(str),
    },
    coerce=True,
)

clean_data_schema.update_columns({
    "continuous": {"nullable": True},
    "categorical": {"checks": pa.Check.isin(["A", "B", "C"]), "nullable": True},
});


# %% [markdown] slideshow={"slide_type": "fragment"}
# #### Class-based API
#
# Inherit from `pandera.SchemaModel` to Define Type Hierarchies

# %%
class RawData(pa.SchemaModel):
    continuous: Series[float]
    categorical: Series[str]

    class Config:
        coerce = True

class CleanData(RawData):
    continuous = pa.Field(ge=0, nullable=True)
    categorical = pa.Field(isin=["A", "B", "C"], nullable=True);


# %% [markdown] slideshow={"slide_type": "slide"}
# ### Integrate Seamlessly with your Pipeline
#
# Use decorators to add IO checkpoints to the critical functions in your pipeline

# %%
@pa.check_types
def fn(raw_data: DataFrame[RawData]) -> DataFrame[CleanData]:
    return raw_data.assign(
        continuous=lambda df: df["continuous"].where(lambda x: x > 0, np.nan),
        categorical=lambda df: df["categorical"].where(lambda x: x.isin(["A", "B", "C"]), np.nan),
    )


fn(raw_data)


# %% [markdown] slideshow={"slide_type": "slide"}
# ### Generative Schemas
#
# Schemas that synthesize valid data under its constraints

# %%
CleanData.example(size=5)

# %% [markdown] slideshow={"slide_type": "fragment"}
# **Data Testing:** Test the functions that produce clean data

# %%
from hypothesis import given


@given(RawData.strategy(size=5))
def test_fn(raw_data):
    fn(raw_data)


def run_test_suite():
    test_fn()
    print("tests passed ✅")


run_test_suite()


# %% [markdown] slideshow={"slide_type": "slide"}
# ## Scaling Pandera with Pandas' Friends 🐼🌈✨
#
# Pandera supports `dask`, `modin`, and `pyspark.pandas` dataframes to scale
# data validation to big data.

# %%
display(raw_data)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Apply a single schema to a suite of dataframe-like objects
#
# #### `dask`

# %%
import dask.dataframe as dd

dask_dataframe = dd.from_pandas(raw_data, npartitions=1)

try:
    CleanData(dask_dataframe, lazy=True).compute()
except pa.errors.SchemaErrors as exc:
    display(exc.failure_cases.sort_index())

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Apply a single schema to a suite of dataframe-like objects
#
# #### `modin`

# %%
import modin.pandas as mpd

modin_dataframe = mpd.DataFrame(raw_data)

try:
    CleanData(modin_dataframe, lazy=True)
except pa.errors.SchemaErrors as exc:
    display(exc.failure_cases.sort_index())


# %% [markdown] slideshow={"slide_type": "slide"}
# ### Apply a single schema to a suite of dataframe-like objects
#
# #### `pyspark.pandas`

# %%
import pyspark.pandas as ps

pyspark_pd_dataframe = ps.DataFrame(raw_data)

try:
    CleanData(pyspark_pd_dataframe, lazy=True)
except pa.errors.SchemaErrors as exc:
    display(exc.failure_cases.sort_index())


# %% [markdown] slideshow={"slide_type": "slide"} tags=[]
# ### Meta Comment
#
# ##### This presentation notebook is validated by pandera 🤯

# %% [markdown] tags=[]
# ![mindblown](https://media.giphy.com/media/xT0xeJpnrWC4XWblEk/giphy-downsized-large.gif)

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
