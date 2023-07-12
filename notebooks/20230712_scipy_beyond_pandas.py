# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: pandera-presentations
#     language: python
#     name: python3
# ---

# %% [markdown] slideshow={"slide_type": "slide"}
#
# # Pandera: Beyond Pandas Data Validation 🐼✅
#
# **Niels Bantilan**, Chief ML Engineer @ Union.ai
#
# *SciPy 2023, July 12th 2023*

# %% [markdown] slideshow={"slide_type": "slide"}
# # Background
#
# - 📜 B.A. in Biology and Dance
# - 📜 M.P.H. in Sociomedical Science and Public Health Informatics
# - 🤖 Chief Machine Learning Engineer @ Union.ai
# - 🛩 Flytekit OSS Maintainer
# - ✅ Author and Maintainer of Pandera
# - 🦾 Author of UnionML
# - 🛠 Make DS/ML practitioners more productive

# %% [markdown] slideshow={"slide_type": "slide"}
# ### This is a talk about open source development 🧑🏾‍💻

# %% [markdown] slideshow={"slide_type": "slide"}
# # Outline 📝
#
# - 🐣 Origins: solving a local problem
# - 🐓 Evolution: solving other people's problems
# - 🦩 Revolution: rewriting Pandera's internals
# - 🌅 What's next?

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Where's the Code?
#
# 🖼 **Slides**: https://unionai-oss.github.io/pandera-presentations/slides/20230712_scipy_beyond_pandas.slides.html
#
# 📓 **Notebook**: https://github.com/unionai-oss/pandera-presentations/blob/master/notebooks/20230712_scipy_beyond_pandas.ipynb

# %% [markdown] slideshow={"slide_type": "slide"}
# # 🐣 Origins

# %% jupyter={"source_hidden": true} tags=["hide_input", "hide_output"]
import os
import warnings
import pyspark

from IPython.display import display, Markdown

warnings.simplefilter("ignore")
pyspark.SparkContext().setLogLevel("OFF")
os.environ["MODIN_ENGINE"] = "ray"

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 🤷‍♂️ Why Should I Validate Data?

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

# %% slideshow={"slide_type": "fragment"} tags=["hide_input"] language="html"
# <p>
#     <a href="https://commons.wikimedia.org/wiki/File:Black_Swans.jpg#/media/File:Black_Swans.jpg">
#     <img src="https://upload.wikimedia.org/wikipedia/commons/6/60/Black_Swans.jpg" alt="Pair of black swans swimming" height="275" width="275"
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
# ## Everyone has a personal relationship with their data

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
from pandera.typing import DataFrame

class RawData(pa.DataFrameModel):
    hours_worked: float = pa.Field(coerce=True, nullable=True)
    wage_per_hour: float = pa.Field(coerce=True, nullable=True)

class ProcessedData(RawData):
    hours_worked: float = pa.Field(ge=0, coerce=True, nullable=True)
    weekly_income: float = pa.Field(nullable=True)


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
    hours_worked: float = pa.Field(coerce=True, nullable=True)
    wage_per_hour: float = pa.Field(coerce=True, nullable=True)


# ... and this is what `process_data` is supposed to return.
class ProcessedData(RawData):
    hours_worked: float = pa.Field(ge=0, coerce=True, nullable=True)
    weekly_income: float = pa.Field(nullable=True)


@pa.check_types
def process_data(df: DataFrame[RawData]) -> DataFrame[ProcessedData]:
    ...

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Moral of the Story

# %% [markdown] slideshow={"slide_type": "fragment"}
# > ##### The better you can reason about the contents of a dataframe, the faster you can debug.

# %% [markdown] slideshow={"slide_type": "fragment"}
# > ##### The faster you can debug, the sooner you can focus on downstream tasks that you care about.

# %% [markdown] slideshow={"slide_type": "fragment"}
# > ##### By validating data through explicit contracts, you also create data documentation *and* a simple, stateless data shift detector.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Pandera Design Principles
#
# From [scipy 2020 - pandera: Statistical Data Validation of Pandas Dataframes](https://conference.scipy.org/proceedings/scipy2020/niels_bantilan.html) 
#
# <image src="../static/pandera_design_principles.png" width="550px">

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Pandera Programming Model
#
# The pandera programming model is an iterative loop of building
# statistical domain knowledge, implementing data transforms and schemas,
# and verifying data.
#
# <br>
#
# <image src="../static/pandera_programming_model.png" width="700px">

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Meta Comment
#
# ##### This presentation notebook is validated by pandera 🤯

# %% [markdown]
# ![mindblown](https://media.giphy.com/media/xT0xeJpnrWC4XWblEk/giphy-downsized-large.gif)


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

# %%
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
# # 🐓 Evolution
#
# <iframe style="width:100%;height:auto;min-width:600px;min-height:400px;" src="https://star-history.com/embed?secret=#unionai-oss/pandera&Date" frameBorder="0"></iframe>

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Major Events
#
# %% [markdown] slideshow={"slide_type": "fragment"}
# > ##### 📖 Documentation Improvements

# %% [markdown] slideshow={"slide_type": "fragment"}
# > ##### 🔤 Class-based API

# %% [markdown] slideshow={"slide_type": "fragment"}
# > ##### 📊 Data Synthesis Strategies

# %% [markdown] slideshow={"slide_type": "fragment"}
# > ##### ⌨️ Pandera Type System


# %% [markdown] slideshow={"slide_type": "slide"}
# ## Expanding scope
#
# Adding `geopandas`, `dask`, `modin`, and `pyspark.pandas` was relatively
# straight forward.

# %%
display(raw_data)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### `dask`

# %%
import dask.dataframe as dd

dask_dataframe = dd.from_pandas(raw_data, npartitions=1)

try:
    CleanData(dask_dataframe, lazy=True).compute()
except pa.errors.SchemaErrors as exc:
    display(exc.failure_cases.sort_index())

# %% [markdown] slideshow={"slide_type": "slide"}
# #### `modin`

# %%
import modin.pandas as mpd

modin_dataframe = mpd.DataFrame(raw_data)

try:
    CleanData(modin_dataframe, lazy=True)
except pa.errors.SchemaErrors as exc:
    display(exc.failure_cases.sort_index())


# %% [markdown] slideshow={"slide_type": "slide"}
# #### `pyspark.pandas`

# %%
import pyspark.pandas as ps

pyspark_pd_dataframe = ps.DataFrame(raw_data)

try:
    CleanData(pyspark_pd_dataframe, lazy=True)
except pa.errors.SchemaErrors as exc:
    display(exc.failure_cases)


# %% [markdown] slideshow={"slide_type": "slide"}
# ### Problem: What about non-pandas-compliant dataframes?

# %% [markdown] slideshow={"slide_type": "fragment"}
# ### 😩 Design weaknesses
#
# - Schemas and checks were strongly coupled with pandas
# - Error reporting and eager validation assumed in-memory data
# - Leaky pandas abstractions

# %% [markdown] slideshow={"slide_type": "fragment"}
# ### 💪 Design strengths
#
# - Generic schema interface
# - Flexible check abstraction
# - Flexible type system

# %% [markdown] slideshow={"slide_type": "slide"}
# # 🦩 Revolution

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Re-writing pandera internals
#
# **High-level approach:** decoupling schema specification from backend
#
# - A `pandera.api` subpackage, which contains the schema specification that
#   defines the properties of an underlying data structure.
# - A `pandera.backends` subpackage, which leverages the schema specification and
#   implements the actual validation logic.
# - A backend registry, which maps a particular API specification to a backend
#   based on the DataFrame type being validated.
# - A common type-aware `Check` namespace and registry, which registers
#   type-specific implementations of built-in checks and allows contributors to
#   easily add new built-in checks.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Writing your own schema
#
# ``` python
# import sloth as sl
# from pandera.api.base.schema import BaseSchema
# from pandera.backends.base import BaseSchemaBackend
#
# class DataFrameSchema(BaseSchema):
#     def __init__(self, **kwargs):
#         # add properties that this dataframe would contain
#
# class DataFrameSchemaBackend(BaseSchemaBackend):
#     def validate(
#         self,
#         check_obj: sl.DataFrame,
#         schema: DataFrameSchema,
#         *,
#         **kwargs,
#     ):
#         # implement custom validation logic
#         
# # register the backend
# DataFrameSchema.register_backend(
#     sloth.DataFrame,
#     DataFrameSchemaBackend,
# )
# ````

# %% [markdown] slideshow={"slide_type": "slide"}
# ### 📢 Pandera now supports `pyspark.sql.DataFrame` in `0.16.0b`!
#
# https://pandera.readthedocs.io/en/latest/
#
# ```python
# import pandera.pyspark as pa
# import pyspark.sql.types as T
#
# from decimal import Decimal
# from pyspark.sql import DataFrame
# from pandera.pyspark import DataFrameModel
#
#
# class PanderaSchema(DataFrameModel):
#     id: T.IntegerType() = pa.Field(gt=5)
#     product_name: T.StringType() = pa.Field(str_startswith="B")
#     price: T.DecimalType(20, 5) = pa.Field()
#     description: T.ArrayType(T.StringType()) = pa.Field()
#     meta: T.MapType(T.StringType(), T.StringType()) = pa.Field()
# ````

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Organizational and Development Challenges
#
# - **Multi-tasking the rewrite with PR reviews**
# - **Centralized knowledge**
# - **Informal governance**

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Retrospective
#
# > #### Things in place that reduced the risk of regressions
# > 
# > - Unit tests.
# > - Localized pandas coupling.
# > - Lessons learned from pandas-compliant integrations.

# %% [markdown] slideshow={"slide_type": "fragment"}
# > #### Additional approaches to put into practice in the future:
# > 
# > - Thoughtful design work.
# > - Library-independent error reporting.
# > - Decoupling metadata from data.
# > - Investing in governance and community.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Updated Principles
#
# <image src="../static/pandera_updated_principles.png" width="600px">

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Announcement
#
# ### 🎉 Pandera has joined Union.ai 🎉

# %% [markdown] slideshow={"slide_type": "fragment"}
#
# What does this mean?
#
# - It will continue to be open source.
# - It will have more resources to maintain and govern it.
# - We can learn from enterprise users.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 🛣️ Roadmap
#
# - 🤝 **Integrations**: support more data manipulation libraries:
#   - Polars support: https://github.com/unionai-oss/pandera/issues/1064
#   - Ibis support: https://github.com/unionai-oss/pandera/issues/1105
#   - Investigate the dataframe-api standard: https://github.com/data-apis/dataframe-api
#   - Open an issue! https://github.com/unionai-oss/pandera/issues
# - 💻 **User Experience:** polish the API:
#   - better error-reporting
#   - more built-in checks
#   - conditional validation
# - 🤝 **Interoperability:** tighter integrations with the python ecosystem:
#   - `pydantic v2`
#   - `pytest`: collect data coverage statistics
#   - `hypothesis`: faster data synthesis
# - 🏆 **Innovations:** new capabilities:
#   - stateful data validation
#   - model-based types

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

# %% [markdown] slideshow={"slide_type": "slide"}
# # Join me at the sprints!
#
# Contribute to the `Flyte` project: https://www.flyte.org
#
# <image src="../static/flyte_sprint.png" width="600px">
