# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: pandera-presentations
#     language: python
#     name: pandera-presentations
# ---

# %% [markdown] slideshow={"slide_type": "slide"}
#
# # Pandera: Towards Better Data Testing Tools for Data Science and Machine Learning
#
# ### Niels Bantilan
#
# Scipy, July 16th 2021

# %% [markdown] slideshow={"slide_type": "notes"}
# This talk about the road to better data testing tools for data science and
# machine learning

# %% [markdown] slideshow={"slide_type": "slide"}
# # Outline 📝

# %% [markdown] slideshow={"slide_type": "fragment"}
# - Introduction to Data Testing

# %% [markdown] slideshow={"slide_type": "fragment"}
# - Pandera Quickstart

# %% [markdown] slideshow={"slide_type": "fragment"}
# - Roadmap: Guiding Principles

# %% [markdown] slideshow={"slide_type": "fragment"}
# - How to Contribute

# %% [markdown] slideshow={"slide_type": "slide"}
#
# # Introduction to Data Testing
#
# > "Data Testing" is the colloquial term [for] "schema validation" or "data validation"...
# > It's merely a fancy way of [saying], "are my data as I expect them to be?" - [Eric Ma](https://ericmjl.github.io/blog/2020/8/30/pandera-data-validation-and-statistics/)

# %% [markdown] slideshow={"slide_type": "fragment"}
# > **Addendum**: "Data Testing" can also be thought of as testing the transformation code
# > that produces the data.

# %% [markdown] slideshow={"slide_type": "notes"}
# To give you a simple example...

# %% [markdown] slideshow={"slide_type": "slide"}
# ## A Simple Example: Life Before Pandera
#
# `data_cleaner.py`

# %%
import pandas as pd

raw_data = pd.DataFrame({
    "continuous": ["-1.1", "4.0", "10.25", "-0.1", "5.2"],
    "categorical": ["A", "B", "C", "Z", "X"],
})

def clean(raw_data):
    return (
        raw_data
        # do some cleaning 🧹✨
    )

# %% [markdown] slideshow={"slide_type": "fragment"}
# <br>
# `test_data_cleaner.py`

# %%
def test_clean():
    mock_raw_data = ...  # hand-written mock data 😅
    result = clean(mock_raw_data)

    # assumptions about clean data
    assert result["continuous"].ge(0).all()
    assert result["categorical"].isin(["A", "B", "C"]).all()

# %% [markdown] slideshow={"slide_type": "slide"}
#
# <img src="https://raw.githubusercontent.com/pandera-dev/pandera/master/docs/source/_static/pandera-logo.png" width="125px" style="margin: 0;"/>
#
# <h2 style="margin-top: 0;">Pandera Quickstart</h2>
#
# An expressive and light-weight statistical validation tool for dataframes
# <br>

# %% [markdown] slideshow={"slide_type": "fragment"}
# - Check the types and properties of dataframes

# %% [markdown] slideshow={"slide_type": "fragment"}
# - Easily integrate with existing data pipelines via function decorators

# %% [markdown] slideshow={"slide_type": "fragment"}
# - Synthesize data from schema objects for property-based testing

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Object-based API
#
# Defining a schema looks and feels like defining a pandas dataframe

# %%
import pandera as pa

schema = pa.DataFrameSchema(
    columns={
        "continuous": pa.Column(float, pa.Check.ge(0)),
        "categorical": pa.Column(str, pa.Check.isin(["A", "B", "C"])),
    },
    coerce=True,
)

# %% [markdown] slideshow={"slide_type": "fragment"}
# #### Class-based API
#
# Inspired by [pydantic](https://pydantic-docs.helpmanual.io/)

# %%
from pandera.typing import Series

class Schema(pa.SchemaModel):
    continuous: Series[float] = pa.Field(ge=0)
    categorical: Series[str] = pa.Field(isin=["A", "B", "C"])

    class Config:
        coerce = True

# %% tags=["hide_input"]
from IPython.display import display, Markdown

# %% [markdown] slideshow={"slide_type": "notes"}
# Pandera comes in two flavors

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Pandera Parses and Validates Data
#
# With `coerce=True` pandera first performs type coercion on the columns before
# validating them.

# %%
raw_data = pd.DataFrame({
    "continuous": ["-1.1", "4.0", "10.25", "-0.1", "5.2"],
    "categorical": ["A", "B", "C", "Z", "X"],
})

try:
    Schema.validate(raw_data, lazy=True)
except pa.errors.SchemaErrors as exc:
    display(exc.failure_cases)

# %% [markdown] slideshow={"slide_type": "notes"}
# The core API of pandera
#
# As a meta-point, this presentation is built with jupyter, so almost all
# the code in this presentation is validated and tested with pandera

# %% [markdown] slideshow={"slide_type": "slide"}
#
# # 🛣 Roadmap: Guiding Principles

# %% [markdown] slideshow={"slide_type": "notes"}
# I'll outline pandera's roadmap by describing to you four principles
# that guide the project's development. Along the way I'll highlight a few of
# pandera's features and specific roadmap items that are it the repo's issues page.

# %% [markdown] slideshow={"slide_type": "slide"}
#
# ### Principle 1: Parse, then Validate
#
# > pydantic [and pandera guarantee] the types and constraints of the output
# > [data], not the input data. -[Pydantic Docs](https://pydantic-docs.helpmanual.io/usage/models/)
#
# ```python
# raw_data = ...
# valid_data = validate(parse(raw_data))  # raise Exception if constraints are not met
# ```

# %% [markdown] slideshow={"slide_type": "subslide"}
# Pandera guarantees that input and output dataframes fulfill the types and
# constraints as defined by type annotations.

# %%
raw_data = pd.DataFrame({
    "continuous": list("123456"),
    "categorical": list("AABBCC"),
})

# %% slideshow={"slide_type": "fragment"}
class Schema(pa.SchemaModel):
    continuous: Series[float] = pa.Field(ge=0)
    categorical: Series[str] = pa.Field(isin=["A", "B", "C"])

    class Config:
        coerce = True

# %% slideshow={"slide_type": "fragment"}
from pandera.typing import DataFrame

@pa.check_types
def summarize_data(clean_data: DataFrame[Schema]):
    return clean_data.groupby("categorical")["continuous"].mean()

display(summarize_data(raw_data).rename("mean_continuous").to_frame())

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### 🛣 Roadmap Item
#
# Extend parsing functionality to support arbitrary transformations [![github-issue](https://img.shields.io/badge/github_issue-252-blue?style=for-the-badge&logo=github)](https://github.com/pandera-dev/pandera/issues/252)

# %% [markdown]
# ``` python
# class Schema(pa.SchemaModel):
#     continuous: Series[float] = pa.Field(ge=0)
#     categorical: Series[str] = pa.Field(isin=["A", "B", "C"])
# 
#     class Config:
#         coerce = True
# 
#     @pa.parser("continuous")
#     def truncate_continuous(cls, series):
#         """set negative values to nan"""
#         return series.mask(series < 0, pd.NA)
# 
#     @pa.parser("continuous")
#     def filter_continuous(cls, series):
#         """filter out records with negative values in the continuous column"""
#         return series[series >= 0]
# ```


# %% [markdown] slideshow={"slide_type": "slide"}
# ### Principle 2: Make Schemas Reuseable, Adaptable, and Portable

# %% [markdown] slideshow={"slide_type": "fragment"}
# Once you've defined a schema, you can use it in your source code

# %%
# data_cleaning.py
from pandera.typing import DataFrame

@pa.check_types
def clean_data(raw_data) -> DataFrame[Schema]:
    return (
        raw_data
        # do some cleaning
    )

# %% [markdown] slideshow={"slide_type": "fragment"}
# <br>
# ... and your test suite (or anywhere you want, really!)

# %%
# test_data_cleaning.py
def test_clean_data():
    raw_data = ...
    clean_data(raw_data)

# %% [markdown] slideshow={"slide_type": "fragment"}
# <br>
# Now the output dataframe type is validated when you call `clean_data` at runtime
# so our test reduces to an execution test!

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Adaptability: define a base schema and build on top of it

# %%
class InputSchema(pa.SchemaModel):
    _categories = ["A", "B", "C"]  # store arbitrary metadata in private class attributes
    continuous: Series[float] = pa.Field(ge=0)
    categorical: Series[str] = pa.Field(isin=_categories)

    class Config:
        coerce = True

# %% slideshow={"slide_type": "fragment"}
class OutputSchema(InputSchema):
    categorical_one_hot: Series[int] = pa.Field(alias="one_hot_", regex=True)

    @pa.check("one_hot_")
    def categorical_one_hot_check(cls, series):
        return series.name[-1] in cls._categories

# %% slideshow={"slide_type": "fragment"}
@pa.check_types
def featurize_data(clean_data: DataFrame[InputSchema]) -> DataFrame[OutputSchema]:
    one_hot = pd.get_dummies(clean_data["categorical"], prefix="one_hot")
    return pd.concat([clean_data, one_hot], axis="columns")

display(featurize_data(raw_data).head(3))

# %% [markdown] slideshow={"slide_type": "notes"}
#
# Since, dataframes are complex objects, pandera focuses on making the process
# of defining schemas as concise as possible, offloading the concerns around
# column types and allowable values so you can focus more on the
# analysis/modeling logic.

# %% [markdown] slideshow={"slide_type": "subslide"}
#
# ### Portability: Support Other Dataframe Libraries and Schema Specifications in the Ecosystem
#
# [![github-issue](https://img.shields.io/badge/github_issue-420-blue?style=for-the-badge&logo=github)](https://github.com/pandera-dev/pandera/issues/420)
#
# Support frictionless data table schemas (✨ coming out in the `0.7.0` release ✨)

# %% [markdown]
# ```python
# from pandera.io import from_frictionless_schema
# 
# frictionless_schema = {
#     "fields": [
#         {
#             "name": "column_1",
#             "type": "integer",
#             "constraints": {"minimum": 10, "maximum": 99}
#         }
#     ],
#     "primaryKey": "column_1"
# }
# 
# pandera_schema = from_frictionless_schema(frictionless_schema)
# ```

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### 🛣 Roadmap Items

# %% [markdown] slideshow={"slide_type": "fragment"}
# - ✅ Decouple pandera and pandas type systems [![github-issue](https://img.shields.io/badge/github_issue-369-blue?style=for-the-badge&logo=github)](https://github.com/pandera-dev/pandera/issues/369)

# %% [markdown] slideshow={"slide_type": "fragment"}
# -  Abstract out parsing/validation logic to support non-pandas dataframes [![github-issue](https://img.shields.io/badge/github_issue-381-blue?style=for-the-badge&logo=github)](https://github.com/pandera-dev/pandera/issues/381)

# %% [markdown] slideshow={"slide_type": "fragment"}
# - Add Titles and Descriptions for SchemaModels [![github-issue](https://img.shields.io/badge/github_issue-331-blue?style=for-the-badge&logo=github)](https://github.com/pandera-dev/pandera/issues/331)

# %% [markdown] slideshow={"slide_type": "slide"}
#
# ### Principle 3: Generative Schemas Facilitate Property-based Testing
#
# You have a schema with a bunch of metadata about it... why not generate
# data for testing?

# %% slideshow={"slide_type": "fragment"}
display(InputSchema.example(size=3))

# %% slideshow={"slide_type": "fragment"}
input_schema_strategy = InputSchema.strategy(size=5)
print(input_schema_strategy)
print(type(input_schema_strategy))

# %% slideshow={"slide_type": "fragment"}
from hypothesis import given

@given(input_schema_strategy)
def test_featurize_data(clean_data):
    featurize_data(clean_data)

test_featurize_data()

# %% [markdown] slideshow={"slide_type": "fragment"}
# Automate the tedium of hand-writing mock dataframes for testing!

# %% [markdown] slideshow={"slide_type": "subslide"}
#
# #### Generate schemas as multi-purpose artifacts

# %% [raw]

<p>At run-time<p/>

<div class="mermaid">
graph LR
    R[(raw data)] -- raw schema --> P[data processor]
    P -- clean schema --> C[(clean data)]
    C -- clean schema --> F[featurizer]
    F -- feature schema --> X[(feature data)]
</div>

<p>At test-time<p/>

<div class="mermaid">
graph LR
    S(raw schema) --> R[(mock raw data)]
    R --> P[data processor]
    P -- clean schema --> O[output]
</div>

<div class="mermaid">
graph LR
    S(clean schema) --> R[(mock clean data)]
    R --> F[featurizer]
    F -- feature schema --> O[output]
</div>

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### 🛣 Roadmap Items

# %% [markdown] slideshow={"slide_type": "fragment"}
# - Add global schema-level override strategy [![github-issue](https://img.shields.io/badge/github_issue-561-blue?style=for-the-badge&logo=github)](https://github.com/pandera-dev/pandera/issues/561)

# %% [markdown] slideshow={"slide_type": "fragment"}
# - Support data synthesis strategies for joint distributions [![github-issue](https://img.shields.io/badge/github_issue-371-blue?style=for-the-badge&logo=github)](https://github.com/pandera-dev/pandera/issues/371)

# %% [markdown] slideshow={"slide_type": "fragment"}
# - Make Hypothesis strategies more efficient [![github-issue](https://img.shields.io/badge/github_issue-404-blue?style=for-the-badge&logo=github)](https://github.com/pandera-dev/pandera/issues/404)

# %% [markdown] slideshow={"slide_type": "slide"}
#
# ### Principle 4: Profile __Data__ _and_ Data Pipelines
#
# Pandera uses basic data profiling to infer a schema from realistic data

# %%
realistic_data = pd.DataFrame({"continuous": [1, 2, 3, 4, 5, 6]})
bootstrapped_schema = pa.infer_schema(realistic_data)
print(bootstrapped_schema)

# %% [markdown] slideshow={"slide_type": "subslide"}
# Write it out into a python file with `bootstrapped_schema.to_script("schemas.py")`

# %% tags=["hide_input"]
Markdown(
f"""
```python
{bootstrapped_schema.to_script()}
```
"""
)

# %% [markdown] slideshow={"slide_type": "subslide"}
# Write it out into a yaml file with `bootstrapped_schema.to_yaml("schema.yaml")`

# %% tags=["hide_input"]
Markdown(
f"""
```yaml
{bootstrapped_schema.to_yaml()}
```
"""
)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### 🛣 Roadmap Item
#
# Create schema from a `pandas-profiling` `ProfileReport` [![github-issue](https://img.shields.io/badge/github_issue-562-blue?style=for-the-badge&logo=github)](https://github.com/pandera-dev/pandera/issues/562)
#
# ```python
# from pandera.io import from_pandas_profile_report
#
# df = ...
# profile = ProfileReport(df)
# schema = from_pandas_profile_report(profile)
# ```

# %% [markdown] slideshow={"slide_type": "subslide"}
#
# ## Profile Data _and_ __Data Pipelines__
#
# If we have pandera schema type annotations...

# %% [markdown] slideshow={"slide_type": "fragment"}
# ```python
# import pandera as pa
# from pandera.typing import DataFrame as DF
# 
# from schemas import Raw, Clean, Training
#
# @pa.check_types
# def load() -> DF[Raw]:
#     ...
# 
# @pa.check_types
# def clean(raw_data: DF[Raw]) -> DF[Clean]:
#     ...
# 
# @pa.check_types
# def featurize(clean_data: DF[Clean]) -> DF[Training]:
#     ...
#
# @pa.check_types
# def train_model(training_data: DF[Training]):
#     ...
# ```

# %% [markdown] slideshow={"slide_type": "subslide"}
# We can potentially create a data flow graph

# %% [raw]
<div class="mermaid">
graph LR
    L[load] --DF-Raw--> C[clean]
    C --DF-Clean--> F[featurize]
    F --DF-Training--> T[train_model]
</div>

# %% [markdown] slideshow={"slide_type": "fragment"}
# <br>
# Collect coverage statistics of schema-annotated dataframes to identify
# points in the pipeline that lack dataframe type coverage
#
# | Function | Input Type | Output Type | Test Errors |
# | -------- | ---------- | ----------- | ----------- |
# | load | - | DF[Raw] | 0 |
# | clean | DF[Raw] | DF[Clean] | 1 |
# | featurize | DF[Clean] | DF[Train] | 7 |
# | train_model | DF[Train] | - | 2 |

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### 🛣 Roadmap Items

# %% [markdown] slideshow={"slide_type": "fragment"}
# - Parse schema-decorated functions to construct a dataflow graph [![github-issue](https://img.shields.io/badge/github_issue-392-blue?style=for-the-badge&logo=github)](https://github.com/pandera-dev/pandera/issues/392)

# %% [markdown] slideshow={"slide_type": "fragment"}
# - Implement error report aggregator [![github-issue](https://img.shields.io/badge/github_issue-425-blue?style=for-the-badge&logo=github)](https://github.com/pandera-dev/pandera/issues/425)

# %% [markdown] slideshow={"slide_type": "fragment"}
# - Implement CLI for pipeline profiling and reports [![github-issue](https://img.shields.io/badge/github_issue-426-blue?style=for-the-badge&logo=github)](https://github.com/pandera-dev/pandera/issues/426)

# %% [markdown] slideshow={"slide_type": "fragment"}
# - Create `pytest-pandera` plugin for profiling data pipelines in your test suite [![github-issue](https://img.shields.io/badge/github-repo-blue?style=for-the-badge&logo=github)](https://github.com/pandera-dev/pytest-pandera)

# %% [markdown] slideshow={"slide_type": "slide"}
# # 🛣 Roadmap
#
# | Guiding Principle | Description | Issue |
# | ----------------- | ----------- | ----- |
# | **Parse, then Validate** | Extend parsing functionality to support arbitrary transformations | [![github-issue](https://img.shields.io/badge/github_issue-252-blue?style=for-the-badge&logo=github)](https://github.com/pandera-dev/pandera/issues/252) |
# | **Make Schemas Reusable, Adaptable, and Portable** | Support Other Schema Specifications in the Ecosystem | [![github-issue](https://img.shields.io/badge/github_issue-420-blue?style=for-the-badge&logo=github)](https://github.com/pandera-dev/pandera/issues/420) |
# |  | Decouple pandera and pandas type systems | [![github-issue](https://img.shields.io/badge/github_issue-369-blue?style=for-the-badge&logo=github)](https://github.com/pandera-dev/pandera/issues/369) |
# |  | Abstract out parsing/validation logic to support non-pandas dataframes | [![github-issue](https://img.shields.io/badge/github_issue-381-blue?style=for-the-badge&logo=github)](https://github.com/pandera-dev/pandera/issues/381) |
# |  | Add Titles and Descriptions for SchemaModels | [![github-issue](https://img.shields.io/badge/github_issue-331-blue?style=for-the-badge&logo=github)](https://github.com/pandera-dev/pandera/issues/331) |
# | **Generative Schemas Facilitate Property-based Testing** | Add global schema-level override strategy | [![github-issue](https://img.shields.io/badge/github_issue-561-blue?style=for-the-badge&logo=github)](https://github.com/pandera-dev/pandera/issues/561) |
# |  | Support data synthesis strategies for joint distributions | [![github-issue](https://img.shields.io/badge/github_issue-371-blue?style=for-the-badge&logo=github)](https://github.com/pandera-dev/pandera/issues/371) |
# |  | Make Hypothesis strategies more efficient | [![github-issue](https://img.shields.io/badge/github_issue-404-blue?style=for-the-badge&logo=github)](https://github.com/pandera-dev/pandera/issues/404) |
# | **Profile Data and Data Pipelines** | Create schema from a `pandas-profiling` `ProfileReport` | [![github-issue](https://img.shields.io/badge/github_issue-562-blue?style=for-the-badge&logo=github)](https://github.com/pandera-dev/pandera/issues/562) |
# |  | Parse schema-decorated functions to construct a dataflow graph | [![github-issue](https://img.shields.io/badge/github_issue-392-blue?style=for-the-badge&logo=github)](https://github.com/pandera-dev/pandera/issues/392) |
# |  | Implement error report aggregator | [![github-issue](https://img.shields.io/badge/github_issue-425-blue?style=for-the-badge&logo=github)](https://github.com/pandera-dev/pandera/issues/425) |
# |  | Implement CLI for pipeline profiling and reports | [![github-issue](https://img.shields.io/badge/github_issue-426-blue?style=for-the-badge&logo=github)](https://github.com/pandera-dev/pandera/issues/426) |
# |  | Create `pytest-pandera` plugin for profiling data pipelines in your test suite | [![github-issue](https://img.shields.io/badge/github-repo-blue?style=for-the-badge&logo=github)](https://github.com/pandera-dev/pytest-pandera) |

# %% [markdown] slideshow={"slide_type": "slide"}
#
# # Where to Learn More
#
# - **Pycon [2021]** - Statistical Typing: A Runtime TypingSystem for Data Science and Machine Learning
#   - video: https://youtu.be/PI5UmKi14cM
# - **Scipy [2020]** - Statistical Data Validation of Pandas Dataframes
#   - video: https://youtu.be/PxTLD-ueNd4
#   - talk: https://conference.scipy.org/proceedings/scipy2020/pdfs/niels_bantilan.pdf
# - **Pandera Blog [2020]**: https://blog.pandera.ci/statistical%20typing/unit%20testing/2020/12/26/statistical-typing.html
# - **PyOpenSci Blog [2019]**: https://www.pyopensci.org/blog/pandera-python-pandas-dataframe-validation
# - **Personal Blog [2018]**: https://cosmicbboy.github.io/2018/12/28/validating-pandas-dataframes.html

# %% [markdown] slideshow={"slide_type": "slide"}
#
# # How to Contribute
#
# ![badge](https://img.shields.io/github/stars/pandera-dev/pandera?style=social)
# [![badge](https://img.shields.io/pypi/pyversions/pandera.svg)](https://pypi.python.org/pypi/pandera/)
# [![badge](https://img.shields.io/pypi/v/pandera.svg)](https://pypi.org/project/pandera/)
# ![badge](https://img.shields.io/github/contributors/pandera-dev/pandera)
# [![badge](https://pepy.tech/badge/pandera/month)](https://pepy.tech/project/pandera)
# [![badge](https://pepy.tech/badge/pandera)](https://pepy.tech/project/pandera)
#
# - **Repo**: https://github.com/pandera-dev/pandera
# - **Docs**: https://pandera.readthedocs.io
# - **Contributing Guide**: https://pandera.readthedocs.io/en/stable/CONTRIBUTING.html

# %% [markdown] slideshow={"slide_type": "fragment"}
#
# ##### Join the Scipy Mentored Sprints! 👟👟 

# %% [markdown] slideshow={"slide_type": "fragment"}
#
# ##### Toss a coin to your maintainer 👍🪙 https://github.com/sponsors/cosmicBboy

# %% [markdown] slideshow={"slide_type": "slide"}
#
# ### 🎉 Shoutouts to [pyopensci](https://www.pyopensci.org/) all the pandera contributors! 🎉
#
# # <img src="https://raw.githubusercontent.com/pandera-dev/pandera-presentations/master/static/pandera-growth-july-2021.png" width="400px" style="margin: auto;"/>
