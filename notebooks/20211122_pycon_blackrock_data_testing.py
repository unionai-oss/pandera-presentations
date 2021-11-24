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
# # Pandera: A Data Testing Toolkit for Data Science & Machine Learning
#
# ### Niels Bantilan
#
# Pycon Blackrock, November 23rd 2021

# %% tags=["hide_input", "hide_output"] jupyter={"source_hidden": true}
import warnings

warnings.simplefilter("ignore")

import pandera as pa

# %% [markdown] slideshow={"slide_type": "slide"}
# # Background 🏞
#
# - 📜 B.A. in Biology and Dance
# - 📜 M.P.H. in Sociomedical Science and Public Health Informatics
# - 🤖 Machine Learning Engineer @ Union.ai
# - 🛩 Flytekit OSS Maintainer
# - ✅ Author and Maintainer of Pandera
# - 🛠 Make DS/ML practitioners more productive

# %% [markdown] slideshow={"slide_type": "slide"}
# # Outline 📝
#
# - 🤔 What's Data Testing?
# - ✅ Pandera Quickstart
# - 🚦 Guiding Principles
# - 🏔 Scaling Pandera
# - ⌨️ Conclusion: Statistical Typing
# - 🛣 Future Roadmap

# %% [markdown] slideshow={"slide_type": "slide"}
# # 🤔 What's Data Testing?
#
# The act of asking the question "are my data as I expect them to be?"

# %% [markdown] slideshow={"slide_type": "fragment"}
# It's the act of writing programs that assert properties about not only the
# data themselves, but also the functions that produce them.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### A Simple Example: Life Before Pandera
#
# `data_cleaner.py`

# %% tags=[]
import pandas as pd

raw_data = pd.DataFrame({
    "continuous": ["-1.1", "4.0", "10.25", "-0.1", "5.2"],
    "categorical": ["A", "B", "C", "Z", "X"],
})

def clean(raw_data):
    # do some cleaning 🧹✨
    clean_data = ...
    return clean_data

# %% [markdown] slideshow={"slide_type": "slide"}
# `test_data_cleaner.py`

# %%
import pytest

def test_clean():
    # assumptions about valid data
    mock_raw_data = pd.DataFrame({"continuous": ["1.0", "-5.1"], "categorical": ["X", "A"]})
    result = clean(mock_raw_data)
    
    # check that the result contains nulls
    assert result.isna().any(axis="columns").all()

    # check data types of each column
    assert result["continuous"].dtype == float
    assert result["categorical"].dtype == object
    
    # check that non-null values have expected properties
    assert result["continuous"].dropna().ge(0).all()
    assert result["categorical"].dropna().isin(["A", "B", "C"]).all()
    
    # assumptions about invalid data
    with pytest.raises(KeyError):
        invalid_mock_raw_data = pd.DataFrame({"categorical": ["A"]})
        clean(invalid_mock_raw_data)
    print("tests pass! ✅")


# %% [markdown] slideshow={"slide_type": "slide"}
# Let's implement the `clean` function:

# %%
def clean(raw_data):
    raw_data = pd.DataFrame(raw_data)
    # do some cleaning 🧹✨
    clean_data = (
        raw_data
        .astype({"continuous": float, "categorical": str})
        .assign(
            continuous=lambda _: _.continuous.mask(_.continuous < 0),
            categorical=lambda _: _.categorical.mask(~_.categorical.isin(["A", "B", "C"]))
        )
    )
    return clean_data

clean(raw_data)

# %%
test_clean()

# %% [markdown] slideshow={"slide_type": "slide"}
# <img src="https://raw.githubusercontent.com/pandera-dev/pandera/master/docs/source/_static/pandera-logo.png" width="125px" style="margin: 0;"/>
#
# <h2 style="margin-top: 0;">Pandera Quickstart</h2>
#
# An expressive and light-weight statistical validation tool for dataframes
#
# - Check the types and properties of dataframes
# - Easily integrate with existing data pipelines via function decorators
# - Synthesize data from schema objects for property-based testing

# %% [markdown] slideshow={"slide_type": "slide"} tags=["hide_output"]
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
# Complex Types with Modern Python, Inspired by [pydantic](https://pydantic-docs.helpmanual.io/) and `dataclasses`

# %%
from pandera.typing import Series

class CleanData(pa.SchemaModel):
    continuous: Series[float] = pa.Field(ge=0)
    categorical: Series[str] = pa.Field(isin=["A", "B", "C"])

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


# %% [markdown] slideshow={"slide_type": "slide"} tags=[]
# #### Meta: This presentation notebook is validated by pandera!

# %% [markdown] tags=[]
# ![mindblown](https://media.giphy.com/media/xT0xeJpnrWC4XWblEk/giphy-downsized-large.gif)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 🚦 Guiding Principles
#
# ### A Simple Example: Life After Pandera
#
# Let's define the types of the dataframes that we expect to see.

# %% [markdown]
# Here's `data_cleaner.py` again:

# %%
import pandera as pa
from pandera.typing import DataFrame, Series

class RawData(pa.SchemaModel):
    continuous: Series[float]
    categorical: Series[str]

    class Config:
        coerce = True


class CleanData(RawData):
    continuous = pa.Field(ge=0, nullable=True)
    categorical = pa.Field(isin=[*"ABC"], nullable=True)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Parse, then Validate
#
# Pandera guarantees that input and output dataframes fulfill the types and
# constraints as defined by type annotations

# %%
@pa.check_types
def clean(raw_data: DataFrame[RawData]) -> DataFrame[CleanData]:
    return raw_data.assign(
        continuous=lambda _: _.continuous.mask(_.continuous < 0),
        categorical=lambda _: _.categorical.mask(~_.categorical.isin(["A", "B", "C"]))
    )


# %%
clean(raw_data)


# %% [markdown] slideshow={"slide_type": "slide"}
# `test_data_cleaner.py`

# %%
def test_clean():
    # assumptions about valid data
    mock_raw_data = pd.DataFrame({"continuous": ["1.0", "-5.1"], "categorical": ["X", "A"]})
    
    # the assertions about the resulting data reduces to an execution test!
    clean(mock_raw_data)
    
    # assumptions about invalid data
    with pytest.raises(pa.errors.SchemaError):
        invalid_mock_raw_data = pd.DataFrame({"categorical": ["A"]})
        clean(invalid_mock_raw_data)
    print("tests pass! ✅")


# %% tags=[]
test_clean()


# %% [markdown] slideshow={"slide_type": "slide"}
# #### Maximize Reusability and Adaptability
#
# Once you've defined a schema, you can import it in other parts of your code
# base, like your test suite!

# %%
# data_cleaner.py
def clean(raw_data: DataFrame[RawData]) -> DataFrame[CleanData]:
    return raw_data.assign(
        continuous=lambda _: _.continuous.mask(_.continuous < 0),
        categorical=lambda _: _.categorical.mask(~_.categorical.isin(["A", "B", "C"]))
    )

# test_data_cleaner.py
def test_clean():
    # assumptions about valid data
    mock_raw_data = RawData(pd.DataFrame({"continuous": ["1.0", "-5.1"], "categorical": ["X", "A"]}))
    
    # the assertions about the resulting data reduces to an execution test!
    CleanData(clean(mock_raw_data))
    
    # assumptions about invalid data
    with pytest.raises(pa.errors.SchemaError):
        invalid_mock_raw_data = RawData(pd.DataFrame({"categorical": ["A"]}))
        clean(invalid_mock_raw_data)
    print("tests pass! ✅")
    
test_clean()


# %% [markdown] slideshow={"slide_type": "slide"}
# You can even represent dataframe joins!

# %%
class CleanData(RawData):
    continuous = pa.Field(ge=0, nullable=True)
    categorical = pa.Field(isin=[*"ABC"], nullable=True)
    
class SupplementaryData(pa.SchemaModel):
    discrete: Series[int] = pa.Field(ge=0, nullable=True)
        
class JoinedData(CleanData, SupplementaryData): pass


clean_data = pd.DataFrame({"continuous": ["1.0"], "categorical": ["A"]})
supplementary_data = pd.DataFrame({"discrete": [1]})
JoinedData(clean_data.join(supplementary_data))

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Bootstrap and Interoperate
#
# ##### Infer a schema definition from reference data

# %% tags=[]
clean_data = pd.DataFrame({
    "continuous": range(100),
    "categorical": [*"ABCAB" * 20]
})

schema = pa.infer_schema(clean_data)
print(schema)

# %% [markdown] slideshow={"slide_type": "slide"}
# ##### Write it to/from a yaml file

# %% tags=[] jupyter={"outputs_hidden": true}
yaml_schema = schema.to_yaml()
print(yaml_schema)

# %% tags=[] jupyter={"outputs_hidden": true}
print(schema.from_yaml(yaml_schema))

# %% [markdown] slideshow={"slide_type": "slide"}
# ##### Write it to a python script for further refinement using `schema.to_script()`

# %% jupyter={"outputs_hidden": true} tags=["hide_input"]
from IPython.display import display, Markdown
display(Markdown(
f"""
```python
{schema.to_script()}
```
"""
))

# %% [markdown] slideshow={"slide_type": "slide"}
# ##### Port schema from a [`frictionless`](https://specs.frictionlessdata.io/table-schema/) table schema

# %%
from pandera.io import from_frictionless_schema

frictionless_schema = {
    "fields": [
        {
            "name": "continuous",
            "type": "number",
            "constraints": {"minimum": 0}
        },
        {
            "name": "categorical",
            "type": "string",
            "constraints": {"isin": ["A", "B", "C"]}
        },
    ],
}
schema = from_frictionless_schema(frictionless_schema)
print(schema)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Facilitate Property-based Testing with Generative Schemas
#
# Generate valid examples under the schema's constraints

# %%
RawData.example(size=3)

# %%
CleanData.example(size=3)

# %% slideshow={"slide_type": "slide"}
# Transform your unit test suite!

from hypothesis import given


@pa.check_types
def clean(raw_data: DataFrame[RawData]) -> DataFrame[CleanData]:
    return raw_data.assign(
        continuous=lambda _: _.continuous.mask(_.continuous < 0),
        categorical=lambda _: _.categorical.mask(~_.categorical.isin(["A", "B", "C"]))
    )

@given(RawData.strategy(size=5))
def test_clean(mock_raw_data):
    clean(mock_raw_data)
    
    
class InvalidData(pa.SchemaModel):
    foo: Series[int]
    

@given(InvalidData.strategy(size=5))
def test_clean_errors(mock_invalid_data):
    with pytest.raises(pa.errors.SchemaError):
        clean(mock_invalid_data)
    

def run_test_suite():
    test_clean()
    test_clean_errors()
    print("tests pass! ✅")
    
    
run_test_suite()

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 🏔 Scaling Pandera
#
# ##### 🚧 _beta_ 🏗
#
# In `0.8.0`, pandera supports `dask`, `modin`, and `koalas` dataframes to scale
# data validation to big data.

# %%
display(raw_data)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Dask

# %%
import dask.dataframe as dd

dask_dataframe = dd.from_pandas(raw_data, npartitions=1)

try:
    CleanData(dask_dataframe, lazy=True).compute()
except pa.errors.SchemaErrors as exc:
    print(exc.failure_cases)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Koalas

# %%
import databricks.koalas as ks

koalas_dataframe = ks.DataFrame(raw_data)

try:
    CleanData(koalas_dataframe, lazy=True).compute()
except pa.errors.SchemaErrors as exc:
    print(exc.failure_cases)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Modin

# %%
import modin.pandas as mpd

modin_dataframe = mpd.DataFrame(raw_data)

try:
    CleanData(modin_dataframe, lazy=True).compute()
except pa.errors.SchemaErrors as exc:
    print(exc.failure_cases)

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
# ### Pandera is a Type System Geared Towards Data Science and Machine Learning

# %% [markdown] slideshow={"slide_type": "fragment"}
# It provides a flexible and expressive API for defining types for dataframes.

# %% [markdown] slideshow={"slide_type": "fragment"}
# This enables a more intuitive way of validating not only data, but also the functions that produce those data.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 🛣 Future Roadmap
#
# - Add support for other dataframe-like libraries, e.g: `pyarrow`, `xarray`, `polars`, `vaex`, `cudf`, etc.
# - Improve interoperability and integrations with data ecosystem, e.g. `pandas-profiling`, `fastapi`, `json-schema`
# - Implement CLI and `pytest-pandera` plugin for data pipeline profiling and reporting

# %% [markdown] slideshow={"slide_type": "slide"}
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
# - **Email**: [niels.bantilan@gmail.com](mailto:niels.bantilan@gmail.com)
# - **Repo**: https://github.com/pandera-dev/pandera
# - **Docs**: https://pandera.readthedocs.io
# - **Contributing Guide**: https://pandera.readthedocs.io/en/stable/CONTRIBUTING.html
