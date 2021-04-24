# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
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
# # Statistical Typing: A Runtime Typing System for Data Science and Machine Learning
#
# ### Niels Bantilan
#
# Pycon, May 15th 2021

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Type systems help programmers reason about code and can make programs more computationally efficient.

# %% slideshow={"slide_type": "fragment"}
from typing import Union

Number = Union[int, float]

def add_and_double(x: Number, y: Number) -> Number:
    ...

# %% [markdown] slideshow={"slide_type": "fragment"}
# <br>
# Can you predict the outcome of these function calls?

# %% slideshow={"slide_type": "fragment"}
add_and_double(5, 2)
add_and_double(5, "hello")
add_and_double(11.5, -1.5)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 🤔 What would a type system geared towards data science and machine learning look like?

# %% [markdown] slideshow={"slide_type": "slide"}
#  ## Outline

# %% [markdown] slideshow={"slide_type": "fragment"}
# - 🐞🐞🐞 introduce you to some of my problems

# %% [markdown] slideshow={"slide_type": "fragment"}
# - 📊📈 define a specification for data types in the statistical domain

# %% [markdown] slideshow={"slide_type": "fragment"}
# - 🤯 make you realize that you've been doing statistical typing all along

# %% [markdown] slideshow={"slide_type": "fragment"}
# - 🛠 demonstrate one way it might be put into practice using `pandera`

# %% [markdown] slideshow={"slide_type": "fragment"}
# - 🏎 discuss where this idea can go next

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 🐞🐞🐞 An Introduction to Some of my Problems

# %% [markdown] slideshow={"slide_type": "slide"}
# ### The worst bugs are the silent ones, especially if they're in ML models that took a lot of ⏰ to train

# %% [markdown] slideshow={"slide_type": "fragment"}
# <br>
# - `model ~ data`

# %% [markdown] slideshow={"slide_type": "fragment"}
# - `Δdata -> Δmodel`

# %% [markdown] slideshow={"slide_type": "fragment"}
# - I define my `model` as a function `f(x) -> y`

# %% [markdown] slideshow={"slide_type": "fragment"}
# - Suppose I'm using `f` in an important business/scientific process

# %% [markdown] slideshow={"slide_type": "fragment"}
# - How do I know if `f` is working as intended?


# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Static Type-checking/Linting
#
# Catches certain type errors before running code

# %% slideshow={"slide_type": "fragment"}
from typing import Union

Number = Union[int, float]

def add_and_double(x: Number, y: Number) -> Number:
    return (x + y) * 2

# %% [markdown]
# ```python
# add_and_double(5, 2)        # ✅
# add_and_double(5, "hello")  # ❌
# add_and_double(11.5, -1.5)  # ✅
# ```

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Static Type-checking/Linting
#
# **Problem:** What if the underlying implementation is wrong?

# %% slideshow={"slide_type": "fragment"}
from typing import Union

Number = Union[int, float]

def add_and_double(x: Number, y: Number) -> Number:
    return (x - y) * 4

# %% [markdown]
# ```python
# add_and_double(5, 2)        # output: 12
# add_and_double(5, "hello")  # raises: TypeError
# add_and_double(11.5, -1.5)  # output: 52
# ```

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Unit Tests
#
# Unit tests verify the behavior of isolated pieces of functionality and
# let you know when changes cause breakages or regressions.

# %% slideshow={"slide_type": "fragment"}
import pytest

def test_add_and_double():
    # 🙂 path
    assert add_and_double(5, 2) == 14
    assert add_and_double(11.5, -15) == 20.0
    assert add_and_double(-10, 1.0) == -18.0

def test_add_and_double_exceptions():
    # 😞 path
    with pytest.raises(TypeError):
        add_and_double(5, "hello")
    with pytest.raises(TypeError):
        add_and_double("world", 32.5)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Property-based Tests
#
# Property-based testing alleviates the burden of explicitly writing test cases

# %% slideshow={"slide_type": "fragment"}
from hypothesis import given
from hypothesis.strategies import integers, floats, one_of, text

numbers = one_of(integers(), floats())

@given(x=numbers, y=numbers)
def test_add_and_double(x, y):
    assert add_and_double(x, y) == (x + y) * 2

@given(x=numbers, y=text())
def test_add_and_double_exceptions():
    with pytest.raises(TypeError):
        add_and_double(x, y)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### 🔎 ⛏ Testing code is hard, testing statistical analysis code is harder!

# %% [markdown] slideshow={"slide_type": "subslide"}
# #### Toy Example: Training a "Which Disney Character Are You?" Model

# %% [raw] slideshow={"slide_type": "fragment"}
<br>
<div class="mermaid">
    graph LR
      U[User] --creates--> S([Survey Response])
      S --store data--> D[(Database)]
      D --create dataset--> X[(Dataset)]
      X --train model--> M([Model])
</div>

# %% slideshow={"slide_type": "subslide"}
from typing import List, TypedDict

Response = TypedDict("Response", q1=int, q2=int, q3=str)
Example = List[float]

def store_data(raw_response: str) -> Response:
    ...

def create_dataset(raw_responses: List[Response], other_features: List[Example]) -> List[Example]:
    ...

def train_which_disney_character_are_you_model(survey_responses: List[Example]) -> str:
    ...

# %% [markdown] slideshow={"slide_type": "fragment"}
# <br>
# - `store_data`'s scope of concern is atomic, i.e. it only operates
#   on a single data point 🧘⚛

# %% [markdown] slideshow={"slide_type": "notes"}
# Easy to write test cases

# %% [markdown] slideshow={"slide_type": "fragment"}
# - `create_dataset` needs to worry about with the statistical patterns of a
#   sample of data points 😓📊

# %% [markdown] slideshow={"slide_type": "fragment"}
# - So what if I want to test `create_dataset` on plausible example data?

# %% [markdown] slideshow={"slide_type": "notes"}
# Difficult to write test case

# %% [markdown] slideshow={"slide_type": "slide"}
# ### 🤲 📀 🖼 hand-crafting example dataframes is a major barrier for unit testing.

# %% [markdown] slideshow={"slide_type": "fragment"}
# <br>
# .... it's not fun 😭

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### What if I could do something like...

# %% tags=["hide_input"]
from IPython.display import display

# %% slideshow={"slide_type": "fragment"}
import pandera as pa
from pandera.typing import Series

class Schema(pa.SchemaModel):
    variable1: Series[int] = pa.Field(ge=0)
    variable2: Series[float] = pa.Field(in_range={"min_value": 0, "max_value": 1})
    variable3: Series[str] = pa.Field(isin=list("abc"))

sample_data = Schema.example(size=5)
display(sample_data.head(3))

# %% slideshow={"slide_type": "fragment"}
sample_data["variable1"] = sample_data["variable1"] * -1
try:
    Schema.validate(sample_data)
except Exception as e:
    print(e)

# %% [markdown] slideshow={"slide_type": "notes"}
# I won't say much else here except for that I'm not a big fan. It's really
# tedious

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 📊📈 Define a Specification for Data Types in the Statistical Domain

# %% [markdown] slideshow={"slide_type": "slide"}
# > Statistical typing extends basic scalar data types with additional
# > semantics about the properties held by a collection of data points

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### `Boolean → Bernoulli`

# %% [markdown] slideshow={"slide_type": "fragment"}
# ```python
# x1 = True
# x2 = False
# ```

# %% [markdown] slideshow={"slide_type": "fragment"}
# ```python
# support: Set[bool] = {x1, x2}
# probability_distribution: Dict[str, float] = {True: 0.5, False, 0.5}
# FairCoin = Bernoulli(support, probability_distribution)
# ```

# %% [markdown] slideshow={"slide_type": "fragment"}
# ```python
# data: FairCoin = [1, 0, 0, 1, 1, 0]
#
# mean(data)
# chi_squared(data)
# ```

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### `Enum → Categorical`

# %% [markdown]
# ```python
# class Animal(Enum):
#     CAT = 1
#     DOG = 2
#     COW = 3
#     OTHER = 4
# ```

# %% [markdown] slideshow={"slide_type": "fragment"}
# ```python
# FarmAnimals = Categorical(
#     Animal,
#     probabilities={
#         Animal.CAT: 0.01,
#         Animal.DOG: 0.04,
#         Animal.COW: 0.5,
#         Animal.OTHER: 0.45,
#     },
#     ordered=False,
# )
# ```

# %% [markdown] slideshow={"slide_type": "fragment"}
# ```python
# data: FarmAnimals = [Animal.CAT] * 50 + [Animal.DOG] * 50
#
# check_type(data)  # raise a TypeError
# ```

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### `Int → Poisson`

# %% [markdown]
# ```python
# PatientsAdmitted = Poisson(expected_rate=10, interval=datetime.timedelta(days=1))
# ```

# %% [markdown] slideshow={"slide_type": "fragment"}
# ```python 
# data: List[int] = sample(PatientsAdmitted)
# ```

# %% [markdown] slideshow={"slide_type": "fragment"}
# ```python 
# assert all(x >= 0 for x in data)
# ```

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### `Float → Gaussian`

# %% [markdown]
# ```python
# TreeHeightMeters = Gaussian(mean=10, standard_deviation=1)
# ```

# %% [markdown] slideshow={"slide_type": "fragment"}
# ```python  
# def test_process_data():
#     data: List[float] = sample(TreeHeightMeters)
#     result = mean(data)
#     assert 8 <= result  <= 12
# ```

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Statistical Type Specification: Types as Schemas

# %% [markdown] slideshow={"slide_type": "fragment"}
# For each variable in my dataset, define:

# %% [markdown] slideshow={"slide_type": "fragment"}
# - **basic datatype**: `int`, `float`, `bool`, `str`, etc.

# %% [markdown] slideshow={"slide_type": "fragment"}
# - **deterministic properties**: domain of possible values, e.g. `x >= 0`

# %% [markdown] slideshow={"slide_type": "fragment"}
# - **probabilistic properties**: distributions that apply to the variable and
#   their sufficient statistics, e.g. `mean` and `standard deviation`

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Have you ever done something like this?

# %% slideshow={"slide_type": "fragment"}
import math

def normalize(x: List[float]):
    """Mean-center and scale with standard deviation"""
    mean = sum(x) / len(x)
    std = math.sqrt(sum((i - mean) ** 2 for i in x) / len(x))
    x_norm = [(i - mean) / std for i in x]

    # runtime assertions
    assert any(i < 0 for i in x_norm)
    assert any(i > 0 for i in x_norm)

    return x_norm


# %% [markdown] slideshow={"slide_type": "fragment"}
# #### 🤯 You've Been Doing Statistical Typing All Along

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Implications

# %% [markdown] slideshow={"slide_type": "fragment"}
# <br>
# Some statistical properties can be checked statically, e.g. the mean operation cannot be applied to categorical data
# ```python
# mean(categorical) ❌
# ```

# %% [markdown] slideshow={"slide_type": "fragment"}
# <br>
# Others can only be checked at runtime, e.g. this sample of data is drawn from a Gaussian of particular parameters
# ```python
# scipy.stats.normaltest(normalize(raw_data))
# ```

# %% [markdown] slideshow={"slide_type": "fragment"}
# <br>
# Schemas can be implemented as generative data contracts that can be used for type checking and sampling

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 🛠 Statistical Typing in Practice with `pandera`

# %% [markdown] slideshow={"slide_type": "subslide"}
# Suppose we're building a predictive model of house prices given features about different houses:

# %%
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

# %% [markdown] slideshow={"slide_type": "fragment"}
# <br>
# - `square_footage`: positive integer

# %% [markdown] slideshow={"slide_type": "fragment"}
# - `n_bedrooms`: positive integer

# %% [markdown] slideshow={"slide_type": "fragment"}
# - `property type`: categorical

# %% [markdown] slideshow={"slide_type": "fragment"}
# - 🎯 `price`: positive real number

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Pipeline

# %%
def process_data(raw_data):  # step 1: prepare data for model training
    ...
    
def train_model(processed_data): # step 2: fit a model on processed data
    ...

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Define Schemas with `pandera`

# %% slideshow={"slide_type": "fragment"}
import pandera as pa
from pandera.typing import Series, DataFrame

PROPERTY_TYPES = ["condo", "townhouse", "house"]


class BaseSchema(pa.SchemaModel):
    square_footage: Series[int] = pa.Field(in_range={"min_value": 0, "max_value": 3000})
    n_bedrooms: Series[int] = pa.Field(in_range={"min_value": 0, "max_value": 10})
    price: Series[int] = pa.Field(in_range={"min_value": 0, "max_value": 1000000})

    class Config:
        coerce = True


class RawData(BaseSchema):
    property_type: Series[str] = pa.Field(isin=PROPERTY_TYPES)


class ProcessedData(BaseSchema):
    property_type_condo: Series[int] = pa.Field(isin=[0, 1])
    property_type_house: Series[int] = pa.Field(isin=[0, 1])    
    property_type_townhouse: Series[int] = pa.Field(isin=[0, 1])

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Pipeline
#
# With Type Annotations

# %%

def process_data(raw_data: DataFrame[RawData]) -> DataFrame[ProcessedData]:
    ...
    
def train_model(processed_data: DataFrame[ProcessedData]):
    ...

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Pipeline
#
# With Implementation

# %%
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression


@pa.check_types
def process_data(raw_data: DataFrame[RawData]) -> DataFrame[ProcessedData]:
    return pd.get_dummies(
        raw_data.astype({"property_type": pd.CategoricalDtype(PROPERTY_TYPES)})
    )


@pa.check_types
def train_model(processed_data: DataFrame[ProcessedData]) -> BaseEstimator:
    return LinearRegression().fit(
        X=processed_data.drop("price", axis=1),
        y=processed_data["price"],
    )

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Running the Pipeline
#
# Validate the statistical type of raw and processed data every time we
# run our pipeline.

# %%
from io import StringIO


def run_pipeline(raw_data):
    processed_data = process_data(raw_data)
    estimator = train_model(processed_data)
    # evaluate model, save artifacts, etc...
    print("✅ model training successful!")


run_pipeline(pd.read_csv(StringIO(raw_data.strip())))


# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Fail Early and with Useful Information

# %%
invalid_data = """
square_footage,n_bedrooms,property_type,price
750,1,unknown,200000
900,2,condo,400000
1200,2,house,500000
"""

try:
    run_pipeline(pd.read_csv(StringIO(invalid_data.strip())))
except Exception as e:
    print(e)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Schemas as Generative Contracts
#
# Define property-based unit tests with `hypothesis`

# %%
from hypothesis import given


@given(RawData.strategy(size=3))
def test_process_data(raw_data):
    process_data(raw_data)

    
@given(ProcessedData.strategy(size=3))
def test_train_model(processed_data):
    estimator = train_model(processed_data)
    predictions = estimator.predict(processed_data.drop("price", axis=1))
    assert len(predictions) == processed_data.shape[0]

# %% [markdown] slideshow={"slide_type": "fragment"}
# <br>
# Run test suite

# %%
def run_test_suite():
    test_process_data()
    test_train_model()
    print("✅ tests successful!")    
    
run_test_suite()

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Catch Errors in Data Processing Code
#
# Define property-based unit tests with `hypothesis`

# %%
@pa.check_types
def process_data(raw_data: DataFrame[RawData]) -> DataFrame[ProcessedData]:
    return raw_data

try:
    run_test_suite()
except Exception as e:
    print(e)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Bootstrapping a Schema from Sample Data
#
# For some datasets, it might make sense to infer a schema from a sample of
# data and go from there:

# %% slideshow={"slide_type": "fragment"}
raw_df = pd.read_csv(StringIO(raw_data.strip()))
display(raw_df.head(3))

# %% slideshow={"slide_type": "fragment"}
schema = pa.infer_schema(raw_df)
schema.to_yaml()
schema.to_script()
print(schema)


# %% [markdown] slideshow={"slide_type": "subslide"}
# ## 🪛🪓🪚 Use Cases
#
# - CI tests for ETL/model training pipeline
# - Alerting for dataset shift
# - Monitoring model quality in production

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 🏎 Where Can this Idea Go Next?

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Statically analyze code that performs statistical operations
# ```python
# data: FarmAnimals = [Animal.CAT] * 50 + [Animal.DOG] * 50
# mean(data)  # ❌ cannot apply mean to Categorical
# ```
# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Infer model architecture space based on function signatures
# ```python
# def model(input_data: Normal) -> Bernoulli:
#     ...
#
# type(model)
# # [LogisticRegression, RandomForestClassifier, ...]
# ```

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Infer Statistical Types from Data
#
# Model-based statistical types
#
# %% [markdown] slideshow={"slide_type": "fragment"}
# - schema inference can be arbitrarily complex

# %% [markdown] slideshow={"slide_type": "fragment"}
# - statistical types can also be arbitrarily complex

# %% [markdown] slideshow={"slide_type": "fragment"}
# - data can be encoded as statistical models, and those model artifacts can be
#   used as components in a schema

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## GAN Schema
#
# In theory, a generative adversarial network can be used as a schema to validate
# real-world data and generate synthetic data

# %% [raw]
<div class="mermaid">
graph TB
  subgraph GAN Architecture
  G[generator]
  D[discriminator]
  end
  W[real world] --real data--> D
  G --synthetic data--> D
  D --> P[real/fake]
</div>

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## GAN Schema
#
# The discriminator, which is typically discarded after training, can validate
# real or upstream synthetic data.

# %% [raw]
<div class="mermaid">
graph LR
  subgraph GAN Schema
  G[generator]
  D[discriminator]
  end
  P[data processor] --real/synthetic data--> D
  G --synthetic data--> M[model trainer]
</div>

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Validation and Data Synthesis for Complex Statistical Types

# %% tags=["hide_input"]
dataframe = pd.DataFrame({
    "category": ["cat", "dog", "cow", "horse", "..."],
    "images": ["image1.jpeg", "image2.jpeg", "image3.jpeg", "image4.jpeg", "..."],
})
display(dataframe)

# %% [markdown] slideshow={"slide_type": "fragment"}
# ```python
# class ImageSchema(pa.SchemaModel):
#     category: Series[str] = pa.Field(isin=["cat", "dog", "cow", "horse", "..."])
#     images: Series[Image] = pa.Field(drawn_from=GenerativeAdversarialNetwork("weights.pt"))
# ```


# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Takeaway
#
# Statistical typing extends basic data types into the statistical domain,
# opening up a bunch of testing capabilities that make statistical code
# more robust and easier to reason about.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Thanks!
