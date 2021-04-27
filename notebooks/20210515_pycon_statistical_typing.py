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

# %% [markdown] slideshow={"slide_type": "notes"}
# Hey everyone, I'm Niels Bantilan, and I'm excited to get the chance to present
# to you at Pycon this year. Just to give you a little background about myself, I'm one
# of the core maintainers of Flyte, which is an open source ML orchestration
# tool that helps ML/DS practitioners scale their workflows. I'm also the
# author of a dataframe validation tool called `pandera`, which is something
# I'll return to a little later. I want to start my presentation by making the
# claim that...

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Type systems help programmers reason about code and can make programs more computationally efficient.

# %% [markdown] slideshow={"slide_type": "notes"}
# Type systems help programmers reason about code and can make programs more
# computationally efficient, depending on the programming language. In the
# case of Python, which takes a gradual typing approach, you can opt in to
# using type hints but it's not necessary.

# %% slideshow={"slide_type": "fragment"}
from typing import Union

Number = Union[int, float]

def add_and_double(x: Number, y: Number) -> Number:
    ...

# %% [markdown] slideshow={"slide_type": "notes"}
# To see the benefits of type hints, if you take a look at this code snippet,
# you can see that we're defining a function called `add_and_double` which takes
# two numbers, either an `int` or a `float`, and produces another number,
# presumably one that's the sum of the two numbers multiplied by 2 as the
# function name suggests.

# %% [markdown] slideshow={"slide_type": "fragment"}
# <br>
# Can you predict the outcome of these function calls?

# %%
add_and_double(5, 2)
add_and_double(5, "hello")
add_and_double(11.5, -1.5)

# %% [markdown] slideshow={"slide_type": "notes"}
# Now I want you to take a few seconds to ask yourself: just with type
# hints and no actual implementation in the function body, can you predict the
# outcome of these function calls?
#
# We'll review this again later, but now I think we have enough context to
# get to the central question of my presentation, which is:

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 🤔 What would a type system geared towards data science and machine learning look like?

# %% [markdown] slideshow={"slide_type": "slide"}
#  ## Outline

# %% [markdown] slideshow={"slide_type": "notes"}
# And to address this question, I'm going to...

# %% [markdown] slideshow={"slide_type": "fragment"}
# - 🐞🐞🐞 introduce you to some of my problems

# %% [markdown] slideshow={"slide_type": "fragment"}
# - 📊📈 define a specification for data types in the statistical domain

# %% [markdown] slideshow={"slide_type": "fragment"}
# - 🛠 demonstrate one way it might be put into practice using `pandera`

# %% [markdown] slideshow={"slide_type": "fragment"}
# - 🏎 discuss where this idea can go next

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 🐞🐞🐞 An Introduction to Some of my Problems

# %% [markdown] slideshow={"slide_type": "notes"}
# So first let me introduce you to some of my problems.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### The worst bugs are the silent ones

# %% [markdown] slideshow={"slide_type": "notes"}
# The first one is that the worst bugs are the silent ones, especially if
# they're in ML models that took a lot of time to train.

# %% [markdown] slideshow={"slide_type": "fragment"}
# Especially if they're in ML models that took a lot of ⏰ to train

# %% [markdown] slideshow={"slide_type": "subslide"}
# #### The Model is the Data, the Data is the Model 🤔 🤯

# %% [markdown] slideshow={"slide_type": "notes"}
# To see why, consider the notion that statistical models are some kind of compression,
# representation, or approximation of the data.

# %% [markdown] slideshow={"slide_type": "fragment"}
# <br>
# - `model ~ data`

# %% [markdown] slideshow={"slide_type": "fragment"}
# - `Δdata -> Δmodel`

# %% [markdown] slideshow={"slide_type": "notes"}
# This implies that if the data changes, the model changes as well.

# %% [markdown] slideshow={"slide_type": "fragment"}
# - I define my `model` as a function `f(x) -> y`

# %% [markdown] slideshow={"slide_type": "notes"}
# So really, when I'm using a model for explanatory or predictive purposes,
# at a high level we're really using the model as a function that takes some
# input x and produces a result y.

# %% [markdown] slideshow={"slide_type": "fragment"}
# - How do I know if `f` is working as intended?

# %% [markdown] slideshow={"slide_type": "notes"}
# The question is, how do I know if `f` is working as intended?
#
# Now there're many answers to this question, but here I'd like to highlight
# two approaches that are well established in the software development world.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Static Type-checking/Linting
#
# Catches certain type errors before running code

# %% [markdown] slideshow={"slide_type": "notes"}
# The first approach is to catch certain type errors statically, even before
# running any code. So let's return to this example I should you in the beginning.
# Since Python 3.6 we've had type hints and tools like `mypy` can identify
# errors like the middle example which invokes `add_and_double` with an invalid
# type.

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

# %% [markdown] slideshow={"slide_type": "notes"}
# But what if the underlying implementation is wrong? Type hints can only
# get us so far because the first and third call to `add_and_double` are
# valid invocations but the actual output is incorrect because the function
# body implements the wrong logic.

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

# %% [markdown] slideshow={"slide_type": "notes"}
# This is where unit tests come in. Unit tests verify the behavior of isolated
# pieces of functionality and let you know when changes cause breakages or regressions.
# As you can see, I've divided up my tests in terms of happy and sad path
# tests.
#
# In the happy path tests, I've manually written examples of valid inputs and
# test that the output is correct, whereas in the sad path tests, I've defined
# cases that should raise some sort of exception. Note that these tests
# right here are actualy redundant since the type linter should be able to
# catch TypeErrors.

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

# %% [markdown] slideshow={"slide_type": "notes"}
# You might have noticed that it's burdensome to explicitly
# write test cases, so another thing we can do to be confident that a piece
# of functionality works as intended is to define property-based tests.
#
# Here I'm using the hypothesis library to define the interface of my function
# as typed strategies. When I run my test suite, under the hood hypothesis
# generates a bunch of data according to those types and attempts to falsify
# the assumption make in the test function body. If it does so, it then tries
# to find the smallest human-readable example that falsifies the test case.

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

# %% [markdown] slideshow={"slide_type": "notes"}
# This brings me to my next problem, which is that testing code is hard, but
# testing statistical analysis code is harder!

# %% [markdown] slideshow={"slide_type": "subslide"}
# #### Toy Example: Survey Data for Modeling

# %% [raw]
<br>
<div class="mermaid">
    graph LR
      U[User] --creates--> S([Survey Response])
      S --store data--> D[(Database)]
      D --create dataset--> X[(Dataset)]
      X --train model--> M([Model])
</div>

# %% [markdown] slideshow={"slide_type": "notes"}
# To see why, let's look at a toy example where we have a system that ingests
# survey data, stores responses in a database, creates a dataset, and trains
# a model to predict a target of interest.

# %% [markdown] slideshow={"slide_type": "subslide"}
# #### Toy Pipeline

# %%
from typing import List, Tuple, TypedDict

Response = TypedDict("Response", q1=int, q2=int, q3=str)
Example = Tuple[List[float], bool]

def store_data(raw_response: str) -> Response:
    ...

def create_dataset(raw_responses: List[Response], target: List[bool]) -> List[Example]:
    ...

def train_model(survey_responses: List[Example]) -> str:
    ...

# %% [markdown] slideshow={"slide_type": "notes"}
# Now here's what your pipeline might look like. To help with readability,
# I've defined two types:
# - one to represent the processed response
# - and another to represent a training example, which consists of a list of floats
#   for the features and a boolean value for the target.

# %% [markdown] slideshow={"slide_type": "fragment"}
# <br>
# - `store_data`'s scope of concern is atomic, i.e. it only operates
#   on a single data point 🧘⚛

# %% [markdown] slideshow={"slide_type": "notes"}
# If we just focus on the `store_data` and `create_dataset` functions, you
# may notice that `store_data`'s scope of concern is atomic, in that it
# only operates on a single data point.

# %% [markdown] slideshow={"slide_type": "fragment"}
# - `create_dataset` needs to worry about the statistical patterns of a
#   sample of data points 😓📊

# %% [markdown] slideshow={"slide_type": "notes"}
# On the other hand, `create_dataset` needs to worry
# about the overall statistical distribution of a data sample when it creates
# a dataset for modeling.

# %% [markdown] slideshow={"slide_type": "fragment"}
# - So what if I want to test `create_dataset` on plausible example data?

# %% [markdown] slideshow={"slide_type": "notes"}
# Going back to the idea of unit testing to be confident that our model
# works as intended, I'd ideally want to test it with some data that looks
# reasonably close to what I'd see in the real world. So what if I want
# to test `create_dataset` on a plausible example data?

# %% [markdown] slideshow={"slide_type": "slide"}
# ### 🤲 📀 🖼 hand-crafting example dataframes is a major barrier for unit testing.

# %% [markdown] slideshow={"slide_type": "notes"}
# Unfortunately the main answer here is you'd need to hand-craft example data,
# which often takes the form of pandas dataframes, and the most I'll say on
# this topic is that it's not fun...

# %% [markdown] slideshow={"slide_type": "fragment"}
# <br>
# .... it's not fun 😭

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### What if I could do something like...

# %% [markdown] slideshow={"slide_type": "notes"}
# What I'd like to be able to do is to specify the data types of my variables
# as a schema along with additional constraints.

# %% tags=["hide_input"]
from IPython.display import display
import pandas as pd

# %% slideshow={"slide_type": "fragment"}
import string
import pandera as pa
from pandera.typing import Series

class SurveySchema(pa.SchemaModel):
    q1: Series[int] = pa.Field(isin=[1, 2, 3, 4, 5])
    q2: Series[int] = pa.Field(isin=[1, 2, 3, 4, 5])
    q3: Series[str] = pa.Field(str_matches="[a-zA-Z0-9 ]+")

data = pd.DataFrame({"q1": [-1], "q2": [5], "q3": ["hello"]})

# %% [markdown] slideshow={"slide_type": "notes"}
# Here you can see that I'm importing pandera and I'm using it to I've defined a schema for
# the survey data in our toy example.
# 
# I'm making sure that questions 1 and 2 have values between 1 and 5 and
# question 3 matches some regular expression.

# %% slideshow={"slide_type": "fragment"}
try:
    SurveySchema.validate(data)
except Exception as e:
    print(e)

# %% [markdown] slideshow={"slide_type": "notes"}
# I can then use this schema to both validate the properties of some data at
# runtime

# %% slideshow={"slide_type": "fragment"}
sample_data = SurveySchema.example(size=3)
display(sample_data)

# %% [markdown] slideshow={"slide_type": "notes"}
# and also sample valid data under the schema's constraints for testing purposes.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 📊📈 Define a Specification for Data Types in the Statistical Domain

# %% [markdown] slideshow={"slide_type": "notes"}
# Now the code that you saw in the previous slide is all valid pandera syntax,
# but actually before diving further into pandera as a specific implementation,
# I want to define statistical typing more generally, with a
# few examples to illustrate what I mean.

# %% [markdown] slideshow={"slide_type": "slide"}
# > Statistical typing extends primitive data types with additional
# > semantics about the properties held by a collection of data points

# %% [markdown] slideshow={"slide_type": "notes"}
# I see statistical typing as a type system that extends
# primitive data types like booleans, strings, and floats with additional semantics
# about the properties held by a collection of data points.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### `Boolean → Bernoulli`

# %% [markdown] slideshow={"slide_type": "notes"}
# So for instance, the boolean data type consists of two possible
# values: true and false. In statistics speak, we would call this the support of
# a particular data distribution.
# 
# Now we can extend booleans to Bernoulli types, and to do that
# we'd need to supply one more piece of metadata, which is a probability
# mass function that maps values to probabilities, all of which sum to one.
#
# This is sufficient to specify a Bernoulli distribution that we can name, for example
# a `FairCoin` type, and you can imagine assigning a variable of this type
# to some data, and then performing statistical operations on it, for example
# getting the mean or mode of the data.

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
# mode(data)
# ```

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### `Enum → Categorical`

# %% [markdown] slideshow={"slide_type": "notes"}
# We can generalize booleans to enumerations, which gives us a way of expressing
# a type with a finite set of values, for example we can define a set of animals
# consisting of cats, dogs, and cows.
#
# Enums can be extended to Categorical types by providing a
# probability mass function as you saw in the previous slide, in addition
# to a flag that indicates whether the values are ordered or not. Here we
# define a FarmAnimals categorical type with a particular distribution of
# animals that you might find in a farm.
#
# You can imagine doing runtime type checks on data associated with the
# FarmAnimals type to make sure it follows the specified distribution.

# %% [markdown]
# ```python
# class Animal(Enum):
#     CAT = 1
#     DOG = 2
#     COW = 3
# ```

# %% [markdown] slideshow={"slide_type": "fragment"}
# ```python
# FarmAnimals = Categorical(
#     Animal,
#     probabilities={
#         Animal.CAT: 0.01,
#         Animal.DOG: 0.04,
#         Animal.COW: 0.95,
#     },
#     ordered=False,
# )
# ```

# %% [markdown] slideshow={"slide_type": "fragment"}
# ```python
# data: FarmAnimals = [Animal.CAT] * 50 + [Animal.DOG] * 50
#
# check_type(data)  # raise a RuntimeError
# ```

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### `Float → Gaussian`

# %% [markdown] slideshow={"slide_type": "notes"}
# The last example I wanted to go through is to extend floats into Gaussian
# types. This is conceptually simple because gaussian distributions only
# have two parameters, which is the mean and standard deviation of the distribution.
# Here we've defined a `TreeHeight` type whose mean is 10 and standard deviation
# is 1.
#
# The cool thing about these types is that, just like the property-based
# testing example that I showed you earlier, I could potentially use them
# to sample data for the purpose of unit testing. So If I have a function that
# processes data drawn from the TreeHeight distribution, I should be able to
# draw samples from this type in my unit tests, pass is to my `process_data`
# function, and then make assertions about the result.

# %% [markdown]
# ```python
# TreeHeight = Gaussian(mean=10, standard_deviation=1)
# ```

# %% [markdown] slideshow={"slide_type": "fragment"}
# ```python  
# def test_process_data():
#     data: List[float] = sample(TreeHeight)
#     result = process_data(data)
#     assert ...
# ```

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Have you ever done something like this?

# %% [markdown] slideshow={"slide_type": "notes"}
# Now I'd like to make the point that statistical typing isn't really new...
# when I first thought of the term, I googled around for some time and to my
# surprise I could only find one or two blog posts that specifically used
# the term in the same way that I was thinking about it.
#
# And to prove this point, I want you to consider the following code snippet,
# and don't worry too much about what the function actually does, just take
# a look at the end. If you've ever written assert statements that
# do runtime validations to check whether the output fulfills certain assumptions
# like value ranges or other such properties, then congratulations, you've been doing
# statistical typing all along.

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
# ### Statistical Type Specification: Types as Schemas

# %% [markdown] slideshow={"slide_type": "notes"}
# So to try and codify a few of these ideas, we can think of statistical
# types as potentially multivariate schemas where for each variable, I
# can define a few things:
#
# - the primitive data type that each element in the distribution belongs to
# - a set of deterministic properties that the type must adhere to
# - a set of probabilistic properties, such as the distributions that apply
#   to the variable along with their sufficient statistics.

# %% [markdown] slideshow={"slide_type": "fragment"}
# For each variable in my dataset, define:

# %% [markdown] slideshow={"slide_type": "fragment"}
# - **primitive datatype**: `int`, `float`, `bool`, `str`, etc.

# %% [markdown] slideshow={"slide_type": "fragment"}
# - **deterministic properties**: domain of possible values, e.g. `x >= 0`

# %% [markdown] slideshow={"slide_type": "fragment"}
# - **probabilistic properties**: distributions that apply to the variable and
#   their sufficient statistics, e.g. `mean` and `standard deviation`

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Implications

# %% [markdown] slideshow={"slide_type": "notes"}
# The implications of a fully implemented statistical type system are that:

# %% [markdown] slideshow={"slide_type": "fragment"}
# <br>
# Some statistical properties can be checked statically, e.g. the mean operation cannot be applied to categorical data
# ```python
# mean(categorical) ❌
# ```

# %% [markdown] slideshow={"slide_type": "fragment"}
# <br>
# Other properties can only be checked at runtime, e.g. this sample of data is drawn from a Gaussian
# ```python
# scipy.stats.normaltest(normalize(raw_data))
# ```

# %% [markdown] slideshow={"slide_type": "fragment"}
# <br>
# Schemas can be implemented as generative data contracts that can be used for type checking and sampling

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 🛠 Statistical Typing in Practice with `pandera`

# %% [markdown] slideshow={"slide_type": "notes"}
# To illustrate this last implication, it's time to look at a concrete example
# using pandera, which I'd say is a rough and incomplete implementation of statistical
# typing, but many of the ideas are there.

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

# %% [markdown] slideshow={"slide_type": "notes"}
# Suppose we're building a predictive model of house prices given features
# about different houses. You can see from the raw data that we're working
# with four variables.

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

# %% [markdown] slideshow={"slide_type": "notes"}
# And we can think of our pipeline in two steps:

# %%
def process_data(raw_data):  # step 1: prepare data for model training
    ...
    
def train_model(processed_data): # step 2: fit a model on processed data
    ...

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Define Schemas with `pandera`

# %% [markdown] slideshow={"slide_type": "notes"}
# Next we'll define schemas in pandera, which in practice requires a little bit of
# data exploration to get a sense of what the data look like.
#
# Here you can see we're defining a BaseSchema, which we'll use as the
# foundational type, and we'll have our raw and processed data inherit from it.
#
# From these schema definitions, you can immediately see which variables the
# raw and processed data have in common, but you can also see what the differences
# are. Namely, that the raw data has a `property_type` variable containing strings
# that need to be converted into a set of binary indicator variables.

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

# %% [markdown] slideshow={"slide_type": "notes"}
# We can then add type annotations to our `process_data` and `train_model`
# functions to make sure that the inputs and outputs conform with the schema
# definition.

# %%

@pa.check_types
def process_data(raw_data: DataFrame[RawData]) -> DataFrame[ProcessedData]:
    ...

@pa.check_types
def train_model(processed_data: DataFrame[ProcessedData]):
    ...

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Pipeline
#
# With Implementation

# %% [markdown] slideshow={"slide_type": "notes"}
# We can fill in our functions with actual implementations using pandas
# and sklearn, to process the data and train a model, respectively.

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

# %% [markdown] slideshow={"slide_type": "notes"}
# The cool thing about this is that our processing and training functions now
# inherently validate our raw and processed data every time we run our pipeline.

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

# %% [markdown] slideshow={"slide_type": "notes"}
# And, if we happen to ingest invalid data, the pipeline fails early and we're
# provided useful information about what exactly went wrong.

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

# %% [markdown] slideshow={"slide_type": "notes"}
# To circle back to property-based testing, pandera exposes a `strategy`
# method that compiles the schema metadata into a hypothesis strategy that
# you can use in your unit tests.

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

def run_test_suite():
    test_process_data()
    test_train_model()
    print("✅ tests successful!")    
    
run_test_suite()

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Catch Errors in Data Processing Code
#
# Define property-based unit tests with `hypothesis`

# %% [markdown] slideshow={"slide_type": "notes"}
# This becomes useful because in the case that we've implemented something
# wrong, we'll also get a useful error message. Here pandera is complaining
# that the output of `process_data` doesn't contain the variable
# `property_type_condo`.

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

# %% [markdown] slideshow={"slide_type": "notes"}
# Finally, you can even bootstrap a schema from a sample of data because it
# can be tedious to write a schema from scratch. All you have to do is
# call the `infer_schema` function, which you can then write out in yaml format
# or as a python script to further edit and refine.

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

# %% [markdown] slideshow={"slide_type": "notes"}
# To sum up the practical use cases of statistical typing and pandera in
# particular, you can use it in the context of continuous integration tests
# for your ETL or modeling pipelines, but you can also use it for alerting
# when detecting dataset shift, or monitoring the quality of your model's
# predictions in production.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 🏎 Where Can this Idea Go Next?

# %% [markdown] slideshow={"slide_type": "notes"}
# But to go from practical use cases into theoretical applications, I want
# to end with a few ideas that might be interesting to consider when thinking
# about the question of type systems for data science and machine learning.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Statically analyze code that performs statistical operations
# ```python
# FarmAnimals = Categorical(
#     Animal,
#     probabilities={
#         Animal.CAT: 0.01,
#         Animal.DOG: 0.04,
#         Animal.COW: 0.95,
#     },
#     ordered=False,
# )
#
# data: FarmAnimals = [Animal.CAT] * 50 + [Animal.DOG] * 50
# mean(data)  # ❌ cannot apply mean to Categorical
# ```

# %% [markdown] slideshow={"slide_type": "notes"}
# The first idea is that it would be really nice to be able to statically
# analyze code to call out cases where statistical operations don't make sense
# given some type.
#
# So returning to the FarmAnimal example for earlier, this would mean that
# even before running code, we could tell that computing the mean of a categorical
# distribution doesn't make sense and our type linter should be able to tell us that.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Infer model architecture space based on function signatures
# ```python
# def model(input_data: Gaussian) -> Bernoulli:
#     ...
#
# type(model)
# # [LogisticRegression, RandomForestClassifier, ...]
# ```

# %% [markdown] slideshow={"slide_type": "notes"}
# The second idea is that it would be possible to infer the model architecture
# space based on function signatures with statistical types. So in theory
# if I have a function that takes a Gaussian distribution as input and outputs
# a Bernoulli distribution, we should be able to infer the space of valid model
# architectures that could approximate this function.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Infer Statistical Types from Data
#
# Model-based statistical types

# %% [markdown] slideshow={"slide_type": "notes"}
# And finally, maybe the whackiest idea is to infer statistical types from
# data. It's not always the case that my data can be neatly characterized
# by the theoretical statistical distributions we know about today. For example,
# image data or natural language data might be drawn from a very complex
# manifold that would be impossible to write down manually.
#
# In this case we'd want a schema inference routine that can be arbitrarily
# complex, to describe statistical types that can also be arbitrarily complex,
# using data that can be encoded as a statistical model, where the model
# artifacts can lend themselves as components in a schema.

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

# %% [markdown] slideshow={"slide_type": "notes"}
# To illustrate this final point, let's consider generative adversarial
# networks. In theory, a GAN can be used as a schema to validate real-world
# data but also generate synthetic data. During training you draw real samples
# from the real world and fake data from a generator whose objective is to fool the
# discriminator into thinking that the its synthetic data is real.
# Conversely, the discriminator's objective is to accurately tell when a data point
# is real or fake.

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

# %% [markdown] slideshow={"slide_type": "notes"}
# Typically, after training a GAN people only care about the generator and discard
# the discriminator. However, in our case we can actually use the discriminator
# to validate data by telling us whether a particular data point is real or
# fake. We can also use the generator for its primary purpose of synthesizing
# data for testing our model training functions.

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

# %% [markdown] slideshow={"slide_type": "notes"}
# In effect, this would imply that we can implement validation and data
# synthesis modules for complex statistical types, for example, an image
# dataset with a particular set of categories describing those images.

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

# %% [markdown] slideshow={"slide_type": "notes"}
# In theory, it would be possible to define a schema like this, where we have
# an Image type that points to a jpeg file. At validation time, that file is
# read into memory and passed into a GAN to verify that the image is drawn from
# a similar distribution as what we saw during training time. This might be
# useful in production to at least warn us when an image is out of distribution.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Takeaway
#
# Statistical typing extends primitive data types into the statistical domain,
# opening up a bunch of testing capabilities that make statistical code
# more robust and easier to reason about.

# %% [markdown] slideshow={"slide_type": "notes"}
# So I'm hoping that in this last section I've given you some food for thought,
# and really the main takeaway I want to leave you with is that statistical
# typing extends primitive data types by enforcing a set of deterministic and
# probabilistic properties about a collection of data points.
# 
# This opens up a bunch of testing capabilities that make the code that we write as
# data scientists and ML practitioners more robust and easier to reason about.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Thanks!
#
# <i class="fa fa-envelope" aria-hidden="true"></i> niels.bantilan@gmail.com
# <br>
# <i class="fa fa-twitter" aria-hidden="true"></i> [@cosmicbboy](https://twitter.com/cosmicBboy)
# <br>
# <i class="fa fa-github" aria-hidden="true"></i> [cosmicBboy](https://github.com/cosmicBboy)

# %% [markdown] slideshow={"slide_type": "notes"}
# And with that I'd like to thank you for your time and attention.
#
# Please feel free to reach out to me via email or on twitter or github
#
# I hope you got something out of this talk, and I hope you enjoy the rest of the conference!
