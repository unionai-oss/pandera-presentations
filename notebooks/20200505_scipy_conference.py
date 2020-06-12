# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: pandera-presentations
#     language: python
#     name: pandera-presentations
# ---

# %% [markdown] slideshow={"slide_type": "slide"}
# # `pandera`: Statistical Data Validation of Pandas Dataframes
#
# ### Niels Bantilan
#
# Scipy 2020

# %% slideshow={"slide_type": "skip"}
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint

from IPython.display import display, Image, Markdown
from IPython.core.display import HTML

sns.set_style(
    "white",
    {
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "axes.spines.right": False,
        "axes.spines.top": False,
    }
)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## What's a `DataFrame`?

# %% slideshow={"slide_type": "skip"}
import uuid

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
#     <img src="https://upload.wikimedia.org/wikipedia/commons/6/60/Black_Swans.jpg" alt="Pair of black swans swimming" height="480" width="300"
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
# - It can be difficult to reason about and debug data processing pipelines.

# %% [markdown] slideshow={"slide_type": "fragment"}
# - Ensuring data quality is critical in many contexts but the data
#   validation process can incur considerable cognitive and software
#   development overhead.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Everyone has a personal relationship with their dataframes

# %% slideshow={"slide_type": "skip"}
import numpy as np
import pandas as pd


def process_data(df):
    return (
        df.assign(
            weekly_income=lambda x: x.hours_worked * x.wage_per_hour
        )
    )

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
    return (
        df.assign(
            weekly_income=lambda x: x.hours_worked * x.wage_per_hour
        )
    )


# %% [markdown] slideshow={"slide_type": "slide"}
# ### You sort of know what's going on, but you need to take a closer look!

# %%
def process_data(df):
    import pdb; pdb.set_trace()
    return (
        df.assign(
            weekly_income=lambda x: x.hours_worked * x.wage_per_hour
        )
    )

# %% [markdown] slideshow={"slide_type": "slide"}
# ### And you find some funny business going on...

# %%
>>> print(df)

# %%
>>> df.dtypes

# %%
>>> df.hours_worked.map(type)


# %% [markdown] slideshow={"slide_type": "slide"}
# ### You squash the bug and add documentation for the next weary traveler who happens upon this code.

# %%
def process_data(df):
    return (
        df
        # make sure these columns are floats
        .astype({"hours_worked": float, "wage_per_hour": float})
        # replace negative values with nans, since nans
        # are imputed downstream of this function.
        .assign(
            hours_worked=lambda x: x.hours_worked.where(
                x.hours_worked >= 0, np.nan
            )
        )
        .assign(
            weekly_income=lambda x: x.hours_worked * x.wage_per_hour
        )
    )


# %%
>>> process_data(df)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### A few months later...

# %% [markdown] slideshow={"slide_type": "slide"}
# ### You find yourself at a familiar function, but it looks a little different from when you left it...

# %% slideshow={"slide_type": "skip"}
# This needs to be here, but skipped for story-telling effect in the slides
import pandera as pa
from pandera import Column, Check

in_schema = pa.DataFrameSchema({
    "hours_worked": Column(pa.Float, coerce=True, nullable=True),
    "wage_per_hour": Column(pa.Float, coerce=True, nullable=True),
})

out_schema = (
    in_schema
    .update_column("hours_worked", checks=Check.greater_than_or_equal_to(0))
    .add_columns({"weekly_income": Column(pa.Float, nullable=True)})
)


# %%
@pa.check_input(in_schema)
@pa.check_output(out_schema)
def process_data(df):
    return (
        # replace negative values with nans, since nans
        # are imputed downstream of this function.
        df.assign(
            hours_worked=lambda x: x.hours_worked.where(
                x.hours_worked >= 0, np.nan
            )
        )
        .assign(
            weekly_income=lambda x: x.hours_worked * x.wage_per_hour
        )
    )


# %% [markdown] slideshow={"slide_type": "slide"}
# ### You look above and see what these `in_schema` and `out_schema` objects are about, finding a `NOTE` that a fellow traveler has left for you.

# %%
import pandera as pa
from pandera import DataFrameSchema, Column, Check

# NOTE: this is what's supposed to be in `df` going into `process_data`
in_schema = DataFrameSchema({
    "hours_worked": Column(pa.Float, coerce=True, nullable=True),
    "wage_per_hour": Column(pa.Float, coerce=True, nullable=True),
})

# ... and this is what `process_data` is supposed to return.
out_schema = (
    in_schema
    .update_column("hours_worked", checks=Check.greater_than_or_equal_to(0))
    .add_columns({"weekly_income": Column(pa.Float, nullable=True)})
)

@pa.check_input(in_schema)
@pa.check_output(out_schema)
def process_data(df):
    ...


# %% [markdown] slideshow={"slide_type": "slide"}
# ## Moral of the Story
#
# The better you can reason about the contents of a dataframe,
# the faster you can debug. The faster you can debug, the sooner
# you can focus on downstream tasks that you care about.

# %% [markdown] slideshow={"slide_type": "slide"}
# # Outline
#
# - Data validation in theory and practice.
# - Introduction to `pandera`
# - End-to-end example using the "Fatal Encounters" dataset
# - Roadmap

# %% [markdown] slideshow={"slide_type": "slide"}
# # Data Validation in Theory and Practice

# %% [markdown]
# According to the European Statistical System:
#
# > Data validation is an activity in which it is verified whether or not a combination of values is a member of a set of acceptable value combinations.
# > [[Di Zio et al. 2015](https://ec.europa.eu/eurostat/cros/system/files/methodology_for_data_validation_v1.0_rev-2016-06_final.pdf)]

# %% [markdown] slideshow={"slide_type": "slide"}
# # Data Validation in Theory and Practice
#
# More formally, we can relate this definition to one of the core principles
# of the scientific method: **falsifiability**
#
# $v(x) \twoheadrightarrow \{ {True}, {False} \}$
#
# Where $x$ is a set of data values and $v$ is a surjective validation
# function, meaning that there exists at least one $x$ that maps onto each
# of the elements in the set $\{True, False\}$ [[van der Loo et al. 2019](https://arxiv.org/pdf/1912.09759.pdf)].
#

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Case 1: Unfalsifiable
#
# $v(x) \rightarrow True$
#
# > **Example:** "my dataframe can have any number of columns of any type"
# > ```python
# > lambda df: True
# > ```

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Case 2: Unverifiable
#
# $v(x) \rightarrow False$
#
# > **Example:** "my dataframe has an infinite number of rows and columns"
# > ```python
# > lambda df: False
# > ```

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Types of Validation Rules
#
# ### Technical Checks
#
# Have to do with the meta-properties of the data structure:
#
# - `income` is a numeric variable, `occupation` is a categorical variable.
# - `email_address` should be unique.
# - null (missing) values are permitted in the `occupation` field.
#
# ### Domain-specific Checks
#
# Have to do with properties specific to the topic under study:
#
# - the `age` variable should be between the range 0 and 120
# - the `income` for records where `age` is below the legal
#   working age should be `nan`
# - certain categories of `occupation` tend to have higher `income` than others

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Types of Validation Rules
#
# ### Deterministic Checks
#
# Checks that express hard-coded logical rules
#
# > the mean `age` should be between `30` and `40` years old.
# > ```python
# > lambda age: 30 <= age.mean() <= 40
# > ```
#
# ### Probabilistic Checks
#
# Checks that explicitly incorporate randomness and distributional variability
#
# > the 95% confidence interval or mean `age` should be between `30` and `40` years old.
# > ```python
# > def prob_check_age(age):
# >     mu = age.mean()
# >     ci = 1.96 * (age.std() / np.sqrt(len(age))
# >     return 30 <= mu - ci and mu + ci <= 40
# > ```

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Data Validation Workflow

# %% tags=["hide_input"] language="html"
# <img src="../figures/pandera_process.png", width=275
#  style="display: block; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "slide"}
# # User Story

# %% [markdown]
# > As a machine learning engineer who uses `pandas` every day,
# I want a data validation tool that's intuitive, flexible,
# customizable, and easy to integrate into my ETL pipelines
# so that I can spend less time worrying about the correctness
# of a dataframe's contents and more time training models.

# %% [markdown] slideshow={"slide_type": "slide"}
# <img src="https://raw.githubusercontent.com/pandera-dev/pandera/master/docs/source/_static/pandera-logo.png" width=150>
#
# # Introducing `pandera`
#
# A design-by-contract data validation package that exposes an intuitive
# interface for expressing dataframe schemas.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Refactoring the Validation Function

# %% [markdown]
# $s(v, x) \rightarrow \begin{cases} \mbox{x,} & \mbox{if } v(x) = true \\ \mbox{error,} & \mbox{otherwise} \end{cases}$
#
# Where $s$ is a *schema* function that takes two arguments: the validation function $v$
# and some data $x$.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Why?
#
# **Compositionality**
#
# Consider a data processing function $f(x) \rightarrow x'$ that cleans the raw dataset $x$.
#
# We can use the schema to define any number of composite functions:

# %% [markdown] slideshow={"slide_type": "fragment"}
# - $f(s(x))$ first validates the raw data to catch invalid data before it's preprocessed.

# %% [markdown] slideshow={"slide_type": "fragment"}
# - $s(f(x))$ validates the output of $f$ to check that the processing function is fulfilling the contract defined in $s$

# %% [markdown] slideshow={"slide_type": "fragment"}
# - $s(f(s'(x))$: the "data validation sandwich"

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Architecture

# %% tags=["hide_input"] language="html"
# <img src="../figures/pandera_architecture.png", width=275
#  style="display: block; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Pandera Basics
#
# ### Step 1: Define a `DataFrameSchema`

# %%
import pandera as pa
from pandera import Column, Check

schema = pa.DataFrameSchema(
    {
        "hours_worked": Column(
            pa.Float, [
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(60),
            ],
            nullable=True
        ),
        "wage_per_hour": Column(
            pa.Float, Check.greater_than_or_equal_to(15), nullable=True
        ),
    },
    coerce=True,
)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Step 2: Call the `schema` on some data

# %%
import pandas as pd

dataframe = pd.DataFrame(
    {
        "hours_worked": [38.5, 41.25, "35.0", 27.75, 22.25, 20.5],
        "wage_per_hour": [15.1, 15, 21.30, 17.5, "19.50", 25.50],
    },
    index=pd.Index([str(uuid.uuid4())[:7] for _ in range(6)], name="person_id")
)

schema(dataframe)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Useful Error Reporting
#
# Expected column not present

# %%
no_column_df = dataframe.drop("hours_worked", axis="columns")

try:
    schema(no_column_df)
except pa.errors.SchemaError as exc:
    print(exc)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Useful Error Reporting
#
# Column type cannot be coerced to expected data type.

# %%
incorrect_type_df = dataframe.assign(hours_worked="string")

try:
    schema(incorrect_type_df)
except ValueError as exc:
    print(exc)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Useful Error Reporting
#
# Empirical column property is falsified

# %%
falsified_column_check = dataframe.copy()
falsified_column_check.loc[-2:, "wage_per_hour"] = -20

try:
    schema(falsified_column_check)
except pa.errors.SchemaError as exc:
    print(exc)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Debugging Support
#
# Inspect invalid data at runtime.

# %%
falsified_column_check = dataframe.copy()
falsified_column_check.loc[-2:, "wage_per_hour"] = -20

try:
    schema(falsified_column_check)
except pa.errors.SchemaError as exc:
    print("Invalid data\n", exc.data, "\n")
    print("Failure cases\n", exc.failure_cases)

# %% [markdown] slideshow={"slide_type": "slide"}
# # Case Study: Fatal Encounters Dataset
#
# _"A step toward creating an impartial, comprehensive, and searchable national
# database of people killed during interactions with law enforcement."_
#
# https://fatalencounters.org

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Bird's Eye View
#
# - 26,000+ records of law enforcement encounters leading to death
# - Records date back to the year 2000
# - Each record contains:
#   - demographics of the decedent
#   - cause and location of the death
#   - agency responsible for death
#   - court disposition of the case (e.g. "Justified", "Accidental", "Suicide", "Criminal")

# %% [markdown] slideshow={"slide_type": "slide"}
# ### What factors are most predictive of the court ruling a case as "Accidental"?

# %% [markdown] slideshow={"slide_type": "fragment"}
# Example 1:
#
# > Undocumented immigrant Roberto Chavez-Recendiz, of Hidalgo, Mexico, was fatally shot while Rivera was arresting him along with his brother and brother-in-law for allegedly being in the United States without proper documentation. The shooting was "possibly the result of what is called a 'sympathetic grip,' where one hand reacts to the force being used by the other," Chief Criminal Deputy County Attorney Rick Unklesbay wrote in a letter outlining his review of the shooting. The Border Patrol officer had his pistol in his hand as he took the suspects into custody. He claimed the gun fired accidentally.

# %% [markdown] slideshow={"slide_type": "fragment"}
# Example 2:
# > Andrew Lamar Washington died after officers tasered him 17 times within three minutes.

# %% [markdown] slideshow={"slide_type": "fragment"}
# Example 3:
# > Perry Simmons died after he was tasered and maced as up to six officers attempted to restrain him and bring him into custody.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Caveats
#
# - As mentioned on the [website](https://fatalencounters.org), the records in this dataset are not comprehensive. Biases may be lurking everywhere!
# - I don't have any domain expertise in criminal justice!
# - The purpose here is to showcase the capabilities of `pandera`, not provide earth-shattering or actionable insights.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Read the Data

# %%
import janitor
import requests
from pandas_profiling import ProfileReport

dataset_url = (
    "https://docs.google.com/spreadsheets/d/"
    "1dKmaV_JiWcG8XBoRgP8b4e9Eopkpgt7FL7nyspvzAsE/export?format=csv"
)
fatal_encounters = pd.read_csv(dataset_url, skipfooter=1, engine="python")


# %% tags=["hide_input"]
with pd.option_context("display.max_columns", 500):
    display(fatal_encounters.head(3))

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Explore the Raw Data with [`pandas-profiling`](https://github.com/pandas-profiling/pandas-profiling)

# %% tags=["hide_input"]
profile = ProfileReport(fatal_encounters, minimal=True)
profile.set_variable("html.navbar_show", False)
profile.to_notebook_iframe()

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Declare the Training Data Schema

# %% slideshow={"slide_type": "skip"}
genders = ["female", "male", "transgender", "transexual"]
races = [
    "african-american/black", "asian/pacific islander",
    "european-american/white", "hispanic/latino", "middle eastern",
    "native american/alaskan", "race unspecified",
]
causes_of_death = [
    'asphyxiated/restrained', 'beaten/bludgeoned with instrument',
    'burned/smoke inhalation', 'chemical agent/pepper spray',
    'drowned', 'drug overdose', 'fell from a height', 'gunshot',
    'medical emergency', 'other', 'stabbed', 'tasered', 'undetermined',
    'unknown', 'vehicle'
]

# %%
schema = pa.DataFrameSchema(
    {
        "age": Column(pa.Float, Check.in_range(0, 120), nullable=True),
        "gender": Column(pa.String, Check.isin(genders), nullable=True),
        "race": Column(pa.String, Check.isin(races), nullable=True),
        "cause_of_death": Column(
            pa.String,
            Check.isin(causes_of_death),
            nullable=True
        ),
        "symptoms_of_mental_illness": Column(pa.Bool, nullable=True),
        "disposition_accidental": Column(pa.Bool),
    },
    coerce=True
)

# %% slideshow={"slide_type": "subslide"} tags=["hide_input"]
display(HTML("Genders"))
pprint(genders)
display(HTML("Races"))
pprint(races, compact=True, width=78)
display(HTML("Causes of Death"))
pprint(causes_of_death, compact=True, width=79)


# %% [markdown] slideshow={"slide_type": "slide"}
# ### Clean the Data

# %%
def clean_columns(df):
    return (
        df.clean_names()
        .rename(
            columns=lambda x: (
                x.strip("_")
                .replace("&", "_and_")
                .replace("subjects_", "")
                .replace("location_of_death_", "")
                .replace("_resulting_in_death_month_day_year", "")
                # logic below applies to the following fields:
                # - "symptoms_of_mental_illness"
                # - "dispositions_exclusions"
                # gonna use them anyway 😬
                .replace("_internal_use_not_for_analysis", "")
            )
        )
    )


# %% [markdown] slideshow={"slide_type": "slide"}
# ### Clean the Data

# %% slideshow={"slide_type": "skip"}
def binarize_mental_illness(symptoms_of_mental_illness: pd.Series):
    return (
        symptoms_of_mental_illness
        .mask(lambda s: s.dropna().str.contains("unknown"))
        != "no"
    )

def normalize_age_range(age: pd.Series, pattern: str = "-|/| or "):
    to_normalize = age.str.contains(pattern) & age.notna()
    return age.mask(
        to_normalize,
        age[to_normalize].str.split(pattern, expand=True).astype(float).mean(axis=1)
    )

def normalize_age_to_year(age: pd.Series, pattern: str, denom: int):
    to_normalize = age.str.contains(pattern) & age.notna()
    return age.mask(
        to_normalize,
        age[to_normalize].str.replace(pattern, "").astype(float) / denom
    )

def normalize_age(age):
    return (
        age.str.replace("s|`", "")
        .pipe(normalize_age_range)
        .pipe(normalize_age_to_year, "month|mon", 12)
        .pipe(normalize_age_to_year, "day", 365)
        .astype(float)
    )

def filter_out_suicide():
    pass

def compute_disposition_accidental(dispositions_exclusions):
    return (
        dispositions_exclusions.str.contains("accident", case=False)
    )


# %%
@pa.check_output(schema)
def clean_data(df):
    return (
        df.dropna(subset=["dispositions_exclusions"])
        .transform_columns(
            [
                "gender", "race", "cause_of_death",
                "symptoms_of_mental_illness", "dispositions_exclusions"
            ],
            lambda s: s.str.lower(),
            elementwise=False
        )
        .transform_column(
            "symptoms_of_mental_illness",
            binarize_mental_illness,
            elementwise=False
        )
        .transform_column("age", normalize_age, elementwise=False)
        .transform_column(
            "dispositions_exclusions",
            compute_disposition_accidental,
            "disposition_accidental",
            elementwise=False
        )
        .query("gender != 'white'")  # probably a data entry error
        .filter_string(
            "dispositions_exclusions",
            "unreported|unknown|pending|suicide",
            complement=True
        )
    )


# %% [markdown] slideshow={"slide_type": "slide"}
# ### Apply Cleaning Functions

# %%
fatal_encounters_clean = fatal_encounters.pipe(clean_columns).pipe(clean_data)
display(fatal_encounters_clean.filter(list(schema.columns)).head(3))
nrows, ncols = fatal_encounters_clean.shape
display(Markdown(f"**rows**: {nrows}, **cols**: {ncols}"))

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Summarize the Data

# %% [markdown]
# What percent of cases in the training data are "accidental"?

# %% tags=["hide_input"]
percent_accidental = fatal_encounters_clean.disposition_accidental.mean()
display(Markdown(f"{percent_accidental * 100:0.02f}%"))

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Hypothesis**: "the `disposition_accidental` target has a
# class balance of ~2.75%"

# %%
from pandera import Hypothesis

target_schema = Column(
    pa.Bool,
    name="disposition_accidental",
    checks=Hypothesis.one_sample_ttest(
        popmean=0.0275, relationship="equal", alpha=0.01
    )
)

target_schema(fatal_encounters_clean);

# %% [markdown] slideshow={"slide_type": "slide"}
# What's the age distribution in the dataset?

# %% tags=["hide_input"]
fatal_encounters_clean.age.plot.hist(figsize=(8, 5)).set_xlabel("age");

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Hypothesis** check: "the `age` column is right-skewed"

# %%
from scipy import stats

stat, p = stats.skewtest(fatal_encounters_clean.age, nan_policy="omit")

age_schema = Column(
    pa.Float,
    name="age",
    checks=Hypothesis(
        test=stats.skewtest,
        # positive stat means distribution is right-skewed
        # null hypothesis: distribution is drawn from a gaussian
        relationship=lambda stat, p: stat > 0 and p < 0.01
    ),
    nullable=True,
)

age_schema(fatal_encounters_clean);

# %% [markdown] slideshow={"slide_type": "slide"}
# What's the race distribution in the dataset?

# %% tags=["hide_input"]
fatal_encounters_clean.race.value_counts().sort_values().plot.barh();

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Check**: "The top 3 most common races are `european-american/white`,
# `african-american/black`, and `hispanic/latino` in that order"

# %%
most_common_races = [
    "european-american/white",
    "african-american/black",
    "hispanic/latino",
]

race_schema = Column(
    pa.String,
    name="race",
    checks=Check(
        lambda race: (
            race.mask(race == "race unspecified")
            .dropna()
            .value_counts()
            .sort_values(ascending=False)
            .head(len(most_common_races))
            .index.tolist() == most_common_races
        )
    ),
    nullable=True,
)

race_schema(fatal_encounters_clean);

# %% [markdown] slideshow={"slide_type": "slide"}
# Is there an association between `symptoms_of_mental_illness` and `cause_of_death`?

# %% tags=["hide_input"]
with sns.plotting_context(context="notebook", font_scale=1.1):
    ax = (
        fatal_encounters_clean
        .assign(
            cause_of_death=lambda x: (
                x.cause_of_death
                .mask(lambda x: x == "unknown")
            )
        )
        .groupby("cause_of_death")
        .symptoms_of_mental_illness
        .apply(lambda x: x.value_counts().pipe(lambda s: s / s.sum()))
        .unstack(1)
        .sort_index(axis="index")
        .sort_index(axis="columns", ascending=False)
        .fillna(0)
        .sort_values(True)
        .plot.barh(figsize=(6, 8), stacked=True)
    )
    ax.set_ylabel("Cause of Death")
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title="mental illness/drug use")
    ax.axvline(0.5, color="k", linewidth=1.5, linestyle=":")
    sns.despine(bottom=True, left=True)

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Validation Rule**: "the proportion of people who died of certain
# causes like drug overdose, being tasered, and falling from a height
# have a greater than random chance of showing symptoms of mental illness."

# %% [markdown]
# **Deterministic Check:**

# %%
causes_of_death_assoc_with_mental_illness = [
    "drug overdose", "tasered", "burned/smoke inhalation",
    "asphyxiated/restrained",  "medical emergency",
    "beaten/bludgeoned with instrument",
]

def check_gt_random_mental_illness(groups):
    return pd.Series({
        group: mental_illness.mean() > 0.5
        for group, mental_illness in
        groups.items()
    })

deterministic_mental_illness_schema = Column(
    pa.Bool,
    name="symptoms_of_mental_illness",
    checks=Check(
        check_gt_random_mental_illness,
        groupby="cause_of_death",
        groups=causes_of_death_assoc_with_mental_illness,
    )
)
deterministic_mental_illness_schema(fatal_encounters_clean);

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Probabilistic Check:**

# %%
from statsmodels.stats.proportion import proportions_ztest

def hypothesis_gt_random_mental_illness(sample):
    return proportions_ztest(
        sample.sum(), sample.shape[0],
        alternative="larger", value=0.5,
    )

probabilistic_mental_illness_schema = Column(
    pa.Bool,
    name="symptoms_of_mental_illness",
    checks=[
        Hypothesis(
            hypothesis_gt_random_mental_illness,
            groupby="cause_of_death",
            samples=cause_of_death,
            relationship=lambda stat, p: p < 0.01,
            error=f"failed > random test: '{cause_of_death}'",
            raise_warning=True,
        )
        for cause_of_death in causes_of_death_assoc_with_mental_illness
    ]
)
probabilistic_mental_illness_schema(fatal_encounters_clean);

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Prepare Training and Test Sets

# %%
from sklearn.model_selection import train_test_split

target_schema = pa.SeriesSchema(
    pa.Bool,
    name="disposition_accidental",
    checks=Hypothesis.one_sample_ttest(
        popmean=0.0275, relationship="equal", alpha=0.01
    )
)
feature_schema = schema.remove_columns([target_schema.name])


@pa.check_input(schema)
@pa.check_output(feature_schema, 0)
@pa.check_output(feature_schema, 1)
@pa.check_output(target_schema, 2)
@pa.check_output(target_schema, 3)
def split_training_data(fatal_encounters_clean):
    return train_test_split(
        fatal_encounters_clean[list(feature_schema.columns)],
        fatal_encounters_clean[target_schema.name],
        test_size=0.2
    )


X_train, X_test, y_train, y_test = split_training_data(fatal_encounters_clean)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Model the Data

# %% [markdown]
# Import the tools

# %%
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


# %% [markdown] slideshow={"slide_type": "subslide"}
# Create a transformer to numericalize the features using a `schema` object.

# %%
def column_transformer_from_schema(feature_schema):

    def transformer_from_column(column):
        column_schema = feature_schema.columns[column]
        if column_schema.pandas_dtype is pa.String:
            return make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OneHotEncoder(categories=[get_categories(column_schema)])
            )
        if column_schema.pandas_dtype is pa.Bool:
            return SimpleImputer(strategy="median")

        # otherwise assume numeric variable
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler()
        )
    
    return ColumnTransformer([
        (column, transformer_from_column(column), [column])
        for column in feature_schema.columns
    ])

def get_categories(column_schema):
    for check in column_schema.checks:
        if check.name == "isin":
            return check.statistics["allowed_values"]
    raise ValueError("could not find Check.isin")


# %% slideshow={"slide_type": "skip"}
from sklearn import set_config

set_config(display='diagram')

# %% [markdown] slideshow={"slide_type": "subslide"}
# Define the transformer

# %%
transformer = column_transformer_from_schema(feature_schema)

# %% tags=["hide_input"]
transformer

# %% [markdown] slideshow={"slide_type": "subslide"}
# Define and fit the modeling pipeline

# %%
pipeline = Pipeline([
    ("transformer", transformer),
    (
        "estimator",
        RandomForestClassifier(
            class_weight="balanced_subsample",
            n_estimators=500,
            min_samples_leaf=20,
            min_samples_split=10,
            max_depth=10,
            random_state=100,
        )
    )
])

pipeline.fit(X_train, y_train)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Evaluate the Model

# %%
from sklearn.metrics import roc_auc_score, roc_curve

predict_fn = pa.check_input(feature_schema)(pipeline.predict_proba)

yhat_train = pipeline.predict_proba(X_train)[:, 1]
print(f"train ROC AUC: {roc_auc_score(y_train, yhat_train):0.04f}")

yhat_test = pipeline.predict_proba(X_test)[:, 1]
print(f"test ROC AUC: {roc_auc_score(y_test, yhat_test):0.04f}")

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Audit the Model
#
# `shap` package: https://github.com/slundberg/shap

# %% slideshow={"slide_type": "skip"}
import itertools

def feature_names_from_schema(column, schema):

    if schema.columns[column].pandas_dtype is pa.String:
        return [
            f"{column}_{x.replace('-', '_').replace('/', '_').replace(' ', '_')}"
            for x in get_categories(schema.columns[column])
        ]

    # otherwise assume numeric feature
    return [column]


feature_names = list(
    itertools.chain.from_iterable(
        feature_names_from_schema(column, feature_schema)
        for column in feature_schema.columns
    )
)

# %% [markdown] slideshow={"slide_type": "subslide"}
# Create an `explainer` object

# %%
import shap

transform_fn = pa.check_input(feature_schema)(
    pipeline.named_steps["transformer"].transform
)

X_test_array = transform_fn(X_test).toarray()

explainer = shap.TreeExplainer(
    pipeline.named_steps["estimator"],
    feature_perturbation="tree_path_dependent",
)
shap_values = explainer.shap_values(X_test_array, check_additivity=False)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### What factors are most predictive of the court ruling a case as "Accidental"?

# %% tags=["hide_input"]
shap.summary_plot(
    shap_values[1],
    X_test_array,
    feature_names,
    plot_type="dot",
    max_display=17,
    plot_size=(10, 6),
    alpha=0.8
)

# %% [markdown]
# The probability of the case being ruled as `accidental` ⬆️ if the `cause_of_death` is `vehicle,` `tasered`, `asphyxiated_restrained`, `medical_emergency`, or `drug_overdose`, or `race` is unspecified.
#
# The probability of the case being ruled as `accidental` ⬇️ if the `cause_of_death` is `gunshot` or `race` is `european_american_white`, `hispanic_latino` or `asian_pacific_islander`.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Write the Model Audit Schema

# %% [markdown]
# Create a dataframe with `{var}` and `{var}_shap` as columns

# %%
audit_dataframe = (
    pd.concat(
        [
            pd.DataFrame(X_test_array, columns=feature_names),
            pd.DataFrame(shap_values[1], columns=[f"{x}_shap" for x in feature_names])
        ],
        axis="columns"
    ).sort_index(axis="columns")
)

audit_dataframe.head(3)


# %% [markdown] slideshow={"slide_type": "subslide"}
# Define two sample t-test that tests the relative impact of a
# variable on the output probability of the model

# %%
def hypothesis_accident_probability(feature, increases=True):
    relationship = "greater_than" if increases else "less_than"
    return {
        f"{feature}_shap": Column(
            checks=Hypothesis.two_sample_ttest(
                sample1=1,
                sample2=0,
                groupby=feature,
                relationship=relationship,
                alpha=0.01,
            )
        ),
        feature: Column(checks=Check.isin([1, 0])),
    }


# %% [markdown] slideshow={"slide_type": "subslide"}
# Programmatically construct the schema and validate `feature_shap_df`.

# %%
columns = {}
# increases probability of disposition "accidental"
for column in [
    "cause_of_death_vehicle",
    "cause_of_death_tasered",
    "cause_of_death_asphyxiated_restrained",
    "cause_of_death_medical_emergency",
    "cause_of_death_drug_overdose",
    "race_race_unspecified",
]:
    columns.update(hypothesis_accident_probability(column, increases=True))
    
# decreases probability of disposition "accidental"
for column in [
    "cause_of_death_gunshot",
    "race_european_american_white",
    "race_hispanic_latino",
    "race_asian_pacific_islander",
]:
    columns.update(hypothesis_accident_probability(column, increases=False))

model_audit_schema = pa.DataFrameSchema(columns)

if isinstance(model_audit_schema(audit_dataframe), pd.DataFrame):
    print("Model audit results pass! ✅")
else:
    print("Model audit results fail ❌")

# %% [markdown] slideshow={"slide_type": "slide"}
# # Takeaways

# %% [markdown] slideshow={"slide_type": "fragment"}
# - Data validation is a means to a several ends: _reproducibility_, _readability_, and _maintainability_

# %% [markdown] slideshow={"slide_type": "fragment"}
# - It's an iterative process between exploring the data, acquiring domain knowledge, and writing validation code.

# %% [markdown] slideshow={"slide_type": "fragment"}
# - `pandera` schemas are executable contracts that enforce the statistical properties of a dataframe
#   at runtime and can be flexibly interleaved with data processing logic.

# %% [markdown] slideshow={"slide_type": "slide"}
# # Experimental Features
#
# **Schema Inference**
#
# ```python
# schema = pa.infer_schema(dataframe)
# ```
#
# **Schema Serialization**
#
# ```python
# pa.io.to_yaml(schema, "schema.yml")
# pa.io.to_script(schema, "schema.py")
# ```

# %% [markdown] slideshow={"slide_type": "slide"}
# # Roadmap: Feature Proposals

# %% [markdown]
# **Express tolerance level for `Check` objects when they return a boolean `Series`**
#
# ```python
# # up to 10% of cases are allowed to fail
# my_check = Check(lambda s: s < 1, tolerance=0.1)
# ```
#
# **Define domain-specific schemas, types, and checks, e.g. for machine learning**
#
# ```python
# # validate a regression model dataset
# schema = pa.machine_learning.supervised.TabularSchema(
#     targets={"regression": pa.TargetColumn(type=pa.ml_dtypes.Continuous)},
#     features = {
#         "continuous": pa.FeatureColumn(type=pa.ml_dtypes.Continuous),
#         "categorical": pa.FeatureColumn(type=pa.ml_dtypes.Categorical),
#         "ordinal": pa.FeatureColumn(type=pa.ml_dtypes.Ordinal),
#     }
# )
# ```
#
# **Generate synthetic data based on schema definition as constraints**
#
# ```python
# dataset = schema.generate_samples(100)
# X, y = dataset[schema.features], dataset[schema.targets]
# estimator.fit(X, y)
# ```

# %% [markdown] slideshow={"slide_type": "slide"}
# # Contributions are Welcome!

# %% [markdown]
# **Repo:** https://github.com/pandera-dev/pandera
#
# 1. Improving documentation
# 1. Submit feature requests (e.g. additional built-in `Check` and `Hypothesis` methods)
# 1. Submit new issues or pull requests on Github

# %% [markdown] slideshow={"slide_type": "slide"}
# # Thank you!
#
# [@cosmicBboy](https://twitter.com/cosmicBboy)
