# Scipy 2020 Conference Presentation Notes

## `pandera`: Statistical Data Validation of Pandas Dataframes

Hi everyone, I'm Niels Bantilan, and I'm very excited to have the chance to
talk to you at the scipy conference this year. I'm a machine learning engineer
at Talkspace, which is a mental health platform that connects people with
therapists. My work centers around using NLP in clinical applications of
machine learning, but today I want to talk to you about statistical data
validation of pandas dataframes.

## What's a Dataframe?

For those of you who're not so familiar with pandas or dataframes, a dataframe
is basically a table that you can manipulate programmatically, and `pandas`
is one of the de facto tools for the manipulation and analysis of tabular data
in Python.

Sometimes you may want to explicitly validate the contents and properties of a
dataframe because you care about data quality.

## What's Data Validation?

Data validation is simply the act of falsifying data against explicit
assumptions for some downstream purpose.

Consider the statement "all swans are white". To verify this statement, we'd
have to go out and find all the swans and find that all of them are white,
which would be virtually impossible. Another approach we can take is to
find the first instance of a black swan to falsify the statement.

## Why Do I Need it?

Practically speaking, I often find it difficult to reason about and debug
data processing pipelines, and I know it's good practice to ensure
data quality, especially when the end product informs business decisions,
supports scientific findings, or generates predictions in a production setting.

## Everyone has a personal relationship with their dataframes

And finally, as I like to say, everyone has a personal relationship with their
dataframes. So you might know it intimately at the time, but future you or
other maintainers of the codebase won't be able to understand the workings of
your data processing pipeline in a few months.

So to illustrate these points, I'd like to tell you a story, involving you!

## One day, you encounter an error log trail and decide to follow it...

The error is a `TypeError` involving a multiplication operation.

## And you find yourself at the top of a function...

Called `process_data`

## You look around, and see some hints of what had happened...

Namely, that the function is trying to compute `weekly_income` from
`hours_worked` and `wage_per_hour`

## You sort of know what's going on, but you want to take a closer look!

So you insert a breakpoint

## And you find some funny business going on...

You notice a negative value, which is a problem in itself, but you know that this
can't be the cause of the `TypeError`. You look a little more closely at the
type of the `hours_worked` column, and notice a string in one of the elements.

## You squash the bug and add documentation for the next weary traveler who happens upon this code.

Making sure that the columns are the expected type, and the `hours_worked`
values must be positive, converting negative values into `nan`s. Doing this,
you merge your commit, and all is well with the world.

## A few months later...

## You find yourself at a familiar function, but it looks a little different from when you left it...

The type coercion logic is gone, and you see these function decorators on
the `process_data` function.

## You look above and see what in_schema and out_schema are, finding a NOTE that a fellow traveler has left for you.

This way, it's immediately clear what should be in the `df` variable coming in
and out of `process_data`.

## Moral of the Story

The better you can reason about the contents of a dataframe, the faster you can debug.

The faster you can debug, the sooner you can focus on the tasks that you care about.


## Outline

In the rest of this talk, I'd like to take you a little more deeply
into what data validation means in theory and practice, briefly introduce
`pandera`, and take you through a case study that uses the library on a
real-world dataset. I'll then wrap up with a bit of a preview on experimental
features available today and some of the items in the roadmap for future
development.

## Data Validation in Theory and Practice

According to the European Statistical System, data validation is an activity in
which it is verified whether or not a combination of values is a member of a
set of acceptable value combinations.

## Data Validation in Theory and Practice

More formally, we can relate this definition to the notion of falsifiability.
Data validation is the act of defining a validation function `v` that takes
some data `x` as input and outputs either `True` or `False`. As van der Loo
et al. point out, `v` needs to be a surjective function, which is just a
fancy way of saying there must exists at least one value of `x` that maps onto
`True` at at least another value of `x` that maps onto `False`.

## Case 1: Unfalsifiable

To see why, consider the following two cases. In case one, `v` always returns
`True`, making the function unfalsifiable. This is effectively like not having
any validation function at all, since all it does is continue the execution of
your program. In case two, `v` always returns `False`, which is like saying
"my dataframe has an infinite number of rows and columns". There's no
practical implementation of such a function, making it unverifiable. You
really need this surjectivity property in order to meaningfully distinguish
between valid and invalid data.

## Types of Validation Rules

So within this space of meaningful validation functions, one way to think about
validation rules is in terms of technical checks, which have to do with 
properties of the data structure, e.g. `income` is a numeric variable, and
domain-specific checks, which have to do with properties specific to the
topic under study, e.g. the `age` variable must be in the range `0` and `120`.

Another way of thinking about validation rules is in statistical terms. On
one hand, you might define deterministic checks, which express hard-coded
logical rules, and on the other you might want to define probabilistic checks,
which explicitly incorporate randomness and distributional variability. In
these examples, you can see that we're roughly saying the same thing about
the mean age being between `30` and `40` years, but the probabilistic check
incorporates information about the sample size and variability of the
sample distribution to assert that the confidence bounds of the distribution
are between `30` and `40`.

## Statistical Type Safety

Considering these kinds of statistical validation rules leads us to think
about statistical type safety, which would be like logical type safety but
applied to the distributional properties of data to ensure that statistical
operations on those data are valid, for example checking whether:

- the training samples are IID
- the features are not multicolinear
- the variance of a feature is greater than some threshold
- or the training and test set labels are drawn from the same distribution

## Data Validation Workflow in Practice

Practically speaking, the data validation workflow may seem familiar. You
start with a goal in your head as to what you want the data to look like, you
write some code, and check if it's good enough for the purposes of your
analysis. If it is, you get on with it, but if it isn't you rinse and repeat
the process.

## User Story

Having experienced this loop many times and writing my own ad hoc validation
code, this is the user story I started out with.

> As a machine learning engineer who uses pandas every day, I want a data
> validation tool that's intuitive, flexible, customizable, and easy to
> integrate into my ETL pipelines so that I can spend less time worrying about
> the correctness of a dataframe's contents and more time training models.

## Introducing `pandera`

What I ended up with is a design-by-contract data validation library that
exposes an intuitive API for expressing dataframe schemas.

## Refactoring the Validation Function

From the very start, I took the que from existing data validation libraries
in Python and formulated the schema function `s` to behave like this: it takes
a validation function and data as input and outputs the data itself if the
validation function evaluates to `True` but raises an `Exception` otherwise.

## Why?

The main reason for this is compositionality. If we have some data processing
function `f`, we can compose it with our schema in various ways. We can:

- validate the raw data before it's processed by `f`
- validate the output of `f`
- and my favorite, make a data validation sandwich

## Architecture

Pictorially, this shows how you'd use the `pandera` schema to interleave data
validation with data processing logic.

## Case Study: Fatal Encounters Dataset

And to give you a hands on sense of how you might integrate this tool into your
workflow, I wanted to take you through an analysis of the fatal encounters
dataset. I've been following this is amazing work for the past couple of years,
headed by D. Brian Burghart and other collaborators to compile the most
comprehensive national database of people killed during interactions with law
enforcement. In light of recent events, I wanted to take this opportunity to
point people to this dataset because, unsurprisingly, it's hard to find
official records of such encounters online.

## Bird's Eye View

To give you an overview of this dataset, there are 26,000+ records of law
enforcement encounters that lead to death dating back to the year 2000, where
each record contains the demographics of the decedent, the cause and location of
death, a description of the encounter, and the court disposition of the case.

The question I want to ask in this analysis is:

## What factors are most predictive of the court ruling a case as "Accidental"?

To give you a sense of some of the circumstances that lead to these accidental
deaths, here are a few examples.

> Undocumented immigrant Roberto Chavez-Recendiz was fatally shot while Rivera was arresting him along with his brother and brother-in-law for allegedly being in the United States without proper documentation. The shooting was "possibly the result of what is called a 'sympathetic grip,' where one hand reacts to the force being used by the other," Chief Criminal Deputy County Attorney Rick Unklesbay wrote in a letter outlining his review of the shooting. The Border Patrol officer had his pistol in his hand as he took the suspects into custody. He claimed the gun fired accidentally.

> Andrew Lamar Washington died after officers tasered him 17 times within three minutes.

> Biddle and his brother, Drake Biddle, were fleeing from a Nashville Police Department officer at a high rate of speed when a second Nashville Police Department officer, James L. Steely, crashed into him head-on.

## Caveats

While I let the emotional impact of these examples sink in, some caveats.

I do want to emphasize that, as mentioned on the website, the records in this
dataset are the most comprehensive to date, but by no means is it a finished
project. Biases may be lurking everywhere!

I don't have any domain expertise in criminal justice!

The main purpose here is to showcase the capabilities of pandera.

And also, now that I've built this tool, I want to validate all the things.

## Notebook is Available Here

Before I dive in, if anyone's curious to take a look at this presentation and
this analysis, I've set up some binder notebooks that you can visit via these
bitly links to run and play around with the code as you see fit.

## Read the Data

The first step in any analysis is to read in the data. You can inspect the first
couple of rows to get a first impression of what's in there.

## Clean up column names

Before proceed further, we should probably clean up column names to make our
analysis more readable, and this is where it might be a good idea to define
a minimal schema definition.

## Minimal Schema Definition

All we're saying here is these are the columns that we're interested in
modeling. Note that the `dispositions_exclusions` column, which we're going to
use to derive our target variable, shouldn't be `nullable`.

We can use this schema in our `clean_columns` function by just `pipe`ing it
at the end of the method chain.

### Sidebar

If for some reason a column specified in our schema isn't there, `pandera`
will raise a `SchemaError` complaining about it.

## Explore the Data with `pandas-profiling`

Now that we've cleaned up the column names let's use `pandas-profiling` to
get a better sense of the distributions in this dataset. Just to give you
a sense of what's in the `dispositions_exclusions` column, these are the
categories related to the court decisions for each record.

## Declare the Training Data Schema

After doing some data exploration, you might arrive at a richer schema for
the training data, including assumptions about the `age` range, acceptable
values in the `gender`, `race`, and `cause_of_death` columns, and the types
of each variable, which we want to ensure with the `coerce` keyword argument.
Because we want to build a model predicting whether a case was ruled as
`accidental`, we'll need to derive this from the `dispositions_exclusions`
column.

## Serialize schema to yaml format

As a nifty feature of `pandera`, we can serialize the schema as a `yaml` file
so we can inspect the schema in more detail.

## Clean the Data

Next, we want to clean and normalize the data values before modeling it. This
involves cleaning strings, binarizing certain variables, and, most importantly,
deriving our target column `disposition_accidental`. Another thing to note
here is that we're removing any records that are `unreported`, `unknown`,
`pending` or that were ruled `suicide`. The reason for this last criterion is
to model cases where law enforcement was more directly linked to the death
event.

## Add the data validation 🥪

To ensure that our `clean_data` function does what it's supposed to do, we
can make a data validation sandwich by using the `check_input` and
`check_output` decorators.

## ValueError: Unable to coerce column to schema data type

So if we try to apply the `clean_data` function as it is on the raw data, we
can an error, which is actually an error that I came across while making this
presentation. The message could be a little clearer, but I can tell you right
now that it's because the `age` column is a `str` and not a numeric variable.

## Normalize the age column

So let's define a `normalize_age` function that converts the values in this
column to a float representing age in years. I've abstracted away a few of the
details for your convenience.

## Apply normalize_age inside the clean_data function.

And now we can insert the `normalize_age` function somewhere in the middle of our
`clean_data` function.

## Create Training Set

And with that change we've fulfilled the contract defined in
`training_data_schema`!

### Sidebar

If, for some reason, the data gets corrupted, `Check` failure cases are
reported as a dataframe indexed by failure case value. The `index` column
contains a list of indexes in the invalid dataframe for a particular failure
case, and the `count` summarizes how many failure cases there were in the
dataframe of a particular value.

### Fine-grained debugging

You can get finer-grained debugging by catching the `SchemaError` and
accessing the invalid `data` attribute, and you can also inspect the
`failure_cases` attribute, which is itself a dataframe referencing the `index`
in the invalid data and value causing the `SchemaError`.

## Summarize the Data

One summary statistic we'd be interested in looking at is the class balance of
the target variable. We can see here that it's 2.75%, which means the training
set is highly imbalanced...

To create a validation rule for this, we can use the `Hypothesis` class
to test that the proportion of `True` values in the `disposition_accidental`
column is equal 2.75% with an alpha value of 1%.

## Prepare Training and Test Sets

Now that we have our clean training data, it's time to split it into our training
and test sets. Here I'd like to highlight a few features in `pandera`. First,
we can define `SeriesSchema` for our target variable, which works a lot like
`Column` schemas but it operates exclusively on pandas `Series` objects.

The second feature is that we can create modified copies of
`DataFrameSchema`s by calling particular methods. In this case, we're going to
use the `remove_columns` method to create a `feature_schema`.

The third feature I want to highlight is the flexibility of the `check_input`
and `check_output` decorators. As you can see, the `split_training_data`
function should return a tuple of four arrays. We can specify an integer index
in `check_output` as the second argument to specify the pandas data structure
we want to verify with a particular schema. Here we're saying we want to
validate the first two outputs with the `feature_schema`, and the last two
outputs with the `target_schema`.

## Model the Data

Now it's time to model the data. We're going to use sklearn to do this, in
particular we'll make a `Pipeline` using the `ColumnTransformer` to
numericalize the features and a `RandomForestClassifier` to predict the target.

## DataFrameSchema -> ColumnTransformer 🤯

To define the `ColumnTransformer`, let's take advantage of the fact that we've
already declared a lot of the properties of our data with the `feature_schema`.
This is by no means a general solution, but the `column_transformer_from_schema`
function will do the job for this analysis. For categorical variables, we're
going to pull out the discrete categories from the `str` columns by accessing
the `statistics` attribute of the `Check` object that defines the
`allowed_values`.

## Define the transformer

And... it works!

## Define and fit the modeling pipeline

Okay, so let's define the estimator and fit the model. But wait! We can use
the `check_input` decorator here to validate the feature and target inputs
that go into the `pipeline.fit` method. In this case, we can specify the
data to validate with a string that references the `X` and `y` argument names
of the sklearn `fit` API.

As I warned you earlier, I have a hammer, and everything's a nail here.

## Evaluate the Model

We can apply the same pattern to model evaluation, where we can validate
the input and output of the `predict` method. Here we see we have a decent
model fit with a test set ROC AUC of 0.82 with some overfitting.

### Plot the ROC curves using an in-line schema

And in this plotting code I want to show you that you can even define in-line
schemas that aren't stored in a variable. This one just checks that the
false positive and true positive rates are values between 0 and 1.

## Audit the Model

To audit the model, we're going to use the `shap` package, which implements
the Shapley Additive Explanations method for obtaining model explanations
I don't have time to dig into the details of this method, but
at a high level `shap` values provide an estimate for a feature's positive
or negative contribution to the model output per instance.

First we create an `explainer` for our random forest estimator. Then, we
use a decorated version of the pipeline `transformer` to obtain an array
of test set features to compute our `shap_values`.

## What factors are most predictive of the court ruling a case as "Accidental"?

Returning to our research question, the way to look at the graph is the
following: along the y-axis we have the features that go into predicting
the `accidental` case disposition. Each dot is a single training instance,
where red means that the value for that feature is high. Since most of the
features here are binary, red means that the value is 1, blue means 0.

Higher values on the x-axis means that the feature value contributed to a
higher probability of an `accidental` case disposition. The thing to look out
for is one particular color being predominantly one side of the zero x-axis
marker.

For example, the `cause_of_death` by `vehicle` contributes to a higher
probability that the predicted disposition is `accidental`, so does being
tasered, asphyxiated or restrained, or belonging to the `race_unspecified`
category.

Conversely, the `cause_of_death` of gunshot, race being white, or asian
pacific islander contributes to a lower probability that the predicted
disposition is `accidental`.

## Write the Model Audit Schema

If we wanted to validate these findings, we can create a dataframe of feature
values and their associated `shap` value for each instance in the test set.

Then for each binary variable, we can define a two-sample ttest hypothesis test
asserting that having a value of `1` for a particular variable tends to have
a higher or lower set of shap values.

We can then programmatically construct a `model_audit_schema` that verifies
the properties that I outlined previously to see if the model explanations
are consistent with the hypotheses. We can see that, given this schema, our
model audit passes. We can use this same schema in future instantiations
of the model to see if the assumptions hold.

## More Questions 🤔

Before I conclude the analysis, I just want to point out a few questions that
you may want to consider if you want to take a look at these data yourself.

For example, the plot below provides a hint at the last question, which looks
at interaction effects between the variable `race_african_american_black` and
`symptoms_of_mental_illness`. My interpretation of this plot is that,
"if the decedent was black, showing symptoms of mental illness contributed to 
a higher predicted probability of the `accidental` case disposition, whereas
being not black and showing symptoms of mental illness contributed to a lower
predicted probability".

I'm sure there are a lot more questions that can be asked of this dataset, and
I'd like to invite you to take a look for yourself.

## Takeaways

In summary, there are four takeaways that I want to leave you with:

Data validation is a means to multiple ends: reproducibility, readability,
maintainability, and statistical type safety.

It's an iterative process between exploring the data, acquiring domain
knowledge, and writing validation code.

`pandera` schemas are executable contracts that enforce the statistical
properties of a dataframe at runtime and can be flexibly interleaved with data
processing and analysis logic.

`pandera` doesn't automate data exploration or the data validation process. The
user is responsible for identifying which parts of the pipeline are critical to
test and defining the contracts under which data are considered valid.

## Experimental Features

If you want to try out this package, I'd be curious to learn about the utility
of some experimental features that come with version 0.4.0 and up. The first
is schema inference, which produces a schema using a dataframe input, and the
second is the ability to write a schema to a yaml file or a python script.

## Roadmap: Feature Proposals

And in terms of features coming down the line, serving some pain points that
I have with training machine learning models with tabular data,
I thought it would be nice to have domain-specific schemas where you can
distinguish between target and feature variables, and then use that schema
to generate samples for testing model training code.

## Contributions are Welcome!

If you're interested in this project, there are many ways of contributing,
like improving documentation, submitting feature requests, bugs, and pull
requests!
