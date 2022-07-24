# pandera-presentations

This repository contains presentations about `pandera`.

## Setup

Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) and [mamba](https://github.com/mamba-org/mamba), then install
the dev environment:

```
$ ./setup-dev-env.sh
```

Reveal slide presentations with speaker notes
[requires a local copy of reveal.js](https://nbconvert.readthedocs.io/en/latest/usage.html#convert-revealjs)

```bash
git clone https://github.com/hakimel/reveal.js.git slides/reveal.js
```

To convert a notebook to slides:
```bash
./nbconv-slide.sh notebooks/{my_notebook}.ipynb
```

To present with speaker notes and timer:
```bash
./nbconv-present.sh notebooks/{my_notebook}.ipynb
```

## Run Locally

Update dependencies

```
bundle update
```

Serve

```
bundle exec jekyll serve
```
