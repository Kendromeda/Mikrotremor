# _mHVSR-Vs30_ - A Set of Data-driven models to predict Vs30 from mHVSR

> Kushal Sharma Wagle, Joseph P. Vantassel [jpvantassel.com](https://www.jpvantassel.com/), Adrian Rodriguez-Marek,

## Table of Contents

-   [About _mHVSR-Vs30_](#About-mHVSR-Vs30)
-   [Getting Started](#Getting-Started)
-   [Data pipeline](#Data-pipeline)

## About _mHVSR-Vs30_

`mHVSR-Vs30` is a collection of data-driven models to predict the
time averaged shear wave velocity in the upper 30 m (Vs30), from 
microtremor horizontal-to-vertical spectral ratio (mHVSR). The developed
models developed include both low-dimensional (`low_dim_models.ipynb`) and
high-dimensional (`high_dim_models.ipynb`). The details of the model's
development and performance are presented in the reference below.

## Citation

If you use these tools in your research or consulting, we ask you please cite the
following:

> Sharma Wagle, K., Vantassel, J.P., Rodriguez-Marek, A., and Anbazhagan, B. (2026). "A Set of Data-Driven Models to Predict VS30 from the Horizontal-to-Vertical Spectral Ratio of Microtremors". Bulletin of the Seismological Society of America. [https://doi.org/10.1785/0120250128](https://doi.org/10.1785/0120250128)

## Getting Started

This repository has two supported workflows. Use the **pipeline package** for
data curation, quality control, preprocessing, and reproducible training. Use
the **legacy notebooks** only to explore the published prediction examples.
They have separate dependency files and should be installed separately.

### Pipeline package (recommended)

The pipeline requires Python 3.12 or later and is locked with
[uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync
```

This creates or updates the project virtual environment from `pyproject.toml`
and the committed `uv.lock`; it also installs the development tools used below.
Do not install the package workflow from `requirements.txt`.

Run the fast local verification suite with:

```bash
uv run ruff check src tests
uv run mypy src/mhvsr_vs30
uv run pytest -m "not requires_real_data"
```

Run the complete suite, including tests that require the local
`datasets_global/` tree when that data is available:

```bash
uv run pytest --cov=mhvsr_vs30 --cov-report=term-missing
uv build
```

For the end-to-end pipeline commands, see
[the data-pipeline guide](docs/DATA_PIPELINE.md). The package command is
available as `uv run mhvsr-vs30`.

### Legacy notebooks

The notebooks, `low_dim_models.ipynb` and `high_dim_models.ipynb`, are an
independent, no-code introduction to the published prediction models. They do
not use the `mhvsr_vs30` package pipeline.

They require Python 3.12 or later. Install their notebook-specific dependencies
with `pip`:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` is retained exclusively for this legacy notebook workflow;
it is intentionally not a replacement for the locked package environment.
MacOS users may also need `libomp` (`brew install libomp` or
`conda install -c conda-forge libomp`) for some notebook dependencies.

### Using _mHVSR-Vs30_

1.  After installing the legacy notebook dependencies, launch `low_dim_models.ipynb` or `high_dim_models.ipynb` (recommended),
  for a no-coding-required introduction to prediction models. If you have not installed `Jupyter Lab`,
  detailed instructions can be found [here](https://jpvantassel.github.io/python3-course/#/intro/installing_jupyter).

2.  Continue to explore the three datasets provided by changing the file names and associated metadata in the
  notebooks. Once you feel comfortable running the example data you can then apply the models on any mHVSR data
  of interest.

3.  Enjoy!

## Data pipeline

Curating new training corpora is handled by the `mhvsr_vs30` package under
`src/`, which builds validated recording-level manifests from published datasets
and parses their raw files into a canonical three-component representation
without modifying the raw data. See [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md).
