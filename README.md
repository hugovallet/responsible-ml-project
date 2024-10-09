# Responsible AI Framework For Supervised ML

This repo implements a toy ML project running on simulated data
with a full suite of MLOps tools to demonstrate how to build
automatic documentation and proofing to support responsible AI systems.

The system implemented is summarised by the following
architecture:

![](docs/_static/architecture.png)

## Scenario description

### Use-case

A data science team is tasked with creating a regression model on some input
data stored in the feature store showed in the diagram above. Their goal
is to make a model that is performant and for which all the possible
RAI risks are well measured, tracked over time and documented.

### Data used

In this fake project, we use as feature store the infamous Boston Dataset.
This dataset contains 2 potentially discriminatory columns: "B" and "LSTAT".
We have created multiple snapshot of that dataset, at different point in time.
At some point in the timeline, we simulate a data drift for a few randomly
selected features.

We assume that the data engineering and governance team has maintained
up-to-date data catalogues (in `data/`) including risk flags for
problematic data. This will allow automatic detection of problematic
data usage.

### Tools used

1. MLFlow: we use MLFlow to store our trained regression models, ensure
they are reviewed, documented and that there exist a segmentation between
production-grade and development-grade models.
2. Evidently: we use Evidently to measure input data drift when performing
inference using production-level models stored on MLFlow

## Getting started

1. Create Python environment
2. Start MLFlow UI
3. Start Evidently
4. Generate the fake data
5. Train model
6. Check model perf and risk on MLFlow, document it manually
7. Pass model to production (to do manually from MLFLow UI)
8. Run inference multiple times
9. Observe drift appearing at month 5 in Evidently
