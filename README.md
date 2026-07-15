# MBA Admission Prediction — End-to-End AWS SageMaker MLOps Pipeline

An end-to-end machine learning system built on **Amazon SageMaker Pipelines** that predicts MBA admission outcomes (Admit / Deny / Waitlist) using a synthetic dataset modeled on the Wharton School's Class of 2025 statistics. The project covers the full ML lifecycle — data exploration, preprocessing, training, hyperparameter tuning, evaluation, conditional model registration, and automated deployment with monitoring — all orchestrated as SageMaker Pipelines.

## Dataset

- **Source**: [MBA Admission Dataset (Kaggle)](https://www.kaggle.com/datasets/taweilo/mba-admission-dataset/data)
- **Target**: `admission` — three classes: `Admit`, `Deny`, `Waitlist` (highly imbalanced, with `Deny` as the majority class)
- **Features**: `gender`, `gpa`, `major`, `race`, `gmat`, `work_exp`, `work_industry`, `international`, `application_id`

## What This Project Demonstrates

This notebook is structured as **12 progressive sessions**, each building on the last, showing how a production ML system evolves from a raw dataset to a fully monitored, self-deploying pipeline on AWS.

| Session | Focus |
|---|---|
| 1 | Environment setup — IAM role, S3 bucket, local-vs-cloud pipeline config, ARM64/Docker handling |
| 2 | Exploratory Data Analysis — missing values, distributions, covariance/correlation matrices, outlier checks, Chi-Square tests, ANOVA/t-tests for feature-target association |
| 3 | Data splitting & transformation via a SageMaker Processing Step + Scikit-Learn pipeline, with step caching |
| 4 | Model training with a TensorFlow neural network, including a class-weighting strategy to handle severe class imbalance, later extended to 5-fold cross-validation |
| 5 | Hyperparameter tuning via a SageMaker Tuning Step (batch size, learning rate) |
| 6–7 | Model evaluation — comparing the cross-validated training model against the tuned model on a held-out test set |
| 8 | Model registration in the SageMaker Model Registry, with evaluation metrics attached as model metadata |
| 9 | Conditional registration — a Condition Step gates registration on accuracy/precision/recall thresholds, with a Fail Step and SQS/SES-based failure notifications |
| 10 | Automated deployment via a Lambda Step triggered directly from the pipeline (endpoint creation + data capture config) |
| 11 | Event-driven deployment — an EventBridge rule triggers deployment via Lambda whenever a model's approval status changes to `Approved` (human-in-the-loop workflow), plus blue-green / canary traffic-shifting deployment and CloudWatch alarms on both model variants |
| 12 | A full **inference pipeline** — pre/post-processing components wrapped around the model so the endpoint accepts raw input and returns human-readable predictions, plus asynchronous inference and endpoint latency monitoring |

## Architecture

```
Raw Data (S3)
     │
     ▼
┌─────────────────────┐
│  Processing Step     │  Scikit-Learn pipeline: clean, split, encode
│  (split & transform) │  → train / validation / test sets + transformer artifact
└─────────┬────────────┘
          ▼
┌─────────────────────┐        ┌──────────────────────┐
│  Training Step       │  or   │  Tuning Step          │
│  (TensorFlow, CV)     │       │  (HPO: lr, batch size)│
└─────────┬────────────┘        └──────────┬───────────┘
          └───────────────┬─────────────────┘
                           ▼
                ┌─────────────────────┐
                │  Evaluation Step     │  TensorFlowProcessor scores model
                │  (Processing)        │  on test set → evaluation report (JSON)
                └─────────┬────────────┘
                           ▼
                ┌─────────────────────┐
                │  Condition Step      │  accuracy / precision / recall
                │                      │  ≥ thresholds?
                └─────┬───────────┬────┘
                      │ pass      │ fail
                      ▼           ▼
           ┌──────────────────┐ ┌────────────┐
           │  Model Step       │ │  Fail Step │ → SQS/SES notification
           │  (register model) │ └────────────┘
           └────────┬──────────┘
                     ▼
           ┌──────────────────────┐
           │  Lambda Step /        │  Creates/updates SageMaker endpoint
           │  EventBridge trigger  │  (data capture, blue-green, canary)
           └────────┬──────────────┘
                     ▼
           ┌──────────────────────┐
           │  Live Endpoint         │  Inference pipeline (pre → model → post)
           │  + CloudWatch alarms   │  Sync + async inference supported
           └──────────────────────┘
```

## Tech Stack

- **ML / Data**: TensorFlow, scikit-learn, pandas, NumPy, statsmodels, SciPy
- **Orchestration**: Amazon SageMaker Pipelines (Processing, Training, Tuning, Condition, Lambda, Callback, Model, Fail steps)
- **AWS Services**: SageMaker (Training Jobs, Hyperparameter Tuning, Model Registry, Endpoints, Data Capture, Async Inference), S3, Lambda, EventBridge, SQS, SES, CloudWatch, IAM
- **Experiment Tracking**: Comet ML
- **Visualization**: seaborn, matplotlib

## Key ML Design Decisions

- **Class imbalance**: The target classes are heavily skewed (e.g., ~900 Admit / ~5,194 Deny / ~100 Waitlist), so the training script applies class weighting to keep the neural network from collapsing to the majority class.
- **Cross-validation**: A 5-fold CV variant of the training script (`script_cv.py`) is used so the model is evaluated more robustly than a single train/validation split, and the learning rate/epoch count are exposed as pipeline parameters rather than hard-coded.
- **Two competing models**: A cross-validated training-step model and a tuning-step (HPO) model are evaluated side by side; the best performer (the cross-validated model, ~75% accuracy / ~85% precision) is the one carried forward into registration and deployment.
- **Governance**: Models only get registered if they clear accuracy/precision/recall thresholds (`ConditionStep` + `FailStep`), and deployment only proceeds after manual or event-driven approval in the Model Registry — a human-in-the-loop safety gate before anything reaches production.
- **Safe rollout**: Deployment uses blue-green / canary traffic shifting with CloudWatch alarms on both the existing and new model variants, so a bad new model can be caught before it takes over all traffic.

## Repository Structure (generated by the notebook)

```
code/
├── processing/
│   └── script.py                  # split & transform logic
├── training/
│   ├── script.py                  # baseline TensorFlow training script
│   ├── script_cv.py                # 5-fold cross-validation training script
│   └── requirements.txt            # Comet ML dependency for the training container
├── evaluation/
│   ├── script.py                   # evaluates both train & tuned models
│   ├── script_onlytrain_model.py    # evaluates only the selected (best) model
│   └── requirements.txt
├── lambda/
│   └── wharton_lambda.py            # deploys/updates the SageMaker endpoint
└── pipeline/
    ├── preprocessing_component.py   # inference-time input transformer
    └── postprocessing_component.py  # inference-time output transformer
```

## Prerequisites

- An AWS account with a SageMaker execution role (with permissions for SageMaker, S3, Lambda, EventBridge, SQS, SES, CloudWatch)
- An S3 bucket to store data, code, and model artifacts
- A [Comet ML](https://www.comet.com/) API key and project name for experiment tracking
- Python environment with:
  ```
  sagemaker
  boto3
  tensorflow
  scikit-learn
  pandas
  numpy
  seaborn
  matplotlib
  statsmodels
  scipy
  ```

## Setup

1. Set the following environment variables before running the notebook:
   ```bash
   export role="arn:aws:iam::<account-id>:role/service-role/<your-sagemaker-execution-role>"
   export bucket="<your-s3-bucket-name>"
   export COMET_API_KEY="<your-comet-api-key>"
   export COMET_PROJECT_NAME="<your-comet-project-name>"
   ```
2. Set `LOCAL_MODE = True` in the notebook to run pipeline steps locally (useful for quick iteration/debugging); set it to `False` to run on managed SageMaker infrastructure.
3. If running locally on an ARM64 machine (e.g., Apple Silicon), a custom Docker image is required for the Processing/Training steps — this is configured automatically based on detected architecture.
4. Run the notebook session by session — each session builds and (optionally) executes its own named SageMaker Pipeline (`session1-...` through `session10-...`), so you can inspect and validate each stage before moving to the next.

## Notes

- ARNs, endpoint names, email addresses, and bucket names in the notebook are illustrative and tied to the original author's AWS account — replace them with your own before running.
- The evaluation and deployment steps assume the model has already cleared the accuracy/precision/recall thresholds defined as pipeline parameters (default accuracy threshold: 0.70).
