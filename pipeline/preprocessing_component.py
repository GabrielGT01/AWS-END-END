#| filename: preprocessing_component.py
#| code-line-numbers: true

import os
import pandas as pd
import json
import joblib

from io import StringIO

#the SageMaker Endpoint  built supports processing a batch of instances, and it's only limited by the amount of available memory. 
#Modify the code that runs the endpoint to raise an exception if the request contains more than 100 instances.

try:
    from sagemaker_containers.beta.framework import worker
except ImportError:
    # We don't have access to the `worker` package when testing locally.
    # We'll set it to None so we can change the way functions create
    # a response.
    worker = None


TARGET_COLUMN = "admission"
FEATURE_COLUMNS = [
    "gender",
    "gpa",
    "major",
    "race",
    "gmat",
    "work_exp",
    "work_industry"
]

#load my fit transform model for the features
def model_fn(model_dir):
    """
    Deserializes the model that will be used in this container.
    """

    return joblib.load(os.path.join(model_dir, "features.joblib"))


def input_fn(input_data, content_type):
    """
    Parses the input payload and creates a Pandas DataFrame.

    This function will check whether the target column is present in the
    input data and will remove it.
    """

    if content_type == "text/csv":
        df = pd.read_csv(StringIO(input_data), header=None, skipinitialspace=True)

        # If we find an extra column, it's probably the target
        # feature, so let's drop it. We'll assume the target
        # is always the first column,
        if len(df.columns) == len(FEATURE_COLUMNS) + 1: 
            df = df.drop(df.columns[0], axis=1)

        df.columns = FEATURE_COLUMNS
        return df

    if content_type == "application/json":
        df = pd.DataFrame([json.loads(input_data)])

        if TARGET_COLUMN in df.columns:
            df = df.drop(TARGET_COLUMN, axis=1)

        return df

    raise ValueError(f"{content_type} is not supported!")


def predict_fn(input_data, model):
    """
    Preprocess the input using the transformer.
    """
  #model is the accessed return joblib.load(os.path.join(model_dir, "features.joblib"))
  #input data  is either from csv or json
    try:
        return model.transform(input_data)
    except ValueError as e:
        print("Error transforming the input data", e)
        return None



def output_fn(prediction, accept, limit=100):
    #generally creating an array to arrange into end point
    """
    Formats the prediction output to generate a response.

    The default accept/content-type between containers for serial inference
    is JSON. Since this model precedes a TensorFlow model, we want to
    return a JSON object following TensorFlow's input requirements.
    """

    if prediction is None:
        raise Exception("There was an error transforming the input data")

    # Convert the prediction to a list and create a response dictionary
    instances = [p for p in prediction.tolist()]
    response = {"instances": instances}

    # Check if the number of instances exceeds the limit
    if len(response['instances']) > limit:
        raise ValueError(f"Batch size exceeds the limit of {limit} instances. Got {len(response['instances'])} instances.")
    
    # Return the response as JSON or the specified content type
    return (
        worker.Response(json.dumps(response), mimetype=accept)
        if worker else (response, accept)
    )
