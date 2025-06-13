#!/usr/bin/env python

import json
import logging

logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger("mlflow").setLevel(logging.DEBUG)

import boto3
import mlflow
from dotenv import load_dotenv

load_dotenv()

mlflow.bedrock.autolog()
mlflow.set_experiment("genai-bedrock-sonnet")

# register a prompt so we can link it when logging the model
system_prompt = mlflow.register_prompt(
    name="chatbot_prompt",
    template="{{prompt}}",
    commit_message="Initial",
)

bedrock_runtime = boto3.client('bedrock-runtime')

MODEL_ID = "arn:aws:bedrock:us-east-1:381492273274:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0"

body = json.dumps({
    "messages": [
        {
            "role": "user",
            "content": system_prompt.format(prompt="Describe a lightbulb as a haiku"),
        },
    ],
    "max_tokens": 1000,
    "anthropic_version": "bedrock-2023-05-31",
})
bedrock_response = bedrock_runtime.invoke_model(modelId=MODEL_ID, body=body)
logging.debug(bedrock_response)
model_response = json.loads(bedrock_response['body'].read())
print(model_response['content'][0]['text'])
