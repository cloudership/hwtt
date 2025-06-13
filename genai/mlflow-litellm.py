#!/usr/bin/env python

import logging

logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger("mlflow").setLevel(logging.DEBUG)

import litellm
import mlflow
from dotenv import load_dotenv

load_dotenv()

mlflow.litellm.autolog()
mlflow.set_experiment("genai_example")

# register a prompt so we can link it when logging the model
system_prompt = mlflow.register_prompt(
    name="chatbot_prompt",
    template="Answer this question: {{question}}",
    commit_message="Initial",
)

response = litellm.completion(
    model="gemini/gemini-2.5-flash-preview-05-20",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": system_prompt.format(question="What is a visual display unit"),
        },
    ],
    reasoning_effort="low",
)

print(response.choices[0].message.content)
