#!/usr/bin/env python

import litellm
import mlflow
from dotenv import load_dotenv
from mlflow.genai import register_prompt

load_dotenv()

mlflow.litellm.autolog()
mlflow.set_experiment("genai-litellm-example")

# register a prompt so we can link it when logging the model
system_prompt = register_prompt(
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
            "content": system_prompt.format(question="Describe a fart with emoji only"),
        },
    ],
    reasoning_effort="low",
)

print(response.choices[0].message.content)
