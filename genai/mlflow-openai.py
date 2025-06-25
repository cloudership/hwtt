#!/usr/bin/env python

from dotenv import load_dotenv

load_dotenv()

import mlflow
from openai import OpenAI

mlflow.openai.autolog()
mlflow.set_experiment("genai-openai-example")


def ask_openai():
    client = OpenAI()

    response = client.responses.create(
        model="gpt-4.1-mini",
        input="Write a one-sentence bedtime story about a unicorn."
    )

    print(response.output_text)


ask_openai()
