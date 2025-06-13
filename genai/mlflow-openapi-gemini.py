#!/usr/bin/env python

import os

import mlflow
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

mlflow.openai.autolog()
mlflow.set_experiment("genai_openai_example")

def ask_openai():
    client = OpenAI()

    response = client.responses.create(
        model="gpt-4.1",
        input="Write a one-sentence bedtime story about a unicorn."
    )

    print(response.output_text)


def ask_gemini_flash():
    client = OpenAI(
        api_key=os.environ["GEMINI_API_KEY"],
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    response = client.chat.completions.create(
        model="gemini-2.5-flash-preview-05-20",
        reasoning_effort="low",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Describe a leech as a haiku"
            }
        ]
    )

    print(response.choices[0].message.content)


ask_gemini_flash()
