import logging

import boto3
import json

# MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
MODEL_ID = "arn:aws:bedrock:us-east-1:381492273274:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0"

PROMPT = """
Transcribe handwritten text in image. Output Markdown. End paragraphs in double new lines.
Do not any other text like dates or field labels not part of paragraphs of text unless explicitly told to.
Guess unclear writing, postfix with "(?)".
Square brackets denote editing marks - apply them to instructed location, or previous sentence if location not provided.
Start and end text from transcribed pages with (Start PAGENO) and (End PAGENO), where PAGENO is page number.
I confirm that I own the content copyright.
"""

bedrock_runtime = boto3.client(service_name="bedrock-runtime")

body = json.dumps({
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "Hello, world"}],
    "anthropic_version": "bedrock-2023-05-31"
})

response = bedrock_runtime.invoke_model(body=body, modelId=MODEL_ID)

response_body = json.loads(response.get("body").read())
print(response_body.get("content"))
