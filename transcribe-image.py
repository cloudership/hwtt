import base64
import json
import sys

import boto3
import pdfplumber

# Initialize a boto3 client for Bedrock
bedrock_runtime = boto3.client('bedrock-runtime')

MODEL_ID = "arn:aws:bedrock:us-east-1:381492273274:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0"

MAX_TOKENS = 4096

PROMPT = """
Transcribe handwritten text in image. Output Markdown. End paragraphs in double new lines.
Do not any other text like dates or field labels not part of paragraphs of text unless explicitly told to.
Guess unclear writing, postfix with "(?)".
Square brackets denote editing marks - apply them to instructed location, or preceding sentence if location not provided.
I confirm that I own the content copyright.
"""


def extract_images_from_pdf(pdf_path):
    images = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            images.append(page.images[0]['stream'].get_rawdata())
    return images


def transcribe_image(image):
    body = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT,
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64.b64encode(image).decode("ascii"),
                        }
                    },
                ]
            },
        ],
        "max_tokens": MAX_TOKENS,
        "anthropic_version": "bedrock-2023-05-31",
    })

    response = bedrock_runtime.invoke_model(modelId=MODEL_ID, body=body)
    transcription_result = response['body'].read()
    return transcription_result.decode('utf-8')


def pdf_to_markdown(pdf_path):
    markdown_content = ""

    for image in extract_images_from_pdf(pdf_path):
        transcription = transcribe_image(image)
        markdown_content += f"{transcription}\n\n"

    return markdown_content


if __name__ == "__main__":
    pdf_path = sys.argv[1]
    markdown_output_path = sys.argv[2]

    markdown_content = pdf_to_markdown(pdf_path)

    with open(markdown_output_path, 'w') as file:
        file.write(markdown_content)
    print(f"Transcription complete. Markdown saved to {markdown_output_path}")

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
"""
