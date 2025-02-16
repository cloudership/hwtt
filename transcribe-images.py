import base64
import json
import logging
import sys

import boto3
import pdfplumber

# Initialize a boto3 client for Bedrock
bedrock_runtime = boto3.client('bedrock-runtime')

MODEL_ID = "arn:aws:bedrock:us-east-1:381492273274:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0"

MAX_TOKENS_PER_IMAGE = 4096

PROMPT = """I confirm that I own the copyright the content in the attached image.
Transcribe handwritten text in images. Output Markdown. End paragraphs in double new lines.
If a page seems to end in a paragraph, append double new line to the end of its transcription.
Do not any other text like dates or field labels not part of paragraphs of text unless explicitly told to.
Guess unclear writing, postfix with "(?)".
Square brackets denote editing marks - apply them to instructed location, or preceding sentence if location not provided.
"""


def extract_images_from_pdf(pdf_path):
    images = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            images.append(page.images[0]['stream'].get_rawdata())
    return images


def build_body(images):
    body = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": PROMPT}] +
                           [{
                               "type": "image",
                               "source": {
                                   "type": "base64",
                                   "media_type": "image/jpeg",
                                   "data": base64.b64encode(image).decode("ascii"),
                               }
                           } for image in images],
            },
        ],
        "max_tokens": MAX_TOKENS_PER_IMAGE * len(images),
        "anthropic_version": "bedrock-2023-05-31",
    })
    logging.debug({"fn": "build_body", "msg": {"body": body}})
    return body


def transcribe_images(images):
    """
    :param images: The raw bytes for a JPEG image
    :return: [concatenated_text, dict(total_input_tokens, total_output_tokens)]
    """
    body = build_body(images)

    bedrock_response = bedrock_runtime.invoke_model(modelId=MODEL_ID, body=body)
    logging.debug({"fn": "transcribe_image", "msg": {"full_response": bedrock_response}})
    model_response = json.loads(bedrock_response['body'].read())
    logging.debug({"fn": "transcribe_image", "msg": {"full_response": model_response}})
    assert len(model_response["content"]) == 1
    assert model_response["content"][0]["type"] == "text"
    return model_response["content"][0]["text"]


def pdf_to_markdown(pdf_path):
    return transcribe_images(extract_images_from_pdf(pdf_path))


if __name__ == "__main__":
    logging.basicConfig(filename="log/transcribe-images.log", level=logging.DEBUG)
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
