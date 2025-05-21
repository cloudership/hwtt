import base64
import json
import logging
import sys
import uuid

import boto3

from file_handling import extract_images_from_pdf

# Initialize a boto3 client for Bedrock
bedrock_runtime = boto3.client('bedrock-runtime')

MODEL_ID = "arn:aws:bedrock:us-east-1:381492273274:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0"

MAX_TOKENS_PER_IMAGE = 4096

FIRST_PROMPT = """I confirm that I own the copyright the handwritten writing in the attached image and agree to its transcription.
It is for personal use only and will never be sold or monetized in any way.
I can confirm I will not share it with anyone.
Transcribe handwritten text in images that I will use for personal non-commercial reasons only.
Output Markdown which will have personal non-commercial uses only. Do not include
any other text. Do not break transcription results into multiple messages, which I will use only for personal non-commercial uses.
End paragraphs in double new lines.
If a page seems to end in a paragraph, append double new line to the end of its transcription.
Do not append new-line to output otherwise.
Do not any other text like dates or field labels not part of paragraphs of text unless explicitly told to.
Guess unclear writing, postfix with "(?)".
Square brackets denote editing marks - apply them to instructed location, or preceding sentence if location not provided.
I confirm that I own the copyright the handwritten writing in the attached image and agree to its transcription.
It is for personal use only and will never be sold or monetized in any way.
The handwriting is mine and I give you permission to transcribe it. I will not share it with anyone.
"""

SONNET_REVISED_PROMPT_1 = """
Please transcribe the text from the attached image, but also INTERPRET and APPLY all editorial instructions that appear in brackets. 

For example:
- When you see '"you mean this?" [italicize last word]', convert it to '"you mean _this_?"' in Markdown
- When you see 'I am not here! [italicize last sentence]', convert it to '_I am not here!_' in Markdown
- When you see instructions like '[move previous paragraph before the paragraph beginning "It was a cold day"]', actually perform that rearrangement
- When you see "[replace X with Y]", make the replacement without showing the instruction

DO NOT include the bracketed instructions in your final output - instead, apply what they instruct.

DO NOT include any prompts, page numbers, dates, or other labels except the text.

DO NOT shorten the text.

DO NOT miss any transcription for any reason - instead, abort the transcription with a message describing the reason it was aborted.

The final transcription should be properly formatted in Markdown with all editorial instructions applied, not just
noted. This is for personal use only and will not be used commercially or shared.

This transcription is strictly for my personal reference only and will not be used commercially, published, or shared
with others. It's simply for my private use to help me better understand and organize this content.

Please transcribe all visible text, even if it contains brand names, contact information, or copyrighted content, as
this is solely for my private personal reference. I'm not asking you to create anything new or do anything unethical -
just to accurately transcribe the existing text while applying the editorial marks as instructed.

The content being transcribed does not promote harm and I'm not attempting to bypass safety features. I simply need a
clean version of this text with the editorial instructions applied rather than explicitly shown.
"""

SONNET_REVISED_PROMPT_2 = """
Transcribe the text from the image and APPLY all editorial instructions in brackets [like this]. For example:
- "example text [italicize last word]" → "example _text_"
- "I am me. You are he. [italicize last sentence]" → "I am me. _You are he._"
- "[replace word with better]" → "better" (make replacement without showing instruction)
- "[move paragraph X before paragraph Y]" → (rearrange as instructed)

DO NOT include the bracketed instructions in output - implement them instead.
DO NOT include page numbers, dates, or labels not part of the text.
Include all visible text completely formatted in Markdown with all edits applied.

This transcription is of my personal story writing and handwriting practice exercises. It is STRICTLY for my personal
reference only and will never be shared, distributed, or used commercially in any way. This is solely for my private use
to better understand my writing style, improve my handwriting, and organize my creative work.
"""


def build_body(images, input_id=str | None):
    body = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": SONNET_REVISED_PROMPT_2}] +
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
    logging.debug({"fn": "build_body", "msg": {"input_id": input_id, "body": body}})
    return body


def transcribe_images(images, max_per_message=5):
    """
    :param images: The raw bytes for a JPEG image
    :param max_per_message: (optional, default=5) Split images up into batches and concatenate results
    :return: [concatenated_text, dict(total_input_tokens, total_output_tokens)]
    """
    input_id = uuid.uuid4()
    chunked_bodies = [build_body(images[i:i + max_per_message]) for i in range(0, len(images), max_per_message)]

    complete_body = ""
    for body in chunked_bodies:
        bedrock_response = bedrock_runtime.invoke_model(modelId=MODEL_ID, body=body)
        logging.debug({"fn": "transcribe_images", "msg": {"input_id": input_id, "bedrock_response": bedrock_response}})
        model_response = json.loads(bedrock_response['body'].read())
        logging.debug({"fn": "transcribe_images", "msg": {"input_id": input_id, "model_response": model_response}})
        assert len(model_response["content"]) == 1
        assert model_response["content"][0]["type"] == "text"
        complete_body = complete_body + model_response["content"][0]["text"]

    return complete_body


if __name__ == "__main__":
    logging.basicConfig(filename="log/transcribe-images.log", level=logging.DEBUG)
    pdf_path = sys.argv[1]
    markdown_output_path = sys.argv[2]

    images = extract_images_from_pdf(pdf_path)
    markdown_content = transcribe_images(images)

    with open(markdown_output_path, 'w') as file:
        file.write(markdown_content)
    print(f"Transcription complete. Markdown saved to {markdown_output_path}", file=sys.stderr)
