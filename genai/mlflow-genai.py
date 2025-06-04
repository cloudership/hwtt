import litellm
import mlflow
from dotenv import load_dotenv

load_dotenv()

mlflow.litellm.autolog()
mlflow.set_experiment("genai_example")

# client = OpenAI(
#     api_key=os.environ["GEMINI_API_KEY"],
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
# )

# register a prompt so we can link it when logging the model
system_prompt = mlflow.register_prompt(
    name="chatbot_prompt",
    template="Answer this question: {{question}}",
    commit_message="Initial",
)

# response = client.chat.completions.create(
#     model="gemini-2.5-flash-preview-05-20",
#     reasoning_effort="low",
#     messages=[
#         {"role": "system", "content": "You are a chatbot that can answer questions about IT."},
#         {
#             "role": "user",
#             "content": system_prompt.format(question="What is a visual display unit"),
#         }
#     ]
# )

response = litellm.completion(
    model="gemini/gemini-2.5-flash-preview-05-20",
    messages=[
        {"role": "user",
         "content": system_prompt.format(question="What is a visual display unit")
         }
    ],
    reasoning_effort="low",
)

print(response.choices[0].message.content)
