import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="novita",
    api_key="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",
)

completion = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": "What is the minimum and maximum entry age allowed under this policy?"
        }
    ],
)

print(completion.choices[0].message)