import os
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key="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",
)

completion = client.chat.completions.create(
    model="openai/gpt-oss-20b:novita",
    messages=[
        {
            "role": "user",
            "content": "how many r in the word 'refrigerator'? give me anser not think  please and not too long, just the number",
        }
    ],
)

print(completion.choices[0].message)