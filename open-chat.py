#!/usr/bin/env python
from openai import OpenAI
client = OpenAI(
            api_key = "AIzaSyAyTUtpimIp9lthyqVD1aFJphWEwcAzrPU",
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
         )


msg = input("Enter your message: ")

response = client.chat.completions.create(
    model="gemini-2.0-flash",
    messages=[
        {"role": "system", "content": "Rewrite the sentence in better english"},
        {
            "role": "user",
            "content": msg
        }
    ]
)

print(response.choices[0].message.content)

