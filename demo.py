import requests
from openai import OpenAI

resp = requests.post(
    url="https://api.chatanywhere.tech/v1/chat/completions",
    headers={
        "Authorization": "Bearer sk-XTFNbTWwvJrOeAUekxS2BpP3MtGVmPEO026E0GxMPV8qPzxf",
        "Content-Type": "application/json",
    },
    data="""
    {
        "model": "gpt-4o-mini",
        "messages": [
        {
            "role": "system",
            "content": "You"
        },
        {
            "role": "user",
            "content": "介绍一下你自己"
        }]
    }""",
)
print(resp.text)
