import openai
import subprocess
import os

def get_token():
    result = subprocess.run(['gcloud', 'auth', 'print-access-token'], capture_output=True, text=True, shell=True)
    return result.stdout.strip()

client = openai.OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=get_token()
)

try:
    response = client.chat.completions.create(
        model="gemini-1.5-flash",
        messages=[{"role": "user", "content": "Hello, identify yourself."}]
    )
    print("Success!")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Failed: {e}")
