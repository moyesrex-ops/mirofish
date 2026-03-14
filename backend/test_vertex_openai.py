import openai
import subprocess
import os

def get_token():
    result = subprocess.run(['gcloud', 'auth', 'print-access-token'], capture_output=True, text=True, shell=True)
    return result.stdout.strip()

client = openai.OpenAI(
    base_url="https://us-central1-aiplatform.googleapis.com/v1/projects/reference-city-xrjsb/locations/us-central1/endpoints/openapi",
    api_key=get_token()
)

try:
    # Use google/ prefix for Vertex AI endpoint
    response = client.chat.completions.create(
        model="google/gemini-1.5-flash",
        messages=[{"role": "user", "content": "Hello, identify yourself."}]
    )
    print("Success!")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Failed: {e}")
