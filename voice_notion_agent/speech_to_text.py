import os
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

# Download the sample audio and save it as a file
url = "https://cdn.openai.com/API/docs/audio/alloy.wav"
response = requests.get(url, timeout=30)
response.raise_for_status()
with open("sample.wav", "wb") as f:
    f.write(response.content)

# Transcribe with Groq's Whisper
with open("sample.wav", "rb") as f:
    result = client.audio.transcriptions.create(
        file=("sample.wav", f.read()),
        model="whisper-large-v3-turbo",
    )

print(result.text)