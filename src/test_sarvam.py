import os
from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()

api_key = os.getenv("SARVAM_API_KEY")

if not api_key:
    raise ValueError("SARVAM_API_KEY was not found in .env")

client = SarvamAI(
    api_subscription_key=api_key
)

audio_path = "test_audio.wav"

with open(audio_path, "rb") as audio_file:
    response = client.speech_to_text.transcribe(
        file=audio_file,
        model="saaras:v3",
        mode="transcribe",
        language_code="en-IN"
    )

print("Transcription:")
print(response.transcript)