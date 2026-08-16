import os
import asyncio
import base64

from dotenv import load_dotenv
from sarvamai import AsyncSarvamAI


load_dotenv()

api_key = os.getenv("SARVAM_API_KEY")

if not api_key:
    raise ValueError("SARVAM_API_KEY was not found in .env")


async def transcribe_streaming_async(
    audio_path,
    language_code="hi-IN"
):
    client = AsyncSarvamAI(
        api_subscription_key=api_key
    )

    with open(audio_path, "rb") as f:
        audio_data = base64.b64encode(
            f.read()
        ).decode("utf-8")

    async with client.speech_to_text_streaming.connect(
        model="saaras:v3",
        mode="transcribe",
        language_code=language_code,
        high_vad_sensitivity=True,
        flush_signal=True
    ) as ws:

        print("Sending audio...")

        await ws.transcribe(
            audio=audio_data,
            encoding="audio/wav",
            sample_rate=16000
        )

        print("Audio sent.")

        await ws.flush()

        print("Waiting for transcription...")

        async for message in ws:

            print("Received:", message)

            if message.type == "data":
                return message.data.transcript

    return ""


def transcribe_streaming(
    audio_path,
    language_code="hi-IN"
):
    return asyncio.run(
        transcribe_streaming_async(
            audio_path,
            language_code
        )
    )