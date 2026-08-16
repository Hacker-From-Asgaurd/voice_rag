import io
import os
import time
from pathlib import Path
from typing import Optional, Union, BinaryIO

from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()


class SarvamTranscriber:
    """
    Official Sarvam AI Saaras v3 Speech-to-Text Transcriber.
    Loads API key from environment and supports both file paths and in-memory byte buffers.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SARVAM_API_KEY")
        if not self.api_key:
            raise ValueError(
                "SARVAM_API_KEY was not found in environment or .env file."
            )

        self.client = SarvamAI(
            api_subscription_key=self.api_key
        )
        self.model_name = "saaras:v3"

    def transcribe(
        self,
        audio_source: Union[str, Path, bytes, BinaryIO],
        language_code: str = "unknown",
        filename: str = "audio.wav",
    ) -> dict:
        """
        Transcribe an audio file or bytes using Sarvam Saaras v3.

        Returns:
            dict with:
                - 'transcript': transcribed text
                - 'language_code': detected/provided language
                - 'latency_ms': time taken for transcription in milliseconds
        """
        t0 = time.perf_counter()

        if isinstance(audio_source, (str, Path)):
            audio_path = Path(audio_source)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            with open(audio_path, "rb") as f:
                response = self.client.speech_to_text.transcribe(
                    file=f,
                    model=self.model_name,
                    mode="transcribe",
                    language_code=language_code,
                )
        elif isinstance(audio_source, bytes):
            bio = io.BytesIO(audio_source)
            bio.name = filename
            response = self.client.speech_to_text.transcribe(
                file=bio,
                model=self.model_name,
                mode="transcribe",
                language_code=language_code,
            )
        else:
            # File-like object
            response = self.client.speech_to_text.transcribe(
                file=audio_source,
                model=self.model_name,
                mode="transcribe",
                language_code=language_code,
            )

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        transcript = getattr(response, "transcript", "") or ""
        detected_lang = getattr(response, "language_code", language_code) or language_code

        return {
            "transcript": transcript.strip(),
            "language_code": detected_lang,
            "latency_ms": latency_ms,
        }


# Convenience function for backward compatibility
_default_transcriber: Optional[SarvamTranscriber] = None


def get_default_transcriber() -> SarvamTranscriber:
    global _default_transcriber
    if _default_transcriber is None:
        _default_transcriber = SarvamTranscriber()
    return _default_transcriber


def transcribe_audio(audio_path: str, language_code: str = "unknown") -> str:
    transcriber = get_default_transcriber()
    result = transcriber.transcribe(audio_path, language_code=language_code)
    return result["transcript"]