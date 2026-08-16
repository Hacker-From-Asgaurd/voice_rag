from streaming_transcriber import transcribe_streaming


AUDIO_FILE = "test_audio_converted.wav"


def main():

    print("=" * 60)
    print("STREAMING STT TEST")
    print("=" * 60)

    print("\nAudio file:", AUDIO_FILE)
    print("Language: hi-IN")

    print("\nStreaming transcription...")

    try:

        text = transcribe_streaming(
            AUDIO_FILE,
            language_code="hi-IN"
        )

        print("\n" + "=" * 60)
        print("TRANSCRIPTION")
        print("=" * 60)

        print(text)

    except Exception as e:

        print("\n" + "=" * 60)
        print("STREAMING STT ERROR")
        print("=" * 60)

        print(type(e).__name__)
        print(e)


if __name__ == "__main__":
    main()