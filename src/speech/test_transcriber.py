from transcriber import transcribe_audio


audio_path = "test_audio.wav"


print("Transcribing audio...")


text = transcribe_audio(
    audio_path,
    language_code="en-IN"
)


print("\nTranscription:")
print(text)