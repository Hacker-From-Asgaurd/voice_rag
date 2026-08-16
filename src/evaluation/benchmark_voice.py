import sys
import time
import statistics
from pathlib import Path

# Add src/ to Python path
SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from speech.transcriber import transcribe_audio


AUDIO_FILE = "test_audio.wav"

RUNS = 10


def percentile(values, p):
    values = sorted(values)

    if not values:
        return 0.0

    index = (len(values) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)

    weight = index - lower

    return (
        values[lower]
        + (values[upper] - values[lower]) * weight
    )


def main():

    print("=" * 70)
    print("VOICE / STT LATENCY BENCHMARK")
    print("=" * 70)

    print(f"\nAudio file: {AUDIO_FILE}")
    print(f"Runs: {RUNS}")

    # --------------------------------------------------
    # Check audio file
    # --------------------------------------------------

    if not Path(AUDIO_FILE).exists():

        raise FileNotFoundError(
            f"Audio file not found: {AUDIO_FILE}"
        )

    # --------------------------------------------------
    # Warm-up
    # --------------------------------------------------

    print("\nWarm-up transcription...")

    transcribe_audio(AUDIO_FILE)

    print("Warm-up complete.")

    # --------------------------------------------------
    # Benchmark
    # --------------------------------------------------

    latencies = []

    print("\nStarting benchmark...")

    for i in range(1, RUNS + 1):

        start = time.perf_counter()

        transcript = transcribe_audio(AUDIO_FILE)

        end = time.perf_counter()

        latency_ms = (end - start) * 1000

        latencies.append(latency_ms)

        print(
            f"Run {i:02d}: "
            f"{latency_ms:.2f} ms"
        )

        if transcript:
            print(
                f"         Transcript: {transcript[:100]}"
            )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("STT BENCHMARK RESULTS")
    print("=" * 70)

    print(
        f"Average : "
        f"{statistics.mean(latencies):.2f} ms"
    )

    print(
        f"P50     : "
        f"{percentile(latencies, 0.50):.2f} ms"
    )

    print(
        f"P70     : "
        f"{percentile(latencies, 0.70):.2f} ms"
    )

    print(
        f"P100    : "
        f"{max(latencies):.2f} ms"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()