import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import time
import statistics
import asyncio

from speech.streaming_transcriber import (
    transcribe_streaming_async
)

import asyncio


AUDIO_FILE = "test_audio_converted.wav"
LANGUAGE_CODE = "hi-IN"

RUNS = 10


async def benchmark_once():

    start_time = time.perf_counter()

    transcript = await transcribe_streaming_async(
        AUDIO_FILE,
        language_code=LANGUAGE_CODE
    )

    end_time = time.perf_counter()

    total_latency_ms = (
        end_time - start_time
    ) * 1000

    return transcript, total_latency_ms


def percentile(values, percentile):

    values = sorted(values)

    index = (
        (len(values) - 1)
        * percentile
        / 100
    )

    lower = int(index)
    upper = min(lower + 1, len(values) - 1)

    if lower == upper:
        return values[lower]

    weight = index - lower

    return (
        values[lower]
        + weight
        * (values[upper] - values[lower])
    )


async def main():

    print("=" * 70)
    print("STREAMING STT LATENCY BENCHMARK")
    print("=" * 70)

    print()
    print("Audio file :", AUDIO_FILE)
    print("Language   :", LANGUAGE_CODE)
    print("Runs       :", RUNS)

    if not Path(AUDIO_FILE).exists():

        raise FileNotFoundError(
            f"Audio file not found: {AUDIO_FILE}"
        )

    print()
    print("Starting benchmark...")

    latencies = []
    transcripts = []

    for i in range(1, RUNS + 1):

        print(
            f"\nRun {i:02d}/{RUNS}"
        )

        transcript, latency = (
            await benchmark_once()
        )

        latencies.append(latency)
        transcripts.append(transcript)

        print(
            f"Latency   : {latency:.2f} ms"
        )

        print(
            f"Transcript: {transcript}"
        )

    print()
    print("=" * 70)
    print("STREAMING STT BENCHMARK RESULTS")
    print("=" * 70)

    print(
        f"Average : "
        f"{statistics.mean(latencies):.2f} ms"
    )

    print(
        f"P50     : "
        f"{percentile(latencies, 50):.2f} ms"
    )

    print(
        f"P70     : "
        f"{percentile(latencies, 70):.2f} ms"
    )

    print(
        f"P100    : "
        f"{max(latencies):.2f} ms"
    )

    print()
    print("=" * 70)
    print("TRANSCRIPT CONSISTENCY")
    print("=" * 70)

    unique_transcripts = set(
        transcripts
    )

    print(
        "Unique transcripts:",
        len(unique_transcripts)
    )

    if len(unique_transcripts) == 1:

        print(
            "Consistency: PASSED"
        )

    else:

        print(
            "Consistency: CHECK REQUIRED"
        )

        for transcript in unique_transcripts:
            print("-", transcript)

    print()
    print("=" * 70)


if __name__ == "__main__":

    asyncio.run(main())