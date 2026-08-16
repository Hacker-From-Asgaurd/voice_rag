import pyarrow.parquet as pq
import fsspec

print("Connecting to Hindi MSMARCO-XI...")

url = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/train/hintrain.parquet"

fs = fsspec.filesystem("https")

with fs.open(url, "rb") as f:
    parquet_file = pq.ParquetFile(f)

    print("\nConnected successfully.")
    print("Total rows:", parquet_file.metadata.num_rows)

    print("\nReading first 2 rows...")

    table = parquet_file.read_row_group(
        0,
        columns=[
            "query",
            "Eng_Query",
            "Answer",
            "Eng_Answer",
            "passages",
        ]
    ).slice(0, 2)

    rows = table.to_pylist()

    for i, row in enumerate(rows, start=1):

        print("\n" + "=" * 70)
        print(f"ROW {i}")
        print("=" * 70)

        print("\nHindi Query:")
        print(row["query"])

        print("\nEnglish Query:")
        print(row["Eng_Query"])

        print("\nHindi Answer:")
        print(row["Answer"])

        print("\nEnglish Answer:")
        print(row["Eng_Answer"])

        passages = row["passages"]

        print("\nNumber of passages:")
        print(len(passages["Translated_passages"]))

        print("\nPassages:")
        for j, passage in enumerate(
            passages["Translated_passages"]
        ):
            print(f"\n--- Passage {j + 1} ---")
            print(passage[:300])

        print("\nSelection labels:")
        print(passages["is_selected"])