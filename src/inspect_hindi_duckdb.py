import duckdb

print("Connecting to MSMARCO-XI Hindi data...")

url = "hf://datasets/ai4bharat/MSMARCO-XI/train/hintrain.parquet"

con = duckdb.connect()

print("\nReading 2 rows...")

result = con.execute("""
    SELECT
        query,
        "Eng_Query",
        "Answer",
        "Eng_Answer",
        passages
    FROM read_parquet(?)
    LIMIT 2
""", [url]).fetchall()

print(f"Rows received: {len(result)}")

for i, row in enumerate(result, start=1):

    query, eng_query, answer, eng_answer, passages = row

    print("\n" + "=" * 70)
    print(f"ROW {i}")
    print("=" * 70)

    print("\nHindi Query:")
    print(query)

    print("\nEnglish Query:")
    print(eng_query)

    print("\nHindi Answer:")
    print(answer)

    print("\nEnglish Answer:")
    print(eng_answer)

    print("\nPassages:")
    print(passages)