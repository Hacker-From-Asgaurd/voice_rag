import duckdb

OUTPUT_FILE = "data/hindi_dev.parquet"

url = "hf://datasets/ai4bharat/MSMARCO-XI/train/hintrain.parquet"

print("Downloading a 5,000-row development subset...")

con = duckdb.connect()

con.execute(f"""
    COPY (
        SELECT
            query_id,
            query_type,
            query,
            Eng_Query,
            Answer,
            Eng_Answer,
            passages
        FROM read_parquet('{url}')
        LIMIT 5000
    )
    TO '{OUTPUT_FILE}'
    (FORMAT PARQUET);
""")

print()
print("Done!")
print(f"Saved to: {OUTPUT_FILE}")