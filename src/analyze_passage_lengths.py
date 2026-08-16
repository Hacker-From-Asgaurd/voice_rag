import duckdb

FILE = "data/hindi_dev.parquet"

con = duckdb.connect()

print("Analyzing passage lengths...")

result = con.execute(f"""
    SELECT
        AVG(length(passage)) AS avg_chars,
        MIN(length(passage)) AS min_chars,
        MAX(length(passage)) AS max_chars
    FROM (
        SELECT
            unnest(passages.Translated_passages) AS passage
        FROM '{FILE}'
    )
    WHERE length(passage) > 0
""").fetchone()

print()
print("Average passage length:", round(result[0], 2), "characters")
print("Minimum passage length:", result[1], "characters")
print("Maximum passage length:", result[2], "characters")