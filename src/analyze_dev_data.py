import duckdb

DB = duckdb.connect()

FILE = "data/hindi_dev.parquet"

print("Analyzing local Hindi dataset...")
print()

# Basic statistics
stats = DB.execute(f"""
    SELECT
        COUNT(*) AS total_rows,
        COUNT(DISTINCT query_id) AS unique_queries
    FROM '{FILE}'
""").fetchone()

print("Total rows:", stats[0])
print("Unique queries:", stats[1])

# Passage statistics
passage_stats = DB.execute(f"""
    SELECT
        AVG(len(passages.Translated_passages)) AS avg_passages,
        MIN(len(passages.Translated_passages)) AS min_passages,
        MAX(len(passages.Translated_passages)) AS max_passages
    FROM '{FILE}'
""").fetchone()

print()
print("Average passages per query:", round(passage_stats[0], 2))
print("Minimum passages:", passage_stats[1])
print("Maximum passages:", passage_stats[2])

# Selected passage count
selected_stats = DB.execute(f"""
    SELECT
        SUM(
            list_sum(
                list_transform(
                    passages.is_selected,
                    x -> CASE WHEN x = 1 THEN 1 ELSE 0 END
                )
            )
        )
    FROM '{FILE}'
""").fetchone()

print()
print("Total selected/relevant passages:", selected_stats[0])

# Sample row
sample = DB.execute(f"""
    SELECT
        query,
        Answer,
        passages
    FROM '{FILE}'
    LIMIT 1
""").fetchone()

print()
print("=" * 70)
print("SAMPLE QUERY")
print("=" * 70)
print(sample[0])

print()
print("ANSWER")
print("=" * 70)
print(sample[1])

print()
print("FIRST 3 PASSAGES")
print("=" * 70)

passages = sample[2]["Translated_passages"]
labels = sample[2]["is_selected"]

for i in range(min(3, len(passages))):
    print(f"\nPassage {i + 1}")
    print("Selected:", labels[i])
    print(passages[i][:500])