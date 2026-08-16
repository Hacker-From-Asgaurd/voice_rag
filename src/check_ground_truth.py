import json

METADATA_FILE = "data/adaptive_metadata.json"

QUERY_ID = 1185869

print("Loading metadata...")

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

matches = [
    record
    for record in metadata
    if record["query_id"] == QUERY_ID
]

print("Matching chunks:", len(matches))

print("\n" + "=" * 70)
print("GROUND TRUTH FOR QUERY", QUERY_ID)
print("=" * 70)

for record in matches:

    print("\nPassage ID:", record["passage_id"])
    print("Selected:", record["is_selected"])
    print("Chunk length:", len(record["chunk"]))

    if record["is_selected"] == 1:
        print("\n*** RELEVANT PASSAGE ***")

    print(record["chunk"][:1000])