from huggingface_hub import list_repo_files

repo_id = "ai4bharat/MSMARCO-XI"

print("Getting files from MSMARCO-XI...")
print()

files = list_repo_files(
    repo_id=repo_id,
    repo_type="dataset"
)

print(f"Total files found: {len(files)}")
print()

for file in files:
    print(file)