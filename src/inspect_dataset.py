from datasets import load_dataset

print("Loading a small streamed sample of MSMARCO-XI...")

dataset = load_dataset(
    "ai4bharat/MSMARCO-XI",
    split="train",
    streaming=True
)

print("Dataset loaded.")

first_example = next(iter(dataset))

print("\nAvailable fields:")
print(first_example.keys())

print("\nQuery:")
print(first_example["query"])

print("\nEnglish Query:")
print(first_example["Eng_Query"])

print("\nAnswer:")
print(first_example["Answer"])

print("\nNumber of passages:")
print(len(first_example["passages"]["Translated_passages"]))

print("\nFirst translated passage:")
print(first_example["passages"]["Translated_passages"][0][:500])