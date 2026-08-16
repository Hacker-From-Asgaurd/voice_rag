from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Model loaded.")

texts = [
    "मैनहट्टन परियोजना क्या थी?",
    "मैनहट्टन परियोजना द्वितीय विश्व युद्ध के दौरान परमाणु हथियार बनाने की परियोजना थी।",
    "भारत की राजधानी नई दिल्ली है।",
]

print("\nGenerating embeddings...")

embeddings = model.encode(
    texts,
    normalize_embeddings=True
)

print("Embedding shape:")
print(embeddings.shape)

print("\nFirst vector:")
print(embeddings[0][:10])