from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

model = SentenceTransformer(MODEL_NAME)

query = "मैनहट्टन परियोजना की सफलता का तुरंत क्या प्रभाव पड़ा?"

relevant = (
    "मैनहट्टन परियोजना की सफलता ने दुनिया को बदल दिया और "
    "परमाणु हथियारों के प्रभाव को स्पष्ट किया।"
)

unrelated = (
    "भारत की राजधानी नई दिल्ली है और यह देश का प्रमुख "
    "राजनीतिक तथा प्रशासनिक केंद्र है।"
)

texts = [query, relevant, unrelated]

embeddings = model.encode(
    texts,
    normalize_embeddings=True
)

query_vector = embeddings[0]

relevant_score = np.dot(query_vector, embeddings[1])
unrelated_score = np.dot(query_vector, embeddings[2])

print("Query:")
print(query)

print("\nSimilarity with relevant passage:")
print(round(float(relevant_score), 4))

print("\nSimilarity with unrelated passage:")
print(round(float(unrelated_score), 4))