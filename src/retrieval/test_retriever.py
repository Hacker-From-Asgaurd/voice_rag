from retriever import Retriever


retriever = Retriever()


query = "मैनहट्टन परियोजना की सफलता का तुरंत क्या प्रभाव पड़ा?"


context = retriever.get_context(
    query,
    top_k=5
)


print("\n" + "=" * 70)
print("CONTEXT FOR LLM")
print("=" * 70)

print(context)