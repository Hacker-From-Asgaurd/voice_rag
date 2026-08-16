from chunking import adaptive_parent_child_chunks


short_text = (
    "मैनहट्टन परियोजना द्वितीय विश्व युद्ध के दौरान एक महत्वपूर्ण "
    "अनुसंधान परियोजना थी। इसका उद्देश्य परमाणु हथियार विकसित करना था।"
)

long_text = """
मैनहट्टन परियोजना द्वितीय विश्व युद्ध के दौरान एक महत्वपूर्ण अनुसंधान परियोजना थी।
इस परियोजना का उद्देश्य परमाणु हथियार विकसित करना था।
इसमें कई वैज्ञानिक और इंजीनियर शामिल थे।
परियोजना की सफलता ने विश्व इतिहास को बदल दिया।
इसके बाद परमाणु ऊर्जा के क्षेत्र में महत्वपूर्ण विकास हुआ।
""" * 20


print("=" * 70)
print("SHORT PASSAGE")
print("=" * 70)

short_chunks = adaptive_parent_child_chunks(short_text)

for item in short_chunks:
    print("Is child:", item["is_child"])
    print("Chunk length:", len(item["chunk"]))
    print("Chunk:", item["chunk"])


print("\n" + "=" * 70)
print("LONG PASSAGE")
print("=" * 70)

long_chunks = adaptive_parent_child_chunks(long_text)

print("Number of child chunks:", len(long_chunks))

for i, item in enumerate(long_chunks, start=1):
    print(f"\nChild chunk {i}")
    print("Is child:", item["is_child"])
    print("Chunk length:", len(item["chunk"]))
    print(item["chunk"][:200])