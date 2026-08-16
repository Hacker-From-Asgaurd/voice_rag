from generator import generate_answer


question = "मैनहट्टन परियोजना के प्रमुख वैज्ञानिक का जन्म किस वर्ष हुआ था?"


context = """
[Source 1]
मैनहट्टन परियोजना द्वितीय विश्व युद्ध के दौरान एक अनुसंधान और विकास उपक्रम था जिसने पहले परमाणु हथियारों का निर्माण किया था।

[Source 2]
परियोजना 1942 से 1946 तक चली और इसका नेतृत्व संयुक्त राज्य अमेरिका ने किया।

[Source 3]
मैनहट्टन परियोजना का उद्देश्य परमाणु हथियार विकसित करना था।
"""


print("Generating answer...\n")


answer = generate_answer(
    question,
    context
)


print("=" * 70)
print("HALLUCINATION TEST")
print("=" * 70)

print(answer)