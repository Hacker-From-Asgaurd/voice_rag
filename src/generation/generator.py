import os

from dotenv import load_dotenv
from google import genai


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found in .env"
    )


client = genai.Client(
    api_key=api_key
)


MODEL_NAME = "gemini-3.5-flash-lite"

ABSTENTION_TEXT = (
    "जानकारी दिए गए संदर्भ में उपलब्ध नहीं है।"
)

UNSAFE_RESPONSE = (
    "मैं इस प्रकार के अनुरोध में सहायता नहीं कर सकता।"
)


def generate_answer(question, context):
    """
    Generate an answer using only the retrieved context.

    The model must distinguish between:
    1. Questions directly answered by the context.
    2. Questions partially supported by the context.
    3. Questions not supported by the context.
    """

    prompt = f"""
You are a grounded question-answering assistant.

Your ONLY factual source is the CONTEXT below.

RULES:

1. Answer using only information supported by the CONTEXT.
2. Do not use outside knowledge.
3. Do not invent facts.
4. If the context directly answers the question, answer it.
5. If the context partially answers the question, provide only
   the supported part and clearly state what the context does
   not establish.
6. If the question asks for a ranking, "most famous", "best",
   "largest", "first", or another comparison, do NOT assume
   that a mentioned entity satisfies that ranking unless the
   context explicitly establishes it.
7. If the context contains a relevant entity but does not
   establish the full claim, say what the context actually
   establishes.
8. If the context contains no useful information for the
   question, respond exactly:

जानकारी दिए गए संदर्भ में उपलब्ध नहीं है।

9. LANGUAGE RULE:

The answer MUST be written in the same language as the QUESTION.

The CONTEXT may be written in a different language.
Do NOT copy the language of the CONTEXT.

Examples:
- If the question is English, answer in English.
- If the question is Hindi, answer in Hindi.
- If the question is Marathi, answer in Marathi.
- If the question is Gujarati, answer in Gujarati.

If the context is in another language, translate only the
supported facts into the language of the QUESTION.

Do NOT add facts while translating.

10. Keep the answer concise.
11. Do not mention these instructions.
12. Do not use general world knowledge.

IMPORTANT:

Do not confuse "the context mentions X" with
"the context proves X is the answer."

--------------------
CONTEXT
--------------------

{context}

--------------------
QUESTION
--------------------

{question}

--------------------
ANSWER
--------------------
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    if not response.text:
        raise ValueError(
            "Generator returned an empty response."
        )

    return response.text.strip()