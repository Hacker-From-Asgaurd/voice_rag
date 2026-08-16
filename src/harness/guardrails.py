import re


# ---------------------------------------------------------
# Guardrail responses
# ---------------------------------------------------------

UNSAFE_RESPONSE = (
    "मैं इस तरह के अनुरोध में सहायता नहीं कर सकता।"
)


# ---------------------------------------------------------
# Unsafe / inappropriate patterns
# ---------------------------------------------------------

UNSAFE_PATTERNS = [

    # Violence / weapons
    r"\bhow to (kill|murder|assassinate)\b",
    r"\bhow to make (a bomb|an explosive)\b",
    r"\bhow to build (a bomb|an explosive)\b",
    r"\bhow to make a weapon\b",

    # Cyber abuse
    r"\bhow to hack\b",
    r"\bhack (a|the) (bank|account|server|website)\b",
    r"\bsteal passwords\b",
    r"\bsteal credentials\b",
    r"\bdeploy malware\b",
    r"\bransomware attack\b",

    # Illegal activity
    r"\bhow to launder money\b",
    r"\bhow to evade police\b",
    r"\bhow to make fake id\b",
    r"\bhow to forge documents\b",

    # Explicit sexual requests
    r"\bhow to (rape|sexually assault)\b",
]


def check_input(query):
    """
    Lightweight input safety guardrail.

    Returns:
        {
            "allowed": bool,
            "reason": str | None
        }
    """

    if not isinstance(query, str):
        return {
            "allowed": False,
            "reason": "Invalid query type."
        }

    query = query.strip()

    if not query:
        return {
            "allowed": False,
            "reason": "Empty query."
        }

    query_lower = query.lower()

    for pattern in UNSAFE_PATTERNS:

        if re.search(
            pattern,
            query_lower
        ):

            return {
                "allowed": False,
                "reason": "Unsafe or inappropriate request."
            }

    return {
        "allowed": True,
        "reason": None
    }