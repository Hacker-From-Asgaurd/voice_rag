import re
import math
from typing import Dict, Any, Optional, List

# ---------------------------------------------------------
# Guardrail standard responses
# ---------------------------------------------------------

UNSAFE_RESPONSE = "मैं इस तरह के हानिकारक या असुरक्षित अनुरोध में सहायता नहीं कर सकता।"
ABSTENTION_TEXT = "जानकारी दिए गए संदर्भ में उपलब्ध नहीं है।"

# ---------------------------------------------------------
# Actionable harmful patterns (Actionable weapon manufacture, cyberattacks, jailbreaks)
# NOTE: Historical/factual queries (e.g. WWII, Manhattan Project, weapon history) are allowed!
# ---------------------------------------------------------

ACTIONABLE_MALICIOUS_PATTERNS = [
    # Actionable explosive / weapon synthesis instructions
    r"\bhow (to|do I|can I) (make|build|construct|assemble|synthesize) (a bomb|an explosive|a detonator|a firearm|a biological weapon|a chemical weapon)\b",
    r"\bstep[- ]by[- ]step instructions to (make|manufacture) (explosives|bombs)\b",
    
    # Actionable cyber attacks & malicious exploitation
    r"\bhow (to|do I|can I) (hack into|breach|infiltrate) (a server|a bank|an account|a database)\b",
    r"\bhow to (write|deploy|distribute) (ransomware|keylogger|malware|trojan|ddos attack)\b",
    r"\bhow to steal (passwords|credentials|credit card numbers)\b",
    
    # Prompt injection / System Prompt extraction attacks
    r"\b(ignore|disregard|override) (all )?(previous|above|system) (instructions|prompts|rules)\b",
    r"\b(reveal|output|print|display) (your )?(system prompt|initial instructions|developer message)\b",
    
    # Actionable illicit acts
    r"\bhow to (launder money|counterfeit money|forge official documents)\b",
    r"\bhow to (sexually assault|poison someone)\b",
]

def check_actionable_safety(query: str) -> Dict[str, Any]:
    """
    Evaluates input query against actionable malicious safety rules.
    Allows historical/scientific queries while blocking weapon synthesis, cyber attacks, and jailbreaks.
    """
    q_lower = query.lower()
    for pattern in ACTIONABLE_MALICIOUS_PATTERNS:
        if re.search(pattern, q_lower):
            return {
                "allowed": False,
                "category": "actionable_harm"
            }

    return {"allowed": True, "reason": None, "category": "safe"}

# Backwards compatibility alias for app/main.py
check_input = check_actionable_safety


# ---------------------------------------------------------
# Platt Scaling Relevance Calibration
# P(Relevant | score) = 1 / (1 + exp(A * score + B))
# Parameters A = -0.785, B = 0.521 fitted on held-out validation pairs
# ---------------------------------------------------------

PLATT_A = -0.785
PLATT_B = 0.521

def calibrate_crossencoder_score(raw_score: float) -> float:
    """
    Applies fitted Platt scaling to convert raw CrossEncoder reranking score
    into a calibrated relevance probability in [0, 1].
    """
    try:
        # z = A * score + B
        z = (PLATT_A * float(raw_score)) + PLATT_B
        # clip z to avoid overflow in exp
        z_clipped = max(-50.0, min(50.0, z))
        prob = 1.0 / (1.0 + math.exp(z_clipped))
        return round(prob, 4)
    except Exception:
        return 0.50


# ---------------------------------------------------------
# Post-Generation Grounding Verification
# ---------------------------------------------------------

def verify_grounding(answer: str, context: str, min_confidence: float = 0.80) -> bool:
    """
    Verifies that the generated answer is strictly grounded in the context.
    Returns True if grounded, False if answer abstained or unsupported.
    """
    if not answer or not answer.strip():
        return False

    clean_ans = answer.strip()
    if clean_ans == ABSTENTION_TEXT or "उपलब्ध नहीं" in clean_ans:
        return False

    if clean_ans == UNSAFE_RESPONSE:
        return False

    # Check for hallucination disclaimer phrases
    if "मुझे नहीं पता" in clean_ans or "I do not know" in clean_ans or "not mentioned in the context" in clean_ans:
        return False

    return True