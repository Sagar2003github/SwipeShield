
"""
=========================================================
SwipeShield — Local Fraud Intelligence Engine
=========================================================

Retrieval:
    TF-IDF + Cosine Similarity

Uses:
    Local knowledge_base/*.txt files
    SHAP feature contributions supplied by app.py
    Existing Random Forest prediction

No paid APIs, external LLMs, cloud embeddings,
or vector database required.

The ML model remains the source of truth.
RAG retrieves supporting knowledge only.
=========================================================
"""

from pathlib import Path
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "knowledge_base"

MIN_RELEVANCE = 0.05
DEFAULT_TOP_K = 3
MAX_TOP_K = 10


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):
    """Normalize whitespace in a knowledge passage."""

    if not isinstance(text, str):
        return ""

    return re.sub(r"\s+", " ", text).strip()


# =========================================================
# LOAD KNOWLEDGE BASE
# =========================================================

def load_knowledge_base():
    """
    Load TXT documents from knowledge_base.

    Blank lines separate passages. Each passage retains
    its source filename for traceability.
    """

    documents = []

    if not KB_DIR.exists():
        print(f"Knowledge base folder not found: {KB_DIR}")
        return documents

    for file_path in sorted(KB_DIR.glob("*.txt")):
        try:
            text = file_path.read_text(encoding="utf-8")

            passages = re.split(r"\n\s*\n", text)

            for passage in passages:
                passage = clean_text(passage)

                if passage:
                    documents.append({
                        "source": file_path.name,
                        "text": passage,
                    })

        except (OSError, UnicodeError) as error:
            print(
                f"Could not load {file_path.name}: {error}"
            )

    return documents


# =========================================================
# BUILD LOCAL DOCUMENT INDEX
# =========================================================

DOCUMENTS = load_knowledge_base()
VECTORIZER = None
DOCUMENT_MATRIX = None


def build_index():
    """Build the TF-IDF index if usable documents exist."""

    global VECTORIZER, DOCUMENT_MATRIX

    VECTORIZER = None
    DOCUMENT_MATRIX = None

    if not DOCUMENTS:
        print("Local RAG unavailable: no knowledge passages.")
        return

    try:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )

        matrix = vectorizer.fit_transform(
            [item["text"] for item in DOCUMENTS]
        )

        VECTORIZER = vectorizer
        DOCUMENT_MATRIX = matrix

        print(
            "Local RAG index loaded: "
            f"{len(DOCUMENTS)} passages"
        )

    except ValueError as error:
        # For example, all documents may contain only stop words.
        print(f"Could not build local RAG index: {error}")


build_index()


# =========================================================
# RETRIEVE SUPPORTING KNOWLEDGE
# =========================================================

def retrieve(query, top_k=DEFAULT_TOP_K):
    """
    Retrieve the most relevant knowledge passages.

    Returns an empty list if the query is empty, the index
    is unavailable, or no passage meets the relevance floor.
    """

    query = clean_text(query)

    if not query:
        return []

    if (
        not DOCUMENTS
        or VECTORIZER is None
        or DOCUMENT_MATRIX is None
    ):
        return []

    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        top_k = DEFAULT_TOP_K

    top_k = max(1, min(top_k, MAX_TOP_K))

    try:
        query_vector = VECTORIZER.transform([query])

        scores = cosine_similarity(
            query_vector,
            DOCUMENT_MATRIX,
        ).ravel()

        ranked_indices = scores.argsort()[::-1]
        results = []

        for index in ranked_indices:
            score = float(scores[index])

            if score < MIN_RELEVANCE:
                continue

            document = DOCUMENTS[int(index)]

            results.append({
                "source": document["source"],
                "text": document["text"],
                "relevance": round(score, 4),
            })

            if len(results) >= top_k:
                break

        return results

    except Exception as error:
        print(f"Local RAG retrieval failed: {error}")
        return []


# =========================================================
# NORMALIZE EXPLANATION
# =========================================================

def normalize_explanation(explanation):
    """Normalize SHAP explanation items for query construction."""

    normalized = []

    if not isinstance(explanation, (list, tuple)):
        return normalized

    for item in explanation:
        if not isinstance(item, dict):
            continue

        feature = clean_text(str(item.get("feature", "")))
        direction = clean_text(
            str(item.get("direction", ""))
        ).lower()
        impact = clean_text(
            str(item.get("impact", ""))
        ).lower()

        if not feature:
            continue

        if direction not in {"increased", "reduced", "neutral"}:
            direction = "neutral"

        normalized.append({
            "feature": feature,
            "direction": direction,
            "impact": impact or "unspecified",
            "value": item.get("value"),
            "contribution": item.get("contribution"),
        })

    return normalized


# =========================================================
# BUILD CONTEXTUAL QUERY
# =========================================================

def build_query(prediction, risk_score, explanation):
    """
    Build a retrieval query from the existing model result
    and its SHAP feature contributions.

    risk_score is expected to be expressed as a percentage.
    """

    prediction_text = clean_text(str(prediction or "Unknown"))

    try:
        score = float(risk_score)
    except (TypeError, ValueError):
        score = 0.0

    if score >= 80:
        risk_level = "critical"
    elif score >= 50:
        risk_level = "high"
    elif score >= 25:
        risk_level = "medium"
    else:
        risk_level = "low"

    factors = []

    for item in normalize_explanation(explanation)[:5]:
        factors.append(
            f"{item['feature']} "
            f"{item['direction']} fraud risk "
            f"{item['impact']} impact"
        )

    return " ".join([
        "transaction fraud investigation",
        f"prediction {prediction_text}",
        f"risk level {risk_level}",
        f"fraud class probability percentage {score:.2f}",
        *factors,
    ])


# =========================================================
# INVESTIGATION GUIDANCE
# =========================================================

def build_investigation_guidance(
    prediction,
    risk_score,
    explanation,
    retrieved_results,
):
    """
    Generate transparent, rule-based investigation checks.

    These checks are guidance, not proof of fraud and not
    a replacement for the ML model or human review.
    """

    try:
        score = float(risk_score)
    except (TypeError, ValueError):
        score = 0.0

    prediction_text = str(prediction or "Unknown")
    normalized = normalize_explanation(explanation)

    increased = [
        item["feature"]
        for item in normalized
        if item["direction"] == "increased"
    ]

    reduced = [
        item["feature"]
        for item in normalized
        if item["direction"] == "reduced"
    ]

    checks = [
        "Verify the transaction details against authorized records.",
        "Review recent activity and confirm whether it matches expected cardholder behaviour.",
    ]

    if score >= 50 or prediction_text.lower() in {
        "fraudulent", "fraud", "suspicious"
    }:
        checks.append(
            "Prioritize manual review under the organization's fraud procedures."
        )

    if increased:
        checks.append(
            "Review the model features contributing toward the fraud prediction: "
            + ", ".join(increased[:3])
            + "."
        )

    if reduced:
        checks.append(
            "Consider the features contributing away from the fraud prediction: "
            + ", ".join(reduced[:3])
            + "."
        )

    if retrieved_results:
        checks.append(
            "Compare the case with the retrieved local fraud-security guidance."
        )
    else:
        checks.append(
            "No sufficiently relevant local knowledge passage was retrieved; "
            "do not treat the missing retrieval as evidence."
        )

    return {
        "checks": checks,
        "notice": (
            "These are review suggestions based on the model output "
            "and available information. They do not prove fraud."
        ),
    }


# =========================================================
# MAIN FRAUD INTELLIGENCE FUNCTION
# =========================================================

def get_fraud_intelligence(
    prediction,
    risk_score,
    explanation,
    top_k=DEFAULT_TOP_K,
):
    """
    Retrieve local knowledge relevant to a prediction.

    This function never changes the prediction or risk score.
    """

    query = build_query(
        prediction=prediction,
        risk_score=risk_score,
        explanation=explanation,
    )

    results = retrieve(query, top_k=top_k)

    guidance = build_investigation_guidance(
        prediction=prediction,
        risk_score=risk_score,
        explanation=explanation,
        retrieved_results=results,
    )

    return {
        "query": query,
        "results": results,
        "guidance": guidance,
        "engine": "TF-IDF + Cosine Similarity",
        "source": "SwipeShield Local Fraud Knowledge Base",
        "available": VECTORIZER is not None,
    }
