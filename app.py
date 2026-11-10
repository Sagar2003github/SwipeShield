import os
import csv
import io
import secrets
import sqlite3
from pathlib import Path
from functools import wraps

import joblib
import numpy as np
import pandas as pd
import networkx as nx

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    session,
    Response,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# =========================================================
# OPTIONAL MODULES
# =========================================================

try:
    import shap
except ImportError:
    shap = None

try:
    from rag_engine import get_fraud_intelligence
except ImportError:
    get_fraud_intelligence = None


# =========================================================
# APPLICATION CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "transactions.db"
MODEL_PATH = BASE_DIR / "fraud_model.pkl"
TEST_DATA_PATH = BASE_DIR / "creditcard.csv"

app = Flask(__name__)
app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "swipeshield_secret_key",
)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

FEATURE_NAMES = ["V1", "V2", "V3", "V4", "V14", "Amount"]

SYSTEM_SETTINGS = {
    "fraud_threshold": 50.0,
    "email_alerts": True,
    "auto_freeze": False,
}


# =========================================================
# LOAD MODEL
# =========================================================

try:
    model = joblib.load(MODEL_PATH)
    print("Fraud detection model loaded successfully.")
except Exception as exc:
    app.logger.warning("Could not load fraud_model.pkl: %s", exc)
    model = None


# =========================================================
# DATABASE
# =========================================================

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                password TEXT,
                full_name TEXT,
                age INTEGER,
                bank_name TEXT,
                card_number TEXT,
                address TEXT,
                city TEXT,
                country TEXT,
                api_key TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                card_number TEXT,
                amount REAL,
                prediction TEXT,
                risk_score REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        user_columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(users)"
            ).fetchall()
        }

        if "api_key" not in user_columns:
            conn.execute(
                "ALTER TABLE users ADD COLUMN api_key TEXT"
            )

        log_columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(audit_logs)"
            ).fetchall()
        }

        if "risk_score" not in log_columns:
            conn.execute(
                "ALTER TABLE audit_logs ADD COLUMN risk_score REAL"
            )

        if "timestamp" not in log_columns:
            conn.execute(
                "ALTER TABLE audit_logs "
                "ADD COLUMN timestamp DATETIME"
            )

        log_columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(audit_logs)"
            ).fetchall()
        }

        for column in ("v1", "v2", "v3", "v4", "v14"):
            if column not in log_columns:
                conn.execute(
                    f"ALTER TABLE audit_logs ADD COLUMN {column} REAL"
                )

        # Actual outcome: 0 = legitimate, 1 = fraud.
        # NULL means the transaction has not yet been verified.
        if "actual_class" not in log_columns:
            conn.execute(
                "ALTER TABLE audit_logs ADD COLUMN actual_class INTEGER"
            )

        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_performance_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_count INTEGER NOT NULL,
                accuracy REAL NOT NULL,
                precision REAL NOT NULL,
                recall REAL NOT NULL,
                f1_score REAL NOT NULL,
                true_negative INTEGER NOT NULL,
                false_positive INTEGER NOT NULL,
                false_negative INTEGER NOT NULL,
                true_positive INTEGER NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_logs_actual_class "
            "ON audit_logs(actual_class)"
        )


init_db()


# =========================================================
# COMMON HELPERS
# =========================================================

def is_logged_in():
    return "user" in session


def unauthorized_response():
    return jsonify({
        "status": "error",
        "message": "Unauthorized. Please log in.",
    }), 401


def login_required(view_function):
    @wraps(view_function)
    def wrapper(*args, **kwargs):
        if not is_logged_in():
            if request.path.startswith("/api/") or request.is_json:
                return unauthorized_response()
            return redirect(url_for("login"))
        return view_function(*args, **kwargs)

    return wrapper


def build_features(v1, v2, v3, v4, v14, amount):
    values = [v1, v2, v3, v4, v14, amount]

    if not np.isfinite(values).all():
        raise ValueError("All transaction values must be finite numbers.")

    return pd.DataFrame([values], columns=FEATURE_NAMES)


def predict_fraud_probability(features):
    """Return the probability associated with fraud class 1."""
    if model is None:
        raise RuntimeError("The fraud model is not loaded.")

    if not hasattr(model, "predict_proba"):
        raise RuntimeError(
            "The loaded model does not support predict_proba()."
        )

    probabilities = model.predict_proba(features)[0]
    classes = list(getattr(model, "classes_", [0, 1]))

    if 1 not in classes:
        raise RuntimeError(
            f"Fraud class 1 was not found in model classes: {classes}"
        )

    return float(probabilities[classes.index(1)])


def log_transaction(
    card_number,
    amount,
    prediction,
    risk_score,
    features=None,
):
    """Save transaction information and optional model features."""
    feature_values = {
        "v1": None,
        "v2": None,
        "v3": None,
        "v4": None,
        "v14": None,
    }

    if features is not None:
        row = features.iloc[0]
        feature_values = {
            "v1": float(row["V1"]),
            "v2": float(row["V2"]),
            "v3": float(row["V3"]),
            "v4": float(row["V4"]),
            "v14": float(row["V14"]),
        }

    with get_db_connection() as conn:
        conn.execute("""
            INSERT INTO audit_logs (
                card_number, amount, prediction, risk_score,
                v1, v2, v3, v4, v14
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            card_number,
            amount,
            prediction,
            risk_score,
            feature_values["v1"],
            feature_values["v2"],
            feature_values["v3"],
            feature_values["v4"],
            feature_values["v14"],
        ))


def get_model_metadata():
    metadata = {
        "model": type(model).__name__ if model is not None else "Unavailable",
        "algorithm": type(model).__name__ if model is not None else "Unavailable",
        "features": FEATURE_NAMES,
        "explainability": "SHAP TreeExplainer" if shap else "Unavailable",
        "rag": (
            "TF-IDF + Cosine Similarity"
            if get_fraud_intelligence
            else "Unavailable"
        ),
    }

    if model is not None and hasattr(model, "n_estimators"):
        metadata["trees"] = int(model.n_estimators)

    if model is not None and hasattr(model, "max_depth"):
        metadata["max_depth"] = model.max_depth

    return metadata


# =========================================================
# UPDATED SHAP EXPLAINABILITY
# =========================================================

def get_shap_explanation(features):
    """
    Return the strongest SHAP feature contributions for a transaction.

    Contributions explain model output and should not automatically
    be interpreted as percentage-point changes in fraud probability.
    """
    if model is None or shap is None:
        return []

    try:
        explainer = shap.TreeExplainer(model)

        try:
            shap_output = explainer(features)
            raw_values = shap_output.values
        except Exception:
            raw_values = explainer.shap_values(features)

        classes = list(getattr(model, "classes_", [0, 1]))
        fraud_class_index = classes.index(1) if 1 in classes else 0

        if isinstance(raw_values, list):
            class_index = min(fraud_class_index, len(raw_values) - 1)
            values = np.asarray(raw_values[class_index])
        else:
            values = np.asarray(raw_values)

            if values.ndim == 3:
                if values.shape[-1] > fraud_class_index:
                    values = values[:, :, fraud_class_index]
                elif values.shape[1] > fraud_class_index:
                    values = values[:, fraud_class_index, :]

            if values.ndim == 2:
                values = values[0]

        values = np.asarray(values).reshape(-1)
        row = features.iloc[0].to_numpy(dtype=float)

        if len(values) != len(FEATURE_NAMES):
            app.logger.warning(
                "Unexpected SHAP output size: got %s values for %s features.",
                len(values),
                len(FEATURE_NAMES),
            )
            return []

        ranked = sorted(
            zip(FEATURE_NAMES, row, values),
            key=lambda item: abs(float(item[2])),
            reverse=True,
        )

        explanation = []

        for name, value, contribution in ranked[:6]:
            contribution = float(contribution)
            magnitude = abs(contribution)

            explanation.append({
                "feature": name,
                "value": round(float(value), 4),
                "contribution": round(contribution, 6),
                "direction": (
                    "increased" if contribution > 0 else "reduced"
                ),
                "impact": (
                    "high" if magnitude >= 0.10
                    else "medium" if magnitude >= 0.03
                    else "low"
                ),
            })

        return explanation

    except Exception as exc:
        app.logger.warning("SHAP explanation failed: %s", exc)
        return []


def build_investigation_guidance(prediction, risk_score, explanation):
    """
    Return consistent risk bands and deterministic investigation guidance.
    Guidance is advisory and does not independently establish fraud.
    """
    risk_score = float(risk_score)

    increased = [
        item.get("feature")
        for item in explanation
        if item.get("direction") == "increased" and item.get("feature")
    ]

    if risk_score >= 80:
        severity = "critical"
        risk_band = "Critical"
        checks = [
            "Prioritize investigation of this transaction.",
            "Verify the transaction directly with the cardholder using an approved contact method.",
            "Review recent transactions for unusual amounts, frequency, and related activity.",
            "Check connected transactions in the fraud network view.",
        ]
        recommended_action = (
            "CRITICAL RISK: Prioritize investigation and verify cardholder "
            "activity and transaction context before approving or taking further action."
        )
    elif risk_score >= 50:
        severity = "high"
        risk_band = "High"
        checks = [
            "Review the transaction details and recent account activity.",
            "Verify cardholder activity before taking further action.",
            "Check related transactions for repeated or unusual patterns.",
        ]
        recommended_action = (
            "HIGH RISK: Review the transaction and verify cardholder activity "
            "before taking further action."
        )
    elif risk_score >= 25:
        severity = "medium"
        risk_band = "Medium"
        checks = [
            "Apply additional scrutiny to the transaction context.",
            "Review recent account activity for unusual patterns.",
            "Escalate if other suspicious activity or alerts are present.",
        ]
        recommended_action = (
            "MEDIUM RISK: Apply additional scrutiny and review transaction "
            "context before deciding whether further verification is needed."
        )
    else:
        severity = "low"
        risk_band = "Low"
        checks = [
            "Continue normal transaction monitoring.",
            "Escalate if other suspicious activity or alerts are present.",
        ]
        recommended_action = (
            "LOW RISK: Continue normal transaction security monitoring."
        )

    return {
        "severity": severity,
        "risk_level": risk_band,
        "model_prediction": prediction,
        "risk_score": round(risk_score, 2),
        "risk_threshold": float(SYSTEM_SETTINGS["fraud_threshold"]),
        "features_increasing_model_output": increased,
        "recommended_action": recommended_action,
        "recommended_checks": checks,
        "disclaimer": (
            "This is model-assisted guidance, not proof of fraud. "
            "Verify evidence before taking action."
        ),
    }


def parse_transaction_input(data):
    """Read transaction fields using the existing lowercase form keys."""
    amount = float(data.get("amount", 0))
    v1 = float(data.get("v1", 0))
    v2 = float(data.get("v2", 0))
    v3 = float(data.get("v3", 0))
    v4 = float(data.get("v4", 0))
    v14 = float(data.get("v14", 0))

    features = build_features(v1, v2, v3, v4, v14, amount)
    card_number = str(
        data.get("card_number", "**** **** **** 1234")
    ).strip()

    if not card_number:
        card_number = "**** **** **** 1234"

    return card_number, amount, features


def classify_risk(risk_score):
    threshold = float(SYSTEM_SETTINGS["fraud_threshold"])
    return "Fraudulent" if risk_score >= threshold else "Legitimate"


def evaluate_dataset():
    """Evaluate against labelled CSV rows; training-data overlap is unverified."""
    if model is None:
        raise RuntimeError("The fraud model is unavailable.")

    if not TEST_DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {TEST_DATA_PATH.name}"
        )

    df = pd.read_csv(TEST_DATA_PATH)
    required = FEATURE_NAMES + ["Class"]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            "Missing dataset columns: " + ", ".join(missing)
        )

    df = df[required].copy()

    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if df.empty:
        raise ValueError("Dataset contains no valid labelled rows.")

    if not df["Class"].isin([0, 1]).all():
        raise ValueError("Class must contain only 0 and 1.")

    expected_features = getattr(model, "n_features_in_", len(FEATURE_NAMES))
    if expected_features != len(FEATURE_NAMES):
        raise ValueError(
            f"Model expects {expected_features} features, "
            f"but the API provides {len(FEATURE_NAMES)}."
        )

    y_true = df["Class"].astype(int)
    y_pred = np.asarray(model.predict(df[FEATURE_NAMES])).ravel()

    if len(y_pred) != len(y_true):
        raise ValueError("Prediction count does not match label count.")

    if not set(np.unique(y_pred)).issubset({0, 1}):
        raise ValueError("Model predictions must be binary labels 0 and 1.")

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
        "precision": round(
            precision_score(y_true, y_pred, zero_division=0) * 100, 2
        ),
        "recall": round(
            recall_score(y_true, y_pred, zero_division=0) * 100, 2
        ),
        "f1_score": round(
            f1_score(y_true, y_pred, zero_division=0) * 100, 2
        ),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "evaluated_rows": int(len(df)),
    }

# =========================================================
# LOGIN
# =========================================================

@app.route("/")
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        data = request.get_json(silent=True) or request.form
        username = str(data.get("username", "")).strip()
        password = str(data.get("password", ""))

        with get_db_connection() as conn:
            user = conn.execute(
                """
                SELECT id FROM users
                WHERE username = ? AND password = ?
                """,
                (username, password),
            ).fetchone()

        if user or (username == "admin" and password == "admin"):
            session.clear()
            session["user"] = username

            if request.is_json:
                return jsonify({
                    "status": "success",
                    "redirect": "/dashboard",
                })

            return redirect(url_for("dashboard"))

        if request.is_json:
            return jsonify({
                "status": "error",
                "message": "Invalid username or password!",
            }), 401

        return render_template(
            "login.html",
            error="Invalid username or password!",
        )

    if is_logged_in():
        return redirect(url_for("dashboard"))

    return render_template("login.html")


# =========================================================
# SIGNUP
# =========================================================

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        data = request.get_json(silent=True) or request.form

        username = str(data.get("username", "")).strip()
        password = str(data.get("password", ""))

        if not username or not password:
            message = "Username and password are required."

            if request.is_json:
                return jsonify({
                    "status": "error",
                    "message": message,
                }), 400

            return render_template("signup.html", error=message)

        full_name = " ".join([
            str(data.get("first_name", "")),
            str(data.get("middle_name", "")),
            str(data.get("last_name", "")),
        ]).strip()

        age_value = data.get("age")

        try:
            age = int(age_value) if age_value not in (None, "") else None
        except (ValueError, TypeError):
            age = None

        try:
            with get_db_connection() as conn:
                conn.execute("""
                    INSERT INTO users (
                        username, password, full_name, age,
                        bank_name, card_number, address, city, country
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    username,
                    password,
                    full_name,
                    age,
                    data.get("bank_name"),
                    data.get("card_number"),
                    data.get("address"),
                    data.get("city"),
                    data.get("country"),
                ))

            if request.is_json:
                return jsonify({
                    "status": "success",
                    "message": "Account created successfully!",
                })

            return redirect(url_for("login"))

        except sqlite3.IntegrityError:
            message = "Username / Email already registered!"

            if request.is_json:
                return jsonify({
                    "status": "error",
                    "message": message,
                }), 409

            return render_template("signup.html", error=message)

    return render_template("signup.html")


# =========================================================
# PAGE ROUTES
# =========================================================

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", username=session["user"])


@app.route("/predict_page")
@login_required
def predict_page():
    return render_template("predict.html", username=session["user"])


@app.route("/logs_page")
@login_required
def logs_page():
    return render_template("logs.html", username=session["user"])


@app.route("/alerts_page")
@login_required
def alerts_page():
    return render_template("alerts.html", username=session["user"])


@app.route("/model_health")
@login_required
def model_health():
    return render_template("model_health.html", username=session["user"])


@app.route("/map_page")
@login_required
def map_page():
    return render_template("geo_map.html", username=session["user"])


@app.route("/network_page")
@login_required
def network_page():
    return render_template("network.html", username=session["user"])


@app.route("/settings")
@login_required
def settings():
    return render_template("settings.html", username=session["user"])


@app.route("/profile")
@login_required
def profile():
    return render_template("profile.html", username=session["user"])


# =========================================================
# SINGLE TRANSACTION PREDICTION + INVESTIGATION ASSISTANT
# =========================================================

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    data = request.get_json(silent=True) or request.form

    try:
        card_number, amount, features = parse_transaction_input(data)

        if model is None:
            return jsonify({
                "status": "error",
                "message": "Fraud model is unavailable. Check fraud_model.pkl.",
            }), 503

        risk_score = round(
            predict_fraud_probability(features) * 100,
            2,
        )

    except (ValueError, TypeError) as exc:
        return jsonify({
            "status": "error",
            "message": f"Invalid transaction input: {exc}",
        }), 400

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "status": "error",
            "message": "Prediction failed. Check the server log.",
        }), 500

    prediction = classify_risk(risk_score)
    explanation = get_shap_explanation(features)

    try:
        log_transaction(
            card_number,
            amount,
            prediction,
            risk_score,
            features=features,
        )
    except sqlite3.Error:
        app.logger.exception("Failed to save transaction audit record")
        return jsonify({
            "status": "error",
            "message": "Prediction completed, but audit logging failed.",
        }), 500

    increased = [
        item["feature"]
        for item in explanation
        if item.get("direction") == "increased"
    ]

    reduced = [
        item["feature"]
        for item in explanation
        if item.get("direction") == "reduced"
    ]

    if prediction == "Fraudulent":
        if increased:
            summary = (
                "The model flagged this transaction. Features contributing "
                "toward the model output include "
                + ", ".join(increased[:2])
                + "."
            )
        else:
            summary = (
                "The model classified this transaction as potentially fraudulent."
            )
    else:
        if reduced:
            summary = (
                "Features contributing toward lowering the model output include "
                + ", ".join(reduced[:2])
                + "."
            )
        else:
            summary = (
                "The model classified this transaction as likely legitimate."
            )

    fraud_intelligence = {
        "query": "",
        "results": [],
        "engine": "unavailable",
        "source": "SwipeShield Local Fraud Knowledge Base",
    }

    if get_fraud_intelligence:
        try:
            retrieved = get_fraud_intelligence(
                prediction=prediction,
                risk_score=risk_score,
                explanation=explanation,
                top_k=3,
            )

            if isinstance(retrieved, dict):
                fraud_intelligence = retrieved

        except Exception as exc:
            app.logger.warning("RAG retrieval failed: %s", exc)

    investigation_guidance = build_investigation_guidance(
        prediction=prediction,
        risk_score=risk_score,
        explanation=explanation,
    )

    action = investigation_guidance["recommended_action"]

    return jsonify({
        "status": "success",
        "prediction": prediction,
        "risk_score": risk_score,
        "risk_level": investigation_guidance["risk_level"],
        "explanation": explanation,
        "ai_summary": summary,
        "recommended_action": action,
        "explainability": "SHAP" if explanation else "unavailable",
        "fraud_intelligence": fraud_intelligence,
        "investigation_guidance": investigation_guidance,
    })


# =========================================================
# BATCH CSV PREDICTION
# =========================================================

@app.route("/predict_batch", methods=["POST"])
@login_required
def predict_batch():
    if "file" not in request.files:
        return jsonify({
            "status": "error",
            "message": "No file uploaded!",
        }), 400

    file = request.files["file"]

    if not file.filename:
        return jsonify({
            "status": "error",
            "message": "Empty file selected!",
        }), 400

    if model is None:
        return jsonify({
            "status": "error",
            "message": "Fraud model is unavailable.",
        }), 503

    try:
        df = pd.read_csv(file)

        required = [
            "card_number", "amount", "v1", "v2",
            "v3", "v4", "v14",
        ]

        missing = [
            column for column in required
            if column not in df.columns
        ]

        if missing:
            return jsonify({
                "status": "error",
                "message": (
                    "Missing required CSV columns: "
                    + ", ".join(missing)
                ),
            }), 400

        prepared = []

        # Validate and predict every row before inserting records.
        for row_number, (_, row) in enumerate(df.iterrows(), start=2):
            try:
                values = [
                    float(row["v1"]),
                    float(row["v2"]),
                    float(row["v3"]),
                    float(row["v4"]),
                    float(row["v14"]),
                    float(row["amount"]),
                ]
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Invalid numeric value in CSV row {row_number}."
                ) from exc

            if not np.isfinite(values).all():
                raise ValueError(
                    f"Non-finite numeric value in CSV row {row_number}."
                )

            features = pd.DataFrame(
                [values],
                columns=FEATURE_NAMES,
            )

            risk = round(
                predict_fraud_probability(features) * 100,
                2,
            )

            prediction = classify_risk(risk)

            prepared.append({
                "card": str(row["card_number"]),
                "amount": values[5],
                "prediction": prediction,
                "risk": risk,
                "risk_level": build_investigation_guidance(
                    prediction, risk, []
                )["risk_level"],
                "v1": values[0],
                "v2": values[1],
                "v3": values[2],
                "v4": values[3],
                "v14": values[4],
            })

        with get_db_connection() as conn:
            conn.executemany("""
                INSERT INTO audit_logs (
                    card_number, amount, prediction, risk_score,
                    v1, v2, v3, v4, v14
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    item["card"],
                    item["amount"],
                    item["prediction"],
                    item["risk"],
                    item["v1"],
                    item["v2"],
                    item["v3"],
                    item["v4"],
                    item["v14"],
                )
                for item in prepared
            ])

        results = [
            {
                "card": item["card"],
                "amount": item["amount"],
                "prediction": item["prediction"],
                "risk": item["risk"],
                "risk_level": item["risk_level"],
            }
            for item in prepared
        ]

        fraud_count = sum(
            item["prediction"] == "Fraudulent"
            for item in results
        )

        return jsonify({
            "status": "success",
            "total": len(results),
            "fraud_count": fraud_count,
            "safe_count": len(results) - fraud_count,
            "results": results,
        })

    except Exception as exc:
        app.logger.exception("Batch prediction failed")
        return jsonify({
            "status": "error",
            "message": f"Error processing CSV: {exc}",
        }), 400

# =========================================================
# AUDIT LOGS
# =========================================================

@app.route("/api/logs", methods=["GET"])
@login_required
def get_logs():
    try:
        limit = int(request.args.get("limit", 100))
    except (ValueError, TypeError):
        limit = 100

    limit = max(1, min(limit, 1000))

    with get_db_connection() as conn:
        rows = conn.execute("""
            SELECT id, card_number, amount, prediction,
                   risk_score, timestamp
            FROM audit_logs
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()

    return jsonify([
        {
            "id": row["id"],
            "card_number": row["card_number"],
            "amount": row["amount"],
            "prediction": row["prediction"],
            "risk_score": row["risk_score"],
            "timestamp": row["timestamp"],
        }
        for row in rows
    ])


@app.route("/api/logs/export", methods=["GET"])
@login_required
def export_logs():
    with get_db_connection() as conn:
        rows = conn.execute("""
            SELECT id, card_number, amount, prediction,
                   risk_score, timestamp
            FROM audit_logs
            ORDER BY id DESC
        """).fetchall()

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "Transaction ID",
        "Card Number",
        "Amount",
        "Prediction",
        "Risk Score",
        "Timestamp",
    ])

    for row in rows:
        writer.writerow([
            row["id"],
            row["card_number"],
            row["amount"],
            row["prediction"],
            row["risk_score"],
            row["timestamp"],
        ])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={
            "Content-Disposition":
                "attachment; filename=SwipeShield_Audit_Trail.csv"
        },
    )


# =========================================================
# LEGACY AUDIT LOG ROUTES
# =========================================================

@app.route("/logs", methods=["GET"])
@login_required
def get_logs_page_data():
    with get_db_connection() as conn:
        rows = conn.execute("""
            SELECT id, card_number, amount, prediction,
                   risk_score, timestamp
            FROM audit_logs
            ORDER BY id DESC
            LIMIT 1000
        """).fetchall()

    return jsonify([
        {
            "id": row["id"],
            "card": row["card_number"],
            "amount": row["amount"],
            "prediction": row["prediction"],
            "risk": row["risk_score"],
            "time": row["timestamp"],
        }
        for row in rows
    ])


@app.route("/export_logs", methods=["GET"])
@login_required
def export_logs_legacy():
    with get_db_connection() as conn:
        rows = conn.execute("""
            SELECT id, card_number, amount, prediction,
                   risk_score, timestamp
            FROM audit_logs
            ORDER BY id DESC
        """).fetchall()

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "Transaction ID",
        "Card Number",
        "Amount ($)",
        "Prediction",
        "Risk Score (%)",
        "Timestamp",
    ])

    for row in rows:
        writer.writerow([
            row["id"],
            row["card_number"],
            row["amount"],
            row["prediction"],
            row["risk_score"],
            row["timestamp"],
        ])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={
            "Content-Disposition":
                "attachment; filename=SwipeShield_Audit_Trail.csv"
        },
    )


# =========================================================
# DASHBOARD ANALYTICS
# =========================================================

@app.route("/analytics", methods=["GET"])
@login_required
def get_analytics():
    with get_db_connection() as conn:
        rows = conn.execute("""
            SELECT id, risk_score, timestamp
            FROM audit_logs
            ORDER BY id DESC
            LIMIT 10
        """).fetchall()

    rows = list(reversed(rows))

    labels = [
        row["timestamp"] or f"Transaction {row['id']}"
        for row in rows
    ]

    scores = [
        float(row["risk_score"] or 0)
        for row in rows
    ]

    transactions = [
        {
            "id": row["id"],
            "risk": row["risk_score"],
            "timestamp": row["timestamp"],
        }
        for row in rows
    ]

    return jsonify({
        "labels": labels,
        "scores": scores,
        "transactions": transactions,
    })


# =========================================================
# FRAUD NETWORK ANALYSIS
# =========================================================

def mask_card_number(card_number):
    """Mask card-like identifiers before returning them to the UI."""
    if card_number is None:
        return "Unknown"

    value = str(card_number).strip()

    if not value:
        return "Unknown"

    if len(value) <= 4:
        return "••••"

    return "•••• •••• " + value[-4:]


@app.route("/api/fraud-network", methods=["GET"])
@login_required
def fraud_network_api():
    """Return recent transactions as graph nodes, links, and statistics."""
    try:
        limit = request.args.get("limit", default=200, type=int)
        limit = max(10, min(limit or 200, 500))

        with get_db_connection() as conn:
            rows = conn.execute("""
                SELECT id, card_number, amount, prediction, risk_score,
                       timestamp, v1, v2, v3, v4, v14
                FROM audit_logs
                ORDER BY id DESC
                LIMIT ?
            """, (limit,)).fetchall()

        transactions = [dict(row) for row in reversed(rows)]
        graph = nx.Graph()

        for tx in transactions:
            try:
                amount = float(tx.get("amount") or 0)
            except (TypeError, ValueError):
                amount = 0.0

            try:
                risk = float(tx.get("risk_score") or 0)
            except (TypeError, ValueError):
                risk = 0.0

            graph.add_node(
                str(tx["id"]),
                transaction_id=int(tx["id"]),
                card=mask_card_number(tx.get("card_number")),
                amount=amount,
                prediction=tx.get("prediction") or "Unknown",
                risk=round(risk, 2),
                timestamp=tx.get("timestamp"),
            )

        # Relationship 1: shared, non-placeholder card identifiers.
        card_groups = {}

        for tx in transactions:
            raw_card = str(tx.get("card_number") or "").strip()
            normalized_card = raw_card.replace(" ", "").replace("-", "")

            if (
                normalized_card
                and "*" not in normalized_card
                and "•" not in normalized_card
                and normalized_card.lower()
                not in {"none", "null", "nan", "unknown"}
            ):
                card_groups.setdefault(normalized_card, []).append(tx)

        for group in card_groups.values():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    graph.add_edge(
                        str(group[i]["id"]),
                        str(group[j]["id"]),
                        relationship="same_card",
                        similarity=1.0,
                    )

        # Relationship 2: similar stored transaction features.
        feature_names = ["v1", "v2", "v3", "v4", "v14", "amount"]
        usable = []

        for tx in transactions:
            values = [tx.get(name) for name in feature_names]

            if any(value is None for value in values):
                continue

            try:
                numeric = np.asarray(values, dtype=float)
            except (TypeError, ValueError):
                continue

            if np.isfinite(numeric).all():
                usable.append((tx, numeric))

        similarity_threshold = 0.95

        if len(usable) >= 2:
            matrix = np.vstack([item[1] for item in usable])
            scaled = StandardScaler().fit_transform(matrix)
            similarities = cosine_similarity(scaled)

            for i in range(len(usable)):
                for j in range(i + 1, len(usable)):
                    similarity = float(similarities[i, j])

                    if similarity < similarity_threshold:
                        continue

                    left = str(usable[i][0]["id"])
                    right = str(usable[j][0]["id"])

                    if graph.has_edge(left, right):
                        continue

                    graph.add_edge(
                        left,
                        right,
                        relationship="similar_features",
                        similarity=round(similarity, 4),
                    )

        degrees = dict(graph.degree())

        if graph.number_of_nodes() > 1:
            centrality = nx.degree_centrality(graph)
        else:
            centrality = {
                node_id: 0.0
                for node_id in graph.nodes
            }

        components = list(nx.connected_components(graph))
        cluster_map = {}

        for cluster_id, component in enumerate(components, start=1):
            for node_id in component:
                cluster_map[node_id] = cluster_id

        nodes = []
        threshold = float(SYSTEM_SETTINGS["fraud_threshold"])

        for node_id, attributes in graph.nodes(data=True):
            suspicious = (
                attributes["prediction"] == "Fraudulent"
                or attributes["risk"] >= threshold
            )

            nodes.append({
                "id": node_id,
                **attributes,
                "connections": int(degrees.get(node_id, 0)),
                "centrality": round(
                    float(centrality.get(node_id, 0)),
                    4,
                ),
                "cluster": int(cluster_map.get(node_id, 0)),
                "suspicious": bool(suspicious),
            })

        edges = []

        for left, right, attributes in graph.edges(data=True):
            edges.append({
                "source": left,
                "target": right,
                "relationship": attributes.get("relationship", "unknown"),
                "similarity": float(attributes.get("similarity", 0)),
            })

        return jsonify({
            "status": "success",
            "nodes": nodes,
            "edges": edges,
            "summary": {
                "transactions": graph.number_of_nodes(),
                "connections": graph.number_of_edges(),
                "clusters": len(components),
                "suspicious_transactions": sum(
                    node["suspicious"] for node in nodes
                ),
            },
            "settings": {
                "feature_similarity_threshold": similarity_threshold,
                "transaction_limit": limit,
                "note": (
                    "Connections indicate shared identifiers or feature "
                    "similarity; they do not prove fraud. Older records "
                    "may not contain feature values."
                ),
            },
        })

    except Exception:
        app.logger.exception("Fraud network analysis failed")

        return jsonify({
            "status": "error",
            "message": (
                "Could not generate fraud network. "
                "Check the server logs."
            ),
        }), 500


# =========================================================
# ALERTS
# =========================================================

@app.route("/api/alerts", methods=["GET"])
@login_required
def get_alerts():
    with get_db_connection() as conn:
        rows = conn.execute("""
            SELECT id, card_number, amount, prediction,
                   risk_score, timestamp
            FROM audit_logs
            WHERE prediction = 'Fraudulent'
            ORDER BY id DESC
            LIMIT 20
        """).fetchall()

    return jsonify([
        {
            "id": row["id"],
            "card": row["card_number"],
            "amount": row["amount"],
            "prediction": row["prediction"],
            "risk": row["risk_score"],
            "time": row["timestamp"],
        }
        for row in rows
    ])


# =========================================================
# MODEL HEALTH
# =========================================================

@app.route("/api/model_metrics", methods=["GET"])
@login_required
def model_metrics():
    result = get_model_metadata()

    result.update({
        "status": "success",
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1_score": None,
        "confusion_matrix": {
            "true_negative": None,
            "false_positive": None,
            "false_negative": None,
            "true_positive": None,
        },
        "feature_importance": {},
        "evaluation_status": "",
    })

    if model is None:
        result["status"] = "error"
        result["evaluation_status"] = (
            "Model could not be loaded. Check fraud_model.pkl."
        )
        return jsonify(result), 503

    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_).ravel()

        if len(importance) == len(FEATURE_NAMES):
            result["feature_importance"] = {
                name: round(float(value) * 100, 2)
                for name, value in zip(FEATURE_NAMES, importance)
            }

    try:
        metrics = evaluate_dataset()
        result.update(metrics)

        result["evaluation_status"] = (
            f"Evaluated on {metrics['evaluated_rows']:,} labelled rows "
            f"from {TEST_DATA_PATH.name}. Training-data overlap "
            "has not been verified."
        )

    except FileNotFoundError as exc:
        result["status"] = "error"
        result["evaluation_status"] = str(exc)
        return jsonify(result), 404

    except Exception as exc:
        app.logger.exception("Model metrics evaluation failed")
        result["status"] = "error"
        result["evaluation_status"] = f"Evaluation failed: {exc}"
        return jsonify(result), 400

    return jsonify(result)

# =========================================================
# MODEL PERFORMANCE MONITORING
# =========================================================

def calculate_verified_performance():
    """
    Calculate performance using only transactions with verified labels.

    actual_class:
        0 = Legitimate
        1 = Fraud

    This measures model performance on the verified records available
    in audit_logs. It does not automatically establish future performance.
    """
    if model is None:
        raise RuntimeError("The fraud model is unavailable.")

    with get_db_connection() as conn:
        rows = conn.execute("""
            SELECT id, prediction, actual_class
            FROM audit_logs
            WHERE actual_class IN (0, 1)
            ORDER BY id ASC
        """).fetchall()

    if not rows:
        return {
            "sample_count": 0,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "confusion_matrix": {
                "true_negative": 0,
                "false_positive": 0,
                "false_negative": 0,
                "true_positive": 0,
            },
            "message": (
                "No verified transaction labels are available yet. "
                "Label transactions after their outcomes are confirmed."
            ),
        }

    y_true = []
    y_pred = []

    for row in rows:
        actual_class = int(row["actual_class"])
        prediction = str(row["prediction"] or "").strip().lower()

        if prediction == "fraudulent":
            predicted_class = 1
        elif prediction == "legitimate":
            predicted_class = 0
        else:
            continue

        y_true.append(actual_class)
        y_pred.append(predicted_class)

    if not y_true:
        raise ValueError("No valid labelled prediction records were found.")

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "sample_count": len(y_true),
        "accuracy": round(
            accuracy_score(y_true, y_pred) * 100, 2
        ),
        "precision": round(
            precision_score(y_true, y_pred, zero_division=0) * 100, 2
        ),
        "recall": round(
            recall_score(y_true, y_pred, zero_division=0) * 100, 2
        ),
        "f1_score": round(
            f1_score(y_true, y_pred, zero_division=0) * 100, 2
        ),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "message": (
            "Metrics are calculated from transactions with verified labels."
        ),
    }


@app.route("/api/model-performance", methods=["GET"])
@login_required
def model_performance_api():
    """Return performance metrics from verified transaction outcomes."""
    try:
        metrics = calculate_verified_performance()

        return jsonify({
            "status": "success",
            "metrics": metrics,
        })

    except Exception:
        app.logger.exception("Verified performance calculation failed")

        return jsonify({
            "status": "error",
            "message": "Could not calculate model performance.",
        }), 500


@app.route("/api/model-performance/label", methods=["POST"])
@login_required
def label_transaction_outcome():
    """
    Record the verified outcome for an existing transaction.

    JSON:
        {
            "transaction_id": 1,
            "actual_class": 1
        }

    Only authorized users should be permitted to submit verified labels.
    """
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "status": "error",
            "message": "A valid JSON request body is required.",
        }), 400

    try:
        transaction_id = int(data.get("transaction_id"))
        actual_class = int(data.get("actual_class"))

        if transaction_id <= 0:
            raise ValueError

        if actual_class not in (0, 1):
            raise ValueError

    except (ValueError, TypeError, OverflowError):
        return jsonify({
            "status": "error",
            "message": (
                "transaction_id must be positive and actual_class "
                "must be 0 (Legitimate) or 1 (Fraud)."
            ),
        }), 400

    try:
        with get_db_connection() as conn:
            existing = conn.execute(
                "SELECT id FROM audit_logs WHERE id = ?",
                (transaction_id,),
            ).fetchone()

            if existing is None:
                return jsonify({
                    "status": "error",
                    "message": "Transaction ID was not found.",
                }), 404

            conn.execute("""
                UPDATE audit_logs
                SET actual_class = ?
                WHERE id = ?
            """, (actual_class, transaction_id))

        return jsonify({
            "status": "success",
            "message": "Verified transaction outcome saved.",
            "transaction_id": transaction_id,
            "actual_class": actual_class,
            "actual_label": (
                "Fraud" if actual_class == 1 else "Legitimate"
            ),
        })

    except sqlite3.Error:
        app.logger.exception("Could not save verified transaction outcome")

        return jsonify({
            "status": "error",
            "message": "Could not save the transaction outcome.",
        }), 500


@app.route("/api/model-performance/snapshot", methods=["POST"])
@login_required
def save_model_performance_snapshot():
    """
    Calculate current verified metrics and save a historical snapshot.
    Snapshots are not created when no verified labels are available.
    """
    try:
        metrics = calculate_verified_performance()

        if metrics["sample_count"] == 0:
            return jsonify({
                "status": "error",
                "message": (
                    "No verified labels available. "
                    "A performance snapshot was not created."
                ),
                "metrics": metrics,
            }), 400

        matrix = metrics["confusion_matrix"]

        with get_db_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO model_performance_snapshots (
                    sample_count,
                    accuracy,
                    precision,
                    recall,
                    f1_score,
                    true_negative,
                    false_positive,
                    false_negative,
                    true_positive
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics["sample_count"],
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1_score"],
                matrix["true_negative"],
                matrix["false_positive"],
                matrix["false_negative"],
                matrix["true_positive"],
            ))

            snapshot_id = cursor.lastrowid

        return jsonify({
            "status": "success",
            "message": "Performance snapshot saved.",
            "snapshot_id": snapshot_id,
            "metrics": metrics,
        })

    except Exception:
        app.logger.exception("Could not save performance snapshot")

        return jsonify({
            "status": "error",
            "message": "Could not save the performance snapshot.",
        }), 500


@app.route("/api/model-performance/history", methods=["GET"])
@login_required
def model_performance_history():
    """Return historical performance snapshots."""
    try:
        limit = request.args.get("limit", default=50, type=int)
        limit = max(1, min(limit or 50, 200))

        with get_db_connection() as conn:
            rows = conn.execute("""
                SELECT
                    id,
                    sample_count,
                    accuracy,
                    precision,
                    recall,
                    f1_score,
                    true_negative,
                    false_positive,
                    false_negative,
                    true_positive,
                    created_at
                FROM model_performance_snapshots
                ORDER BY id DESC
                LIMIT ?
            """, (limit,)).fetchall()

        return jsonify({
            "status": "success",
            "history": [dict(row) for row in rows],
        })

    except Exception:
        app.logger.exception("Could not load performance history")

        return jsonify({
            "status": "error",
            "message": "Could not load model performance history.",
        }), 500


# =========================================================
# GEO DATA
# Note: these are demo coordinates, not actual transaction locations.
# =========================================================

@app.route("/api/geo_data", methods=["GET"])
@login_required
def geo_data():
    cities = {
        "Mumbai, India": (19.0760, 72.8777),
        "Delhi, India": (28.6139, 77.2090),
        "Bengaluru, India": (12.9716, 77.5946),
        "New York, USA": (40.7128, -74.0060),
        "London, UK": (51.5074, -0.1278),
        "Singapore": (1.3521, 103.8198),
    }

    with get_db_connection() as conn:
        rows = conn.execute("""
            SELECT id, prediction, risk_score, timestamp
            FROM audit_logs
            ORDER BY id DESC
            LIMIT 30
        """).fetchall()

    city_names = list(cities.keys())
    points = []

    for index, row in enumerate(rows):
        city = city_names[index % len(city_names)]
        lat, lng = cities[city]

        points.append({
            "id": row["id"],
            "prediction": row["prediction"],
            "risk": row["risk_score"],
            "timestamp": row["timestamp"],
            "city": city,
            "lat": lat + ((((index * 17) % 10) - 5) * 0.01),
            "lng": lng + ((((index * 23) % 10) - 5) * 0.01),
        })

    return jsonify(points)


# =========================================================
# SYSTEM SETTINGS API
# =========================================================

@app.route("/api/settings", methods=["GET", "POST"])
@login_required
def handle_settings():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}

        if "fraud_threshold" in data:
            try:
                threshold = float(data["fraud_threshold"])

                if (
                    not np.isfinite(threshold)
                    or not 0 <= threshold <= 100
                ):
                    raise ValueError

                SYSTEM_SETTINGS["fraud_threshold"] = threshold

            except (ValueError, TypeError):
                return jsonify({
                    "status": "error",
                    "message": (
                        "fraud_threshold must be between 0 and 100."
                    ),
                }), 400

        for key in ("email_alerts", "auto_freeze"):
            if key in data:
                if not isinstance(data[key], bool):
                    return jsonify({
                        "status": "error",
                        "message": f"{key} must be true or false.",
                    }), 400

                SYSTEM_SETTINGS[key] = data[key]

        return jsonify({
            "status": "success",
            "message": "System settings updated successfully!",
            "settings": SYSTEM_SETTINGS,
        })

    return jsonify(SYSTEM_SETTINGS)


# =========================================================
# USER PROFILE API
# =========================================================

@app.route("/api/get_profile", methods=["GET"])
@login_required
def get_profile():
    with get_db_connection() as conn:
        row = conn.execute("""
            SELECT username, full_name, bank_name, card_number,
                   city, country, api_key
            FROM users
            WHERE username = ?
        """, (session["user"],)).fetchone()

    if row:
        return jsonify({
            "username": row["username"] or session["user"],
            "full_name": row["full_name"] or "Security Officer",
            "bank_name": row["bank_name"] or "Sentinel Global Bank",
            "card_number": row["card_number"] or "•••• •••• •••• 8892",
            "city": row["city"] or "Uttar Pradesh",
            "country": row["country"] or "India",
            "api_key": row["api_key"] or "No API Key Generated Yet",
        })

    return jsonify({
        "username": session["user"],
        "api_key": "No API Key Generated Yet",
    })


# =========================================================
# API KEY GENERATOR
# =========================================================

@app.route("/api/generate_key", methods=["POST"])
@login_required
def generate_api_key():
    new_key = "swipeshield_live_" + secrets.token_hex(16)

    with get_db_connection() as conn:
        cursor = conn.execute("""
            UPDATE users
            SET api_key = ?
            WHERE username = ?
        """, (new_key, session["user"]))

    if cursor.rowcount == 0:
        return jsonify({
            "status": "error",
            "message": "User profile was not found.",
        }), 404

    return jsonify({
        "status": "success",
        "api_key": new_key,
    })


# =========================================================
# ERROR HANDLERS
# =========================================================

@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({
        "status": "error",
        "message": "Uploaded file exceeds the 10 MB limit.",
    }), 413


# =========================================================
# START APPLICATION
# =========================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SwipeShield Fraud Detection System")
    print("=" * 60)
    print("Model:", "Active" if model is not None else "Unavailable")
    print("NetworkX:", nx.__version__)
    print("SHAP:", "Active" if shap is not None else "Unavailable")
    print(
        "Local RAG:",
        "Active" if get_fraud_intelligence else "Unavailable",
    )
    print("Dataset:", TEST_DATA_PATH.name)
    print("Performance Monitoring: Enabled")
    print("=" * 60)

    app.run(
        debug=os.environ.get("FLASK_DEBUG", "1") == "1",
        host="127.0.0.1",
        port=int(os.environ.get("PORT", "5000")),
    )