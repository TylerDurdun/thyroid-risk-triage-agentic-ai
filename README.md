# 🧠 Thyroid Risk Triage – Agentic AI

**Decision Support System (Not a Medical Diagnosis)**

---

## 📌 Problem Statement

Early identification of thyroid-related risk is critical for timely clinical review.
However, clinicians often face inconsistent triage, fragmented data, and limited time to interpret laboratory results alongside medical guidelines.

This project implements an **Agentic AI system** that supports thyroid risk triage by combining:

- Predictive machine learning
- Evidence-based medical knowledge retrieval
- Transparent, explainable reasoning

while strictly maintaining **clinical safety boundaries**.

---

## 🎯 Objective

To build an **Agentic AI decision support system** that:

- Estimates thyroid risk with confidence
- Grounds decisions in medical evidence
- Explains *why* a risk level was assigned
- Clearly distinguishes decision support from diagnosis
- Is reproducible, auditable, and safe for clinical settings

---

## 🧩 System Architecture (Agentic Design)

This is **not a single model**, but a coordinated system of agents, each with a bounded responsibility.

### 1️⃣ Risk Scoring Agent (ML Agent)

- Uses **Logistic Regression** with class-balanced learning
- Handles missing values via imputation
- Outputs:
  - Risk level (Low / Medium / High)
  - Confidence score
- Focused solely on risk estimation, not explanation

### 2️⃣ Medical Knowledge Agent (RAG)

- Retrieves relevant medical context from a controlled knowledge base
- Uses **TF-IDF–based retrieval** for transparency and auditability
- Prevents hallucination by restricting output to known sources
- Provides **supporting evidence**, not conclusions

### 3️⃣ Reasoning Layer

- Connects model output with retrieved evidence
- Explains *why* a risk level was assigned
- Explicitly communicates uncertainty
- Avoids diagnostic or treatment language

### 4️⃣ Orchestration Layer

- Coordinates agent execution
- Enforces safety constraints
- Ensures consistent training–inference feature alignment

---

## 🧠 Why This Is an Agentic System

- Each agent has a narrow, well-defined role
- No agent has full autonomy
- No single component makes medical claims
- Decisions emerge from collaboration, not monolithic logic

This design prevents unsafe behavior and mirrors real clinical decision-support workflows.

---

## 📊 Machine Learning Details

- **Model:** Logistic Regression
- **Task:** Multiclass risk classification
- **Class imbalance:** Handled using `class_weight="balanced"`
- **Metrics:** Precision, Recall, F1-score
- **Output:** Risk level + confidence score

### Data Handling

- Raw data used (no pre-cleaned CSVs)
- All preprocessing encoded in code for reproducibility

---

## 📚 Retrieval-Augmented Knowledge (RAG)

- Knowledge base stored as plain text medical guidance
- Deterministic retrieval (TF-IDF)
- Fully auditable and reproducible
- No external APIs required
- Ensures explanations are evidence-grounded

---

## 🖥️ User Interface (Streamlit)

A lightweight Streamlit UI allows:

- Selection of a patient record
- Execution of the full agentic pipeline
- Display of:
  - Risk level
  - Confidence
  - Supporting medical evidence

Clear safety disclaimers are displayed at all times.

The UI is designed for **demonstration and interpretability**, not clinical deployment.

---

## ⚠️ Safety, Ethics, and Limitations

This system explicitly enforces the following principles:

- 🚫 Not a medical diagnosis
- 🚫 No treatment recommendations
- ✅ Decision support only
- ✅ Requires clinician review
- ✅ Explicit confidence and uncertainty
- ✅ No hallucinated medical facts
- ✅ Reproducible preprocessing and inference

### Disclaimer

> This system provides decision support only and must be used by qualified clinicians.
> Outputs may be incomplete or incorrect if inputs are missing or inaccurate.
>


