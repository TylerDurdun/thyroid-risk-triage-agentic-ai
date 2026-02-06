from src.data_preprocessing import load_and_preprocess_data
from src.risk_model import train_risk_model
from src.rag import load_knowledge, build_retriever, retrieve_evidence
from src.pipeline import run_pipeline

df = load_and_preprocess_data("data/Thyroid_Data.csv")

feature_cols = [
    'age', 'sex',
    'on thyroxine', 'on antithyroid medication', 'sick',
    'pregnant', 'thyroid surgery', 'lithium',
    'goitre', 'tumor', 'hypopituitary', 'psych',
    'TSH', 'T3', 'TT4', 'T4U', 'FTI'
]

model, encoder = train_risk_model(df, feature_cols)

knowledge = load_knowledge("knowledge/thyroid_knowledge.txt")
vectorizer, vectors = build_retriever(knowledge)


def retriever(q):
    return retrieve_evidence(q, vectorizer, vectors, knowledge)

# --------- Single Patient Demo ---------

sample_patient = df.iloc[[0]]  # keep as DataFrame

risk, evidence = run_pipeline(
    model,
    encoder,
    sample_patient,
    retriever,
    feature_cols
)


print("\n=== RISK OUTPUT ===")
print(risk)

print("\n=== SUPPORTING MEDICAL EVIDENCE ===")
for e in evidence:
    print("-", e)

