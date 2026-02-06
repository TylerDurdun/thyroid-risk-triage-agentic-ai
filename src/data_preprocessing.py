import pandas as pd
import numpy as np

# ---------- Risk Label Creation ----------
def add_risk_labels(df):
    df = df.copy()

    df["risk_score"] = 0.0
    df.loc[df["TSH"] > 4.5, "risk_score"] += 0.4
    df.loc[df["T3"] < 1.0, "risk_score"] += 0.3
    df.loc[(df["TT4"] < 60) | (df["TT4"] > 140), "risk_score"] += 0.3

    df["risk_score"] = df["risk_score"].clip(0, 1)

    def label_risk(score):
        if score >= 0.7:
            return "High Risk"
        elif score >= 0.4:
            return "Medium Risk"
        else:
            return "Low Risk"

    df["risk_level"] = df["risk_score"].apply(label_risk)
    return df


# ---------- Main Preprocessing ----------
def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)

    df = df.replace("?", np.nan)

    binary_cols = [
        'on thyroxine', 'on antithyroid medication', 'sick',
        'pregnant', 'thyroid surgery', 'lithium', 'goitre',
        'tumor', 'hypopituitary', 'psych',
        'TSH measured', 'T3 measured', 'TT4 measured',
        'T4U measured', 'FTI measured'
    ]

    df[binary_cols] = df[binary_cols].replace({'t': 1, 'f': 0})
    df['sex'] = df['sex'].map({'M': 1, 'F': 0})

    lab_cols = ['TSH', 'T3', 'TT4', 'T4U', 'FTI']
    df[lab_cols] = df[lab_cols].astype(float)

    df = add_risk_labels(df)

    return df
