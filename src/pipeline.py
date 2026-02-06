def generate_patient_summary(patient_df, risk_level):
    tsh = patient_df["TSH"].values[0]
    t3 = patient_df["T3"].values[0]
    tt4 = patient_df["TT4"].values[0]
    sex = "Male" if patient_df["sex"].values[0] == 1 else "Female"

    summary = []

    summary.append(
        f"The patient is classified as **{risk_level}** based on the current thyroid profile."
    )

    if tsh > 4.5:
        summary.append(
            "TSH levels are elevated, which may indicate altered thyroid regulation."
        )
    else:
        summary.append(
            "TSH levels are within the expected reference range."
        )

    if t3 < 1.0:
        summary.append(
            "T3 levels are below the typical range, suggesting reduced thyroid hormone activity."
        )

    if tt4 < 60 or tt4 > 140:
        summary.append(
            "TT4 values are outside the normal range and warrant clinical review."
        )
    else:
        summary.append(
            "TT4 levels appear to be within the normal range."
        )

    if sex == "Female":
        summary.append(
            "Female patients may require closer monitoring due to hormonal variability."
        )

    summary.append(
        "These observations should be interpreted alongside full clinical history by a qualified clinician."
    )

    return summary


from src.report_generator import build_report


def run_pipeline(model, encoder, patient_df, retriever, feature_cols):
    probs = model.predict_proba(patient_df[feature_cols])[0]
    risk_index = probs.argmax()
    risk_level = encoder.inverse_transform([risk_index])[0]
    confidence = probs[risk_index]

    risk = {
        "risk_level": risk_level,
        "confidence": confidence
    }

    evidence = retriever(f"thyroid risk {risk_level}")
    precautions = retriever(f"general precautions for {risk_level} thyroid risk")

    report = build_report(risk, patient_df)

    return risk, evidence, precautions, report

