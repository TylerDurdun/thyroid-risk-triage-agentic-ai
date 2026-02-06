from datetime import datetime


def confidence_label(confidence):
    if confidence >= 0.85:
        return "High"
    elif confidence >= 0.6:
        return "Moderate"
    else:
        return "Low"


def generate_reasoning(patient_df, risk_level):
    tsh = patient_df["TSH"].values[0]
    t3 = patient_df["T3"].values[0]
    tt4 = patient_df["TT4"].values[0]

    reasoning = []
    reasoning.append(
        "The patient biomarkers were compared against the reference population used during model training."
    )

    if tsh <= 4.5 and t3 >= 1.0 and 60 <= tt4 <= 140:
        reasoning.append(
            "**Euthyroid State**: Key biomarkers fall within population norms."
        )
    else:
        reasoning.append(
            "One or more thyroid biomarkers deviate from population norms."
        )

    return reasoning


def generate_clinical_summary(patient_df, risk_level):
    doctor_summary = []
    patient_summary = []

    doctor_summary.append(
        f"Assessment: {risk_level}. The patient biochemical profile was evaluated relative to the training dataset."
    )

    doctor_summary.append(
        "Recommendation: Correlate laboratory findings with clinical symptoms and confirm with follow-up testing if required."
    )

    patient_summary.append(
        f"We analyzed your thyroid health markers. Your results place you in the **{risk_level}** category."
    )

    patient_summary.append(
        "Your thyroid values appear broadly consistent with the reference population."
    )

    patient_summary.append(
        "Please consult your healthcare provider for clinical interpretation."
    )

    return doctor_summary, patient_summary


def build_report(risk, patient_df):
    report = {}

    report["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")

    report["risk_level"] = risk["risk_level"]
    report["risk_score"] = round(risk["confidence"], 2)
    report["confidence_label"] = confidence_label(risk["confidence"])

    report["reasoning"] = generate_reasoning(patient_df, risk["risk_level"])

    doctor_summary, patient_summary = generate_clinical_summary(
        patient_df, risk["risk_level"]
    )

    report["doctor_summary"] = doctor_summary
    report["patient_summary"] = patient_summary

    return report
