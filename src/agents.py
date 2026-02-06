def reasoning_agent(risk_output, evidence):
    return {
        "risk_level": risk_output["risk_level"],
        "confidence": risk_output["confidence"],
        "evidence": evidence,
        "note": "Decision support only. Not a diagnosis."
    }
