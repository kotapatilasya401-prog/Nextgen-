import re
from difflib import SequenceMatcher

import numpy as np
from sklearn.ensemble import IsolationForest


def _sim(a, b):
    return SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()


def build_features(s, others):
    same_id = sum(1 for o in others if o["aapar_id"] == s["aapar_id"])
    same_phone = sum(1 for o in others if o["phone"] == s["phone"])
    same_email = sum(1 for o in others if o["email"] == s["email"])
    max_sim = max((_sim(o["name"], s["name"]) for o in others
                   if o["institution"] == s["institution"]), default=0.0)
    bad_phone = 0 if re.fullmatch(r"[6-9]\d{9}", s["phone"] or "") else 1
    bad_email = 0 if re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", s["email"] or "") else 1
    return [same_id, same_phone, same_email, max_sim, bad_phone, bad_email]


def predict(student: dict, others: list[dict]) -> dict:
    same_id, same_phone, same_email, max_sim, bad_phone, bad_email = build_features(student, others)
    reasons = []
    risk = 0.0

    if same_id:
        risk += 0.5
        reasons.append(f"AAPAR ID already used by {same_id} record(s)")
    if same_phone:
        risk += 0.25
        reasons.append(f"Phone number shared with {same_phone} record(s)")
    if same_email:
        risk += 0.2
        reasons.append(f"Email shared with {same_email} record(s)")
    if max_sim > 0.9:
        risk += 0.3
        reasons.append(f"Name {int(max_sim * 100)}% similar to another student at the same institution")
    if bad_phone:
        risk += 0.25
        reasons.append("Phone number format looks invalid")
    if bad_email:
        risk += 0.25
        reasons.append("Email format looks invalid")

    anomaly = False
    if len(others) >= 10:
        X = np.array([build_features(o, [x for x in others if x is not o]) for o in others])
        model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42).fit(X)
        anomaly = bool(model.predict([build_features(student, others)])[0] == -1)
        if anomaly:
            risk += 0.15
            reasons.append("Record is statistically unusual compared with existing data")

    return {
        "risk_score": round(min(risk, 1.0), 2),
        "is_duplicate": bool(same_id or same_phone or same_email or max_sim > 0.9),
        "anomaly": anomaly,
        "reasons": reasons,
    }
