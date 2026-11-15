"""Calls the RUNNING API on the saved samples (2 test + 2 train), then on a
patient with missing fields to show the warning.

    terminal 1:  uvicorn app:app
    terminal 2:  python test_api.py
"""
import json
import urllib.request

BASE = "http://localhost:8000"


def call(path, body=None):
    req = urllib.request.Request(
        BASE + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


print("health:", call("/health")["status"])

for s in call("/samples"):
    r = call("/predict", s["features"])
    ok = "OK " if r["prediction"] == s["true_label"] else "ERR"
    print(f"[{ok}] {s['source']:5s} true={s['true_label']} pred={r['prediction']} "
          f"p={r['probability_diabetic']} -> {r['label']}")

print("\nPatient with missing fields:")
r = call("/predict", {"age": 58, "weight": 210, "height": 67})
print(json.dumps(r, indent=2))
