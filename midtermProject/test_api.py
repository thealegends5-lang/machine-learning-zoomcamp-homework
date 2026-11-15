# test_api.py
# Calls the running API on the saved samples, then on a patient with missing fields.
#
#   terminal 1:  uvicorn app:app
#   terminal 2:  python test_api.py

import json
import urllib.request

BASE = "http://localhost:8000"


def call(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


print("health:", call("/health")["status"])

for s in call("/samples"):
    r = call("/predict", s["features"])
    status = "OK " if r["prediction"] == s["true_label"] else "ERR"
    print(f"[{status}] {s['source']:5s} true={s['true_label']} "
          f"pred={r['prediction']} p={r['probability_diabetic']} -> {r['label']}")

print("\nPatient with missing fields:")
r = call("/predict", {"age": 58, "weight": 210, "height": 67})
print(json.dumps(r, indent=2))
