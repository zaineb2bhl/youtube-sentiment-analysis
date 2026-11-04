"""Test d'intégration contre une API en cours d'exécution (locale ou Codespaces).

Usage :
    python scripts/check_live_api.py                      # http://127.0.0.1:8000
    python scripts/check_live_api.py https://<codespace>-7860.app.github.dev
"""
import statistics
import sys
import time

import requests

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
SAMPLES = ["I love this video, great explanation!", "This is the worst tutorial ever",
           "What camera do you use?", "😍🔥", "meh"]


def main():
    print(f"API : {BASE}")
    r = requests.get(f"{BASE}/health", timeout=60)
    print(f"/health -> {r.status_code} {r.json()}")
    assert r.ok and r.json()["model_loaded"]

    r = requests.post(f"{BASE}/predict_batch", json={"comments": SAMPLES}, timeout=30)
    assert r.ok, r.text
    for p in r.json()["predictions"]:
        print(f"  {p['sentiment']:8s} {p['confidence']:.2f}  {p['text']!r}")

    for size in (1, 10, 50, 100, 500):
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            r = requests.post(f"{BASE}/predict_batch", json={"comments": SAMPLES * (size // 5 or 1)},
                              timeout=30)
            times.append((time.perf_counter() - t0) * 1000)
            assert r.ok
        print(f"batch={size:4d}  réseau+API médiane={statistics.median(times):7.1f} ms  "
              f"(modèle {r.json()['processing_time_ms']} ms)")

    for payload in ({"comments": []}, {"comments": "x"}, {}):
        code = requests.post(f"{BASE}/predict_batch", json=payload, timeout=30).status_code
        print(f"payload invalide {payload} -> {code}")
        assert code == 422
    print("\nTous les tests d'intégration sont OK")


if __name__ == "__main__":
    main()
