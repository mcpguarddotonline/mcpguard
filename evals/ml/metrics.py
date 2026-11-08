"""Recall and false-positive rate from score files written by score.py.

    python evals/ml/metrics.py THRESHOLD NAME=scores.json [NAME=scores.json ...]

THRESHOLD is fixed in advance (chosen on dev data, or the vendor's default),
never tuned on the held-out sets.
"""

import json
import sys


def main():
    thr = float(sys.argv[1])
    print(f"{'detector':24} {'set':12} {'attacks':>8} {'benign':>7} {'recall':>8} {'FPR':>7}")
    for arg in sys.argv[2:]:
        name, path = arg.split("=", 1)
        for setname, rows in json.load(open(path)).items():
            att = [s for l, s in rows if l == "injection"]
            ben = [s for l, s in rows if l != "injection"]
            rec = sum(s >= thr for s in att) / len(att) if att else float("nan")
            fpr = sum(s >= thr for s in ben) / len(ben) if ben else float("nan")
            print(f"{name:24} {setname:12} {len(att):8} {len(ben):7} {rec:8.1%} {fpr:7.1%}")


if __name__ == "__main__":
    main()
