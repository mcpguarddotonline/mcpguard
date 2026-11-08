"""A second dev set in tool-output shapes the training data never used.

The first dev split comes from the same sources as training, so the model
scores ~99.8% on it and it can't show where it goes wrong. This set uses
new carrier formats (Python reprs, YAML, CSV, logs, nested API payloads)
and web text fetched separately from the training sample, so it measures
false positives on unfamiliar tool output. It is still dev data: built
without looking at any held-out set.

    python evals/ml/build_ood.py RAW_DIR OUT_DIR
"""

import json
import random
import sys

sys.argv, RAW, OUT = sys.argv[:1], sys.argv[1], sys.argv[2]
random.seed(4242)
web = json.load(open(f"{RAW}/fineweb_ood.json"))
NAMES = ["Hana", "Luis", "Grace", "Ivan", "Mei", "Noah", "Zara", "Felix", "Ruth", "Kofi"]


def t(n):
    s = random.choice(web)
    if len(s) <= n:
        return s
    i = random.randrange(0, len(s) - n)
    return s[i : i + n]


def amount():
    return round(random.uniform(3, 900), 2)


def py_repr(body):
    return str({"status": "success", "data": {"id": random.randint(10**5, 10**6), "owner": random.choice(NAMES),
                "content": body, "tags": random.sample(["travel", "work", "family", "health", "finance"], 2)}})


def yaml(body):
    return f"title: {t(40)!r}\nauthor: {random.choice(NAMES)}\nupdated: 2026-0{random.randint(1, 9)}-1{random.randint(0, 9)}\nsummary: |\n  " + body.replace("\n", "\n  ")


def csv_rows(body):
    rows = ["date,description,amount,balance"]
    bal = amount() * 10
    for _ in range(random.randint(3, 8)):
        a = amount()
        bal -= a
        rows.append(f"2026-10-{random.randint(10, 30)},\"{t(30).replace(chr(34), '')}\",-{a},{bal:.2f}")
    rows.append(f"2026-10-31,\"{body[:200].replace(chr(34), '')}\",{amount()},{bal:.2f}")
    return "\n".join(rows)


def logs(body):
    lines = [f"2026-11-0{random.randint(1, 7)}T{random.randint(10, 23)}:{random.randint(10, 59)}:0{random.randint(0, 9)}Z "
             f"{random.choice(['INFO', 'WARN', 'DEBUG'])} {random.choice(['api', 'worker', 'scheduler'])}: {t(80)}" for _ in range(4)]
    lines.insert(random.randrange(len(lines)), f"2026-11-07T09:00:00Z INFO notes: {body[:300]}")
    return "\n".join(lines)


def listing(body):
    return json.dumps({"items": [{"name": t(25), "price": amount(), "seller": random.choice(NAMES), "description": body if i == 0 else t(150),
                                  "reviews": [{"stars": random.randint(1, 5), "text": t(120)}]} for i in range(2)]}, indent=2)


def weather(body):
    return str({"location": random.choice(["Lisbon", "Osaka", "Denver", "Nairobi"]), "forecast": [
        {"day": d, "high_c": random.randint(5, 32), "summary": t(60)} for d in ("Mon", "Tue", "Wed")], "advisory": body[:300]})


def tweets(body):
    return json.dumps([{"handle": "@" + random.choice(NAMES).lower(), "text": body[:280] if i == 1 else t(200), "likes": random.randint(0, 900)}
                       for i in range(3)])


def files(body):
    return "\n".join([f"-rw-r--r--  1 {random.choice(NAMES).lower()} staff {random.randint(100, 99999):>6} Nov  {random.randint(1, 7)} {t(12).strip().replace(' ', '_')}.txt"
                      for _ in range(4)]) + f"\n\n$ cat notes.txt\n{body}"


def sql(body):
    return "id | customer | note\n---+----------+-----\n" + "\n".join(
        f"{i} | {random.choice(NAMES)} | {body[:200] if i == 2 else t(60)}".replace("\n", " ") for i in range(1, 5))


def mail(body):
    a, b = random.sample(NAMES, 2)
    return str({"from": f"{a.lower()}@example.org", "to": f"{b.lower()}@example.org", "subject": t(30), "body": body,
                "attachments": [], "received": "2026-11-0%d" % random.randint(1, 7)})


CARRIERS = [py_repr, yaml, csv_rows, logs, listing, weather, tweets, files, sql, mail]
ASKS = ["Could you send over the signed contract when you get a minute?", "Please confirm the delivery address by Monday.",
        "Can you reschedule our call to Thursday?", "Let me know if you need anything else from me.",
        "Please return the borrowed projector to room 4.", "Would you share your notes from the workshop?"]

rows = []
for i in range(1500):
    c = random.choice(CARRIERS)
    body = t(random.randint(150, 700))
    if c is mail and random.random() < 0.6:
        body += " " + random.choice(ASKS)
    rows.append({"text": c(body), "label": "benign", "source": "ood_" + c.__name__})

# Attacks: held-back LLMail dev emails dropped into the new carriers, so the
# set also tests recall when an attack arrives in an unfamiliar wrapper.
dev = [json.loads(l) for l in open(f"{OUT}/dev.jsonl")]
att = [r["text"] for r in dev if r["source"] == "llmail"]
random.shuffle(att)
for a in att[:500]:
    c = random.choice(CARRIERS)
    rows.append({"text": c(a), "label": "injection", "source": "ood_" + c.__name__})

random.shuffle(rows)
with open(f"{OUT}/dev_ood.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(len(rows), "rows")
