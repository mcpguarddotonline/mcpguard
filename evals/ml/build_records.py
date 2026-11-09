"""Harmless and attacked structured records, the shapes agent tools return.

    python evals/ml/build_records.py RAW_DIR DATA_DIR OUT_DIR

Rounds 1 and 2 flag most harmless tool output that comes back as YAML or as
lists of records (89% of harmless YAML lists on dev_agent, round 1). This
script teaches the model that a record's structure is not a sign of attack:
only what is inside a field matters.

Reads RAW_DIR/{fineweb,wiki,wiki_multi}.json and DATA_DIR/train.jsonl.
Writes OUT_DIR/train.jsonl (the existing rows plus the new training records),
OUT_DIR/records_dev.jsonl (the records held back) and unchanged copies of
DATA_DIR/{dev,dev_ood,dev_agent}.jsonl.

* Record types: message, document, calendar, chat, contact, order, listing,
  ticket, search, api_status. Field names vary from row to row. Short fields
  are made up (example.com addresses, fictional phone ranges, fake IBANs);
  long fields come from fineweb-edu / Wikipedia / multilingual Wikipedia.
  AgentDojo (dev_agent's source) is never read for field names or values.
* Shapes: one record (~35%) or a list of 2-5, sometimes wrapped as
  {"results": [...], "count": n}; JSON (compact or indented) or YAML (block
  style, long text quoted or as | block scalars); some records nest lists.
* Attacks: one long field in one random record holds a text taken verbatim
  from a row already labelled injection in DATA_DIR/train.jsonl, from the
  public datasets only (not our synthetic templates), spread evenly across
  them. Appended to the field's harmless text or replacing it. We write no
  attack text.
* Truncation: train.py reads the first 384 tokens. Every row (benign too, so
  length is not a cue) keeps its target field's payload within an
  estimated ~270 tokens of the start, so the attack is not cut off.
* Split: ticket, api_status and calendar-as-YAML only in records_dev, plus a
  random share of the other rows, about 20% in all. Source documents and
  attack texts are split too, so no text is in both.
* Fails if any dev / dev_ood / dev_agent / held-out text appears in the
  output, if an attack text is not a train injection row, or if counts are
  outside the expected range. Deterministic: fixed seed, no hash() ordering.
"""

import bisect
import collections
import glob
import json
import math
import os
import random
import re
import shutil
import sys
import zlib

RAW, DATA, OUT = sys.argv[1:4]
HERE = os.path.dirname(os.path.abspath(__file__))
HELDOUT = os.path.join(HERE, "..", "public", "heldout")
SEED = 20261110
rng = random.Random(SEED)

N_BENIGN, N_ATTACK = 10_000, 4_000
DEV_SHARE = 0.20
HELD_COMBOS = {("ticket", "json"), ("ticket", "yaml"), ("api_status", "json"), ("api_status", "yaml"), ("calendar", "yaml")}
TYPE_WEIGHT = {"message": 1, "document": 1, "calendar": 1, "chat": 1, "contact": 1, "order": 1,
               "listing": 1, "search": 1, "ticket": 0.4, "api_status": 0.4}
ATTACK_SOURCES = ["deepset", "gandalf", "jackhhao", "llmail", "llmail2", "spml", "yanismiraoui"]
MAX_ATTACK_REUSE = 2
TOKEN_BUDGET = 270  # estimated tokens before the payload starts (train.py truncates at 384)


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def stable(s):
    return zlib.crc32(s.encode("utf-8"))


# ------------------------------------------------------------- text cleaning
BAD = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ud800-\udfff﻿￾￿]")


def clean(t):
    """Normalise whitespace and drop characters YAML can't carry. Applied to
    every long text, benign or attack, before it goes into a record."""
    t = t.replace("\r\n", "\n").replace("\r", "\n").replace(" ", "\n").replace(" ", "\n")
    t = BAD.sub(" ", t)
    lines = [re.sub(r"[ \t]+", " ", l).strip() for l in t.split("\n")]
    t = "\n".join(lines)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def norm(t):
    return re.sub(r"\s+", " ", t.lower()).strip()


# --------------------------------------------------------- leakage machinery
W, STRIDE = 50, 8
eval_rows = {}
for s in ("dev", "dev_ood", "dev_agent"):
    eval_rows[s] = jl(f"{DATA}/{s}.jsonl")
for p in sorted(glob.glob(f"{HELDOUT}/*.jsonl")):
    eval_rows["heldout/" + os.path.basename(p)] = jl(p)
eval_texts = [r["text"] for rows in eval_rows.values() for r in rows]
eval_norm = {norm(t) for t in eval_texts}
shingles = set()
for t in eval_texts:
    n = norm(t)
    for i in range(0, max(1, len(n) - W + 1), STRIDE):
        if len(n) >= W:
            shingles.add(n[i : i + W])


# Round 3's records_dev, when DATA_DIR is round 3's output (round 4). Kept
# apart: later rounds' training rows must avoid it, but it was built from the
# same dev documents a later round's dev rows draw on.
rd_texts = [r["text"] for r in jl(f"{DATA}/records_dev.jsonl")] if os.path.exists(f"{DATA}/records_dev.jsonl") else []
rd_norm = {norm(t) for t in rd_texts}
rd_shingles = {norm(t)[i : i + W] for t in rd_texts for i in range(0, max(1, len(norm(t)) - W + 1), STRIDE)
               if len(norm(t)) >= W}


def overlaps_eval(t, records_dev=False):
    """True if t shares a substring of >= W+STRIDE-1 normalised characters with
    any eval text (or equals one). records_dev=True also checks round 3's
    records_dev."""
    n = norm(t)
    if n in eval_norm or (records_dev and n in rd_norm):
        return True
    return any(n[i : i + W] in shingles or (records_dev and n[i : i + W] in rd_shingles)
               for i in range(0, len(n) - W + 1))


# ------------------------------------------------------------ benign text pools
fineweb = json.load(open(f"{RAW}/fineweb.json"))
wiki = json.load(open(f"{RAW}/wiki.json"))
multi = json.load(open(f"{RAW}/wiki_multi.json"))
pools = {"train": {"en": [], "multi": []}, "dev": {"en": [], "multi": []}}
dropped_docs = 0
for name, docs in (("fineweb", fineweb), ("wiki", wiki)) + tuple((f"wiki_{k}", v) for k, v in multi.items()):
    for i, d in enumerate(docs):
        d = clean(d)
        if len(d) < 200:
            continue
        if overlaps_eval(d):
            dropped_docs += 1
            continue
        split = "dev" if stable(f"{name}:{i}") % 100 < 20 else "train"
        pools[split]["multi" if name.startswith("wiki_") else "en"].append(d)
print(f"benign docs: train {len(pools['train']['en'])}+{len(pools['train']['multi'])} multi, "
      f"dev {len(pools['dev']['en'])}+{len(pools['dev']['multi'])} multi, dropped for eval overlap {dropped_docs}")


def cut(doc, n):
    """About n characters from doc, starting near a sentence and ending on a word."""
    if len(doc) <= n:
        return doc
    start = rng.randrange(0, len(doc) - n)
    j = doc.find(". ", start, start + 200)
    if j != -1 and j + 2 + n <= len(doc):
        start = j + 2
    t = doc[start : start + n]
    if " " in t[int(n * 0.7) :]:
        t = t[: t.rfind(" ")]
    return t.strip()


LANG_DRAWN = collections.Counter()


def harmless(split, lo=40, hi=900):
    lang = "multi" if rng.random() < 1 / 9 else "en"
    LANG_DRAWN[lang] += 1
    n = int(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    while True:
        t = cut(rng.choice(pools[split][lang]), n)
        if len(t) >= min(lo, 40):
            return t


# ----------------------------------------------------------------- attack pool
train_rows = jl(f"{DATA}/train.jsonl")
train_inj = {r["text"] for r in train_rows if r["label"] == "injection"}
by_src = collections.defaultdict(list)
skipped_attacks = 0
seen = set()
for r in train_rows:
    if r["label"] != "injection" or r["source"] not in ATTACK_SOURCES:
        continue
    c = clean(r["text"])
    if len(c) < 20 or c in seen:
        continue
    seen.add(c)
    by_src[r["source"]].append(r["text"])
# Attack texts are already train rows; drop any that equal an eval text or
# share a 200-character run with one (either direction), as add_round2.py
# does. Many attacks share template phrases with eval attacks, so the
# stricter shingle test used for benign text would discard most of them.
eval_blob = "\n".join(norm(t) for t in eval_texts)
cands = [(s, t) for s in ATTACK_SOURCES for t in by_src[s]]
cn = [norm(t) for _, t in cands]
starts, pos = [], 0
for k in cn:
    starts.append(pos)
    pos += len(k) + 1
cblob = "\n".join(cn)
drop = set()
for i, k in enumerate(cn):
    if k in eval_norm or (len(k) >= 200 and k[:200] in eval_blob):
        drop.add(i)
for t in eval_texts:
    k = norm(t)
    if len(k) < 200:
        continue
    j = cblob.find(k[:200])
    while j != -1:
        drop.add(bisect.bisect_right(starts, j) - 1)
        j = cblob.find(k[:200], j + 1)
skipped_attacks = len(drop)
by_src = collections.defaultdict(list)
for i, (s, t) in enumerate(cands):
    if i not in drop:
        by_src[s].append(t)
attack_pool = {"train": {}, "dev": {}}
for s in ATTACK_SOURCES:
    texts = sorted(by_src[s])
    for t in texts:
        sp = "dev" if stable(t) % 100 < 20 else "train"
        attack_pool[sp].setdefault(s, []).append(t)
for sp in attack_pool:
    for s in attack_pool[sp]:
        rng.shuffle(attack_pool[sp][s])
print("attack texts (unique, usable):", {s: len(by_src[s]) for s in ATTACK_SOURCES},
      "skipped for eval overlap:", skipped_attacks)


def attack_schedule(split, n):
    """n attack texts for a split, as even across sources as the reuse cap allows."""
    avail = {s: attack_pool[split].get(s, []) for s in ATTACK_SOURCES}
    quota = {s: 0 for s in ATTACK_SOURCES}
    left, open_ = n, [s for s in ATTACK_SOURCES if avail[s]]
    while left and open_:
        share = max(1, left // len(open_))
        nxt = []
        for s in open_:
            cap = len(avail[s]) * MAX_ATTACK_REUSE - quota[s]
            take = min(share, cap, left)
            quota[s] += take
            left -= take
            if quota[s] < len(avail[s]) * MAX_ATTACK_REUSE:
                nxt.append(s)
        open_ = nxt
    out = []
    for s in ATTACK_SOURCES:
        for k in range(quota[s]):
            out.append((s, avail[s][k % len(avail[s])]))
    rng.shuffle(out)
    return out


# ------------------------------------------------------------ short-field fakes
FIRST = ["Ava", "Liam", "Noor", "Mateo", "Hana", "Kofi", "Ingrid", "Ravi", "Sofia", "Tariq", "Mei", "Lucas",
         "Amara", "Jonas", "Yuki", "Elena", "Dmitri", "Fatima", "Oscar", "Zainab", "Theo", "Leila", "Bruno",
         "Chiara", "Kwame", "Anya", "Felix", "Rosa", "Arjun", "Greta", "Samir", "Ines", "Hugo", "Malia", "Pavel",
         "Lucía", "Søren", "Zoë", "Björn", "Chloé", "Emeka", "Aditi", "Tomás", "Wren", "Idris", "Maren"]
LAST = ["Okafor", "Lindqvist", "Haddad", "Moreau", "Tanaka", "Silva", "Novak", "Brennan", "Kaur", "Rossi",
        "Mensah", "Fischer", "Castillo", "Ivanova", "Nakamura", "Osei", "Park", "Duarte", "Hartley", "Varga",
        "Petrov", "Abara", "Quinn", "Zhou", "Lambert", "Eriksen", "Romero", "Kowalski", "Achebe", "Sato",
        "Müller", "Núñez", "Dubois", "Kim", "Shah", "Wolfe", "Gallo", "Byrne", "Aziz", "Holm"]
COMPANIES = ["Larkspur Analytics", "Bramblewood Foods", "Copperline Freight", "Halcyon Health", "Ridgeway Legal",
             "Tidewater Energy", "Oakhurst Media", "Quillstone Press", "Silverpine Labs", "Marrow & Finch",
             "Northgate Retail", "Pemberton Civil", "Juniper Robotics", "Saltmarsh Studio", "Everfield Bank"]
DOMAINS = ["example.com", "example.org", "example.net", "mail.example.com", "corp.example", "example.co.uk"]
CITIES = ["Lisbon", "Osaka", "Toronto", "Nairobi", "Kraków", "Melbourne", "Bogotá", "Edinburgh", "Hanoi",
          "Valencia", "Reykjavík", "Austin", "Porto", "Seoul", "Cape Town", "Ljubljana", "Montréal", "Tbilisi",
          "Wellington", "Bergen", "Marrakesh", "Leipzig", "Oaxaca", "Busan", "Galway"]
STREETS = ["Harbour Road", "Elm Street", "Rua das Flores", "Station Lane", "Kingsway", "Mill Street",
           "Lakeview Drive", "Calle Mayor", "Bahnhofstraße", "Chestnut Avenue", "Quay Street"]
CHANNELS = ["general", "random", "eng-backend", "design", "ops-alerts", "marketing", "support", "travel",
            "book-club", "release-planning", "data-team", "office-london"]
STATUSES_ORDER = ["completed", "pending", "refunded", "shipped", "cancelled", "processing", "settled"]
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CAD", "CHF", "AUD", "SEK"]
LABELS = ["bug", "enhancement", "docs", "question", "p1", "p2", "backend", "ui", "needs-triage", "good first issue"]
EMOJI = ["thumbsup", "tada", "eyes", "heart", "rocket", "white_check_mark", "joy", "pray"]
SUBJECTS = ["Quarterly numbers", "Lunch on Thursday?", "Notes from today's call", "Invoice {n}", "Updated draft",
            "Trip itinerary", "Re: budget review", "Fwd: reading list", "Board meeting minutes", "Welcome aboard",
            "Your receipt", "Weekly digest", "Question about the lease", "Photos from the weekend",
            "Re: Re: project timeline", "Course materials", "Volunteer schedule", "Shipping update"]
EVENT_TITLES = ["Team sync", "Dentist", "Design review", "1:1", "Quarterly planning", "Flight to {city}",
                "Yoga class", "Interview loop", "Book club", "Sprint retro", "Parent-teacher meeting",
                "Product demo", "Dinner with {first}", "Offsite day 1", "Budget workshop"]
PLACE_ADJ = ["Blue", "Old", "Golden", "Little", "Quiet", "Corner", "Riverside", "Green", "Copper", "Lantern"]
PLACE_NOUN = ["Fig", "Anchor", "Kettle", "Orchard", "Harbor", "Willow", "Pantry", "Courtyard", "Mill", "Lark"]
PRODUCTS = ["wireless earbuds", "standing desk", "cast-iron skillet", "e-reader", "rain jacket", "desk lamp",
            "travel adapter", "yoga mat", "coffee grinder", "backpack", "monitor arm", "water bottle"]
SERVICES = ["auth-gateway", "billing", "search-indexer", "media-transcoder", "notifications", "ledger",
            "geo-lookup", "report-builder", "edge-cache", "scheduler"]
REGIONS = ["us-east-1", "eu-west-2", "ap-northeast-1", "eu-central-1", "us-west-2", "sa-east-1"]
FILE_EXT = [("pdf", "application/pdf"), ("docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            ("txt", "text/plain"), ("md", "text/markdown"), ("xlsx", "application/vnd.ms-excel"), ("csv", "text/csv")]
FILE_STEM = ["meeting-notes", "budget_2026", "draft", "proposal", "reading-list", "itinerary", "research_summary",
             "handbook", "minutes", "report-q3", "lecture-notes", "recipe_collection"]


def pick(*xs):
    return rng.choice(xs)


def person():
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def email(name=None):
    name = name or person()
    a, b = name.split(" ", 1)
    local = pick(f"{a}.{b}", f"{a[0]}{b}", f"{a}", f"{a}_{b}{rng.randrange(10, 99)}")
    local = re.sub(r"[^a-z0-9._]", "", local.lower().translate(str.maketrans("áéíóúñøöüëçå", "aeiounoouaca")))
    return f"{local or 'user'}@{rng.choice(DOMAINS)}"


def phone():
    return pick(f"+1 555-01{rng.randrange(10, 99)}", f"(555) 01{rng.randrange(10, 99)}-{rng.randrange(1000, 9999)}",
                f"+44 20 7946 0{rng.randrange(100, 999)}", f"+61 491 570 {rng.randrange(100, 999)}",
                f"+49 30 {rng.randrange(1000, 9999)} {rng.randrange(1000, 9999)}",
                f"+33 1 {rng.randrange(10, 99)} {rng.randrange(10, 99)} {rng.randrange(10, 99)} {rng.randrange(10, 99)}")


def iban():
    cc = rng.choice(["GB", "DE", "FR", "NL", "ES", "IE"])
    digits = "".join(str(rng.randrange(10)) for _ in range(18))
    s = f"{cc}{rng.randrange(10, 99)}{digits}"
    return pick(s, " ".join(s[i : i + 4] for i in range(0, len(s), 4)))


def account():
    return pick(iban(), f"****{rng.randrange(1000, 9999)}", f"acct-{rng.randrange(10**7, 10**8)}",
                f"{rng.randrange(10**8, 10**9)}")


def ts(fmt=None):
    y, mo, d = 2026, rng.randrange(1, 13), rng.randrange(1, 29)
    h, mi, s = rng.randrange(7, 22), rng.choice([0, 15, 30, 45, rng.randrange(60)]), rng.randrange(60)
    fmt = fmt or rng.choice(["iso", "isoz", "space", "date"])
    if fmt == "iso":
        return f"{y}-{mo:02d}-{d:02d}T{h:02d}:{mi:02d}:{s:02d}"
    if fmt == "isoz":
        return f"{y}-{mo:02d}-{d:02d}T{h:02d}:{mi:02d}:{s:02d}Z"
    if fmt == "space":
        return f"{y}-{mo:02d}-{d:02d} {h:02d}:{mi:02d}"
    return f"{y}-{mo:02d}-{d:02d}"


def ident(prefix=None):
    return pick(str(rng.randrange(1, 99999)), f"{prefix or 'id'}_{rng.randrange(16**8):08x}",
                f"{rng.randrange(16**12):012x}", f"{(prefix or 'x').upper()}-{rng.randrange(100, 9999)}")


def money():
    return round(rng.choice([rng.uniform(2, 80), rng.uniform(80, 900), rng.uniform(900, 9000)]), 2)


SCHEMA = {}


def var(*names):
    """One field name from its variants. Chosen once per row, so the records
    in one list share a schema the way a real tool's output does."""
    if names not in SCHEMA:
        SCHEMA[names] = rng.choice(names)
    return SCHEMA[names]


def short(split, lo=20, hi=90):
    t = harmless(split, lo, hi).split("\n")[0]
    return t[:hi]


# ------------------------------------------------------------------ record types
# Each returns (record, slots); a slot is a path inside the record to a long
# text field that may carry the attack.

def r_message(sp):
    sender = person()
    r = {var("id", "uid", "message_id", "msg_id"): ident("msg"),
         var("from", "sender", "from_address"): pick(email(sender), f"{sender} <{email(sender)}>"),
         var("to", "recipients"): [email() for _ in range(rng.randrange(1, 4))]}
    if rng.random() < 0.3:
        r["cc"] = [email() for _ in range(rng.randrange(1, 3))]
    r[var("subject", "title")] = rng.choice(SUBJECTS).format(n=rng.randrange(1000, 9999))
    r[var("date", "sent_at", "timestamp", "received")] = ts()
    k = var("read", "is_read", "unread", "starred")
    r[k] = rng.random() < 0.5
    body = var("body", "text", "content", "body_text")
    r[body] = harmless(sp)
    slots = [(body,)]
    if rng.random() < 0.25:
        r[var("attachments", "files")] = [{"filename": f"{rng.choice(FILE_STEM)}.{e}", "size": rng.randrange(2_000, 4_000_000)}
                                          for e, _ in rng.sample(FILE_EXT, rng.randrange(1, 3))]
    return r, slots


def r_document(sp):
    ext, mime = rng.choice(FILE_EXT)
    r = {var("id", "file_id", "doc_id"): ident("file"),
         var("name", "filename", "title"): f"{rng.choice(FILE_STEM)}.{ext}",
         var("owner", "owner_email", "created_by"): email(),
         var("size", "size_bytes", "bytes"): rng.randrange(400, 6_000_000),
         var("modified", "last_modified", "updated_at"): ts()}
    if rng.random() < 0.4:
        r[var("mime_type", "type", "content_type")] = mime
    if rng.random() < 0.35:
        r[var("shared_with", "collaborators")] = [email() for _ in range(rng.randrange(1, 4))]
    k = var("content", "preview", "text", "excerpt")
    r[k] = harmless(sp)
    return r, [(k,)]


def r_calendar(sp):
    start = ts("iso")
    h = int(start[11:13]) + rng.randrange(1, 3)
    end = f"{start[:11]}{min(h, 23):02d}{start[13:]}"
    r = {var("id", "event_id", "uid"): ident("evt"),
         var("title", "summary", "name"): rng.choice(EVENT_TITLES).format(city=rng.choice(CITIES), first=rng.choice(FIRST)),
         var(("start", "end"), ("start_time", "end_time"), ("starts_at", "ends_at"))[0]: start,
         var(("start", "end"), ("start_time", "end_time"), ("starts_at", "ends_at"))[1]: end,
         var("location", "where", "venue"): pick(f"{rng.randrange(1, 200)} {rng.choice(STREETS)}, {rng.choice(CITIES)}",
                                                 f"Room {rng.randrange(1, 12)}{rng.choice('ABC')}", "Video call", rng.choice(CITIES)),
         var("attendees", "participants", "guests"): [email() for _ in range(rng.randrange(1, 6))]}
    if rng.random() < 0.4:
        r["organizer"] = email()
    if rng.random() < 0.3:
        r[var("all_day", "recurring")] = rng.random() < 0.3
    k = var("description", "notes", "details", "agenda")
    r[k] = harmless(sp)
    return r, [(k,)]


def r_chat(sp):
    r = {var("channel", "channel_name", "room"): pick(rng.choice(CHANNELS), "#" + rng.choice(CHANNELS)),
         var("user", "author", "sender", "username"): pick(person(), rng.choice(FIRST).lower() + str(rng.randrange(1, 99))),
         var("timestamp", "ts", "time", "sent"): pick(ts("isoz"), f"{rng.randrange(1_760_000_000, 1_800_000_000)}.{rng.randrange(10**6):06d}")}
    k = var("text", "message", "body", "content")
    r[k] = harmless(sp, 40, 600)
    if rng.random() < 0.6:
        r[var("reactions", "emoji")] = pick(
            [{"name": e, "count": rng.randrange(1, 9)} for e in rng.sample(EMOJI, rng.randrange(1, 4))],
            [e for e in rng.sample(EMOJI, rng.randrange(1, 4))])
    if rng.random() < 0.3:
        r[var("reply_count", "thread_replies")] = rng.randrange(0, 14)
    return r, [(k,)]


def r_contact(sp):
    n = person()
    r = {var("name", "full_name", "display_name"): n,
         var("email", "email_address"): email(n),
         var("phone", "phone_number", "mobile"): phone()}
    if rng.random() < 0.5:
        r["address"] = {"street": f"{rng.randrange(1, 300)} {rng.choice(STREETS)}", "city": rng.choice(CITIES),
                        var("postcode", "zip", "postal_code"): str(rng.randrange(10000, 99999))}
    else:
        r["address"] = f"{rng.randrange(1, 300)} {rng.choice(STREETS)}, {rng.choice(CITIES)}"
    r[var("company", "organization", "employer")] = rng.choice(COMPANIES)
    if rng.random() < 0.5:
        r[var("title", "job_title", "role")] = pick("Account Manager", "Nurse", "Teacher", "Engineer", "Analyst",
                                                     "Paralegal", "Chef", "Product Designer", "Accountant")
    if rng.random() < 0.3:
        r[var("birthday", "dob")] = ts("date")
    if rng.random() < 0.25:
        r[var("account", "iban", "bank_account")] = account()
    k = var("notes", "bio", "about")
    r[k] = harmless(sp, 40, 600)
    return r, [(k,)]


def r_order(sp):
    r = {var("id", "txn_id", "order_id", "transaction_id"): ident("txn"),
         var("amount", "total", "value"): money(),
         var("currency", "ccy"): rng.choice(CURRENCIES),
         var("date", "created", "posted_at"): ts(),
         var("status", "state"): rng.choice(STATUSES_ORDER),
         var("merchant", "payee", "counterparty", "recipient"): pick(rng.choice(COMPANIES), person())}
    if rng.random() < 0.5:
        r[var("account", "iban", "from_account")] = account()
    if rng.random() < 0.3:
        r[var("recurring", "subscription")] = rng.random() < 0.4
    if rng.random() < 0.3:
        r[var("items", "line_items")] = [{"sku": f"SKU-{rng.randrange(10000, 99999)}", "name": rng.choice(PRODUCTS),
                                          "qty": rng.randrange(1, 4), "price": money()} for _ in range(rng.randrange(1, 4))]
    k = var("memo", "note", "description", "reference")
    r[k] = harmless(sp, 40, 500)
    return r, [(k,)]


def r_listing(sp):
    kind = rng.choice(["hotel", "restaurant", "product", "rental"])
    name = (rng.choice(PRODUCTS).title() + " " + pick("Pro", "Lite", "2", "Max", "Mini")) if kind == "product" \
        else f"{pick('The ', '')}{rng.choice(PLACE_ADJ)} {rng.choice(PLACE_NOUN)}{pick('', ' Inn', ' Bistro', ' House', ' Suites', ' Kitchen')}"
    r = {var("name", "title"): name, var("city", "location"): rng.choice(CITIES),
         var("price", "price_per_night", "avg_price", "cost"): money(),
         var("rating", "stars", "score"): round(rng.uniform(2.5, 5.0), 1)}
    if rng.random() < 0.4:
        r[var("address", "street")] = f"{rng.randrange(1, 300)} {rng.choice(STREETS)}"
    if rng.random() < 0.3:
        r[var("phone", "contact")] = phone()
    k = var("description", "about", "summary", "details")
    r[k] = harmless(sp, 40, 600)
    slots = [(k,)]
    if rng.random() < 0.6:
        rk = var("reviews", "review_snippets", "comments")
        if rng.random() < 0.6:
            r[rk] = [harmless(sp, 40, 300) for _ in range(rng.randrange(1, 4))]
            slots += [(rk, i) for i in range(len(r[rk]))]
        else:
            r[rk] = [{"author": rng.choice(FIRST), "rating": rng.randrange(1, 6), "text": harmless(sp, 40, 300)}
                     for _ in range(rng.randrange(1, 4))]
            slots += [(rk, i, "text") for i in range(len(r[rk]))]
    return r, slots


def r_ticket(sp):
    r = {var("id", "number", "key"): pick(rng.randrange(1, 9000), f"{pick('PROJ', 'OPS', 'WEB', 'APP')}-{rng.randrange(1, 9000)}"),
         var("title", "summary"): short(sp),
         var("state", "status"): pick("open", "closed", "in progress", "triage", "done"),
         var("assignee", "owner"): pick(email(), rng.choice(FIRST).lower(), None),
         var("labels", "tags"): rng.sample(LABELS, rng.randrange(0, 4)),
         var("created_at", "opened", "created"): ts()}
    k = var("body", "description", "details")
    r[k] = harmless(sp)
    slots = [(k,)]
    if rng.random() < 0.55:
        ck = var("comments", "activity")
        r[ck] = [{"author": pick(email(), rng.choice(FIRST).lower()), "created": ts(), "body": harmless(sp, 40, 400)}
                 for _ in range(rng.randrange(1, 4))]
        slots += [(ck, i, "body") for i in range(len(r[ck]))]
    return r, slots


def r_search(sp):
    words = re.sub(r"[^\w ]", "", short(sp, 30, 60)).split()
    slug = "-".join(w.lower() for w in words[:5]) or "page"
    host = pick("www.example.com", "blog.example.org", "news.example.net", "docs.example.com", "wiki.example.org")
    r = {"title": " ".join(words[:8]) or "Untitled",
         var("url", "link", "href"): f"https://{host}/{pick('', 'articles/', 'posts/', '2026/')}{slug}"}
    k = var("snippet", "description", "summary", "content")
    r[k] = harmless(sp, 40, 500)
    if rng.random() < 0.4:
        r[var("rank", "position")] = rng.randrange(1, 11)
    if rng.random() < 0.3:
        r[var("published", "date")] = ts("date")
    return r, [(k,)]


def r_api_status(sp):
    r = {var("service", "name", "component"): rng.choice(SERVICES),
         "version": f"{rng.randrange(0, 6)}.{rng.randrange(0, 30)}.{rng.randrange(0, 20)}",
         "region": rng.choice(REGIONS),
         var("status", "health", "state"): pick("ok", "degraded", "healthy", "maintenance", "up"),
         var("flags", "features", "feature_flags"): pick(
             {f: rng.random() < 0.5 for f in rng.sample(["beta_ui", "new_billing", "fast_path", "dark_launch", "v2_api", "audit_log"], 3)},
             rng.sample(["beta_ui", "new_billing", "fast_path", "dark_launch", "v2_api", "audit_log"], rng.randrange(0, 4)))}
    if rng.random() < 0.5:
        r[var("uptime", "uptime_pct")] = round(rng.uniform(97, 100), 3)
    if rng.random() < 0.4:
        r[var("last_deploy", "deployed_at", "updated")] = ts("isoz")
    k = var("message", "notes", "status_message", "detail")
    r[k] = harmless(sp, 40, 600)
    return r, [(k,)]


TYPES = {"message": r_message, "document": r_document, "calendar": r_calendar, "chat": r_chat,
         "contact": r_contact, "order": r_order, "listing": r_listing, "ticket": r_ticket,
         "search": r_search, "api_status": r_api_status}
WRAP_LIST = ["results", "items", "data", "records", "entries", "hits"]
WRAP_COUNT = ["count", "total", "total_count", "n"]

# ------------------------------------------------------------------ rendering
RESERVED = {"y", "n", "yes", "no", "on", "off", "true", "false", "null", "none", "~"}
PLAIN = re.compile(r"^[^\W\d_][\w .,@/+()'&:#%-]*$")
LOOKS_TYPED = re.compile(r"^[-+]?(\d[\d_,]*)?(\.\d*)?([eE][-+]?\d+)?$|^\d{4}-\d\d-\d\d|^0[xob]|^[-+]?\.(inf|nan)$", re.I)


def y_scalar(v, flow=False):
    if flow and isinstance(v, str) and re.search(r"[,\[\]{}]", v):
        return json.dumps(v, ensure_ascii=False)
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if ("\n" not in v and PLAIN.match(v) and v.lower() not in RESERVED and not LOOKS_TYPED.match(v)
            and ": " not in v and " #" not in v and not v.endswith(":") and not v.endswith(" ")):
        return v
    if "\n" not in v and rng.random() < 0.5 and all(c.isprintable() for c in v):
        return "'" + v.replace("'", "''") + "'"
    return json.dumps(v, ensure_ascii=False)


class YStyle:
    def __init__(self):
        self.indentless = rng.random() < 0.6  # PyYAML's default list style
        self.block = rng.random()  # share of long strings written as | block scalars
        self.flow_short = rng.random() < 0.3

    def is_long(self, v):
        return isinstance(v, str) and (len(v) > 80 or "\n" in v)


def y_long(v, ind):
    chomp = "|-"
    return [chomp] + [(ind + l) if l else "" for l in v.split("\n")]


def y_map(d, ind, st):
    out = []
    for k, v in d.items():
        if isinstance(v, dict) and v:
            out.append(f"{ind}{k}:")
            out += y_map(v, ind + "  ", st)
        elif isinstance(v, list) and v:
            if st.flow_short and all(not isinstance(x, (dict, list)) and not st.is_long(x) for x in v):
                out.append(f"{ind}{k}: [" + ", ".join(y_scalar(x, flow=True) for x in v) + "]")
            else:
                out.append(f"{ind}{k}:")
                out += y_list(v, ind if st.indentless else ind + "  ", st)
        elif isinstance(v, (dict, list)):
            out.append(f"{ind}{k}: " + ("{}" if isinstance(v, dict) else "[]"))
        elif st.is_long(v) and v == v.strip() and rng.random() < st.block:
            b = y_long(v, ind + "  ")
            out.append(f"{ind}{k}: {b[0]}")
            out += b[1:]
        else:
            out.append(f"{ind}{k}: {y_scalar(v)}")
    return out


def y_list(items, ind, st):
    out = []
    for it in items:
        if isinstance(it, dict) and it:
            sub = y_map(it, ind + "  ", st)
            sub[0] = ind + "- " + sub[0][len(ind) + 2 :]
            out += sub
        elif isinstance(it, list) and it:
            sub = y_list(it, ind + "  ", st)
            sub[0] = ind + "- " + sub[0][len(ind) + 2 :]
            out += sub
        elif st.is_long(it) and it == it.strip() and rng.random() < st.block:
            b = y_long(it, ind + "  ")
            out.append(f"{ind}- {b[0]}")
            out += b[1:]
        else:
            out.append(f"{ind}- {y_scalar(it)}")
    return out


def to_yaml(obj, st):
    lines = y_map(obj, "", st) if isinstance(obj, dict) else y_list(obj, "", st)
    return "\n".join(lines)


def to_json(obj, st):
    return json.dumps(obj, ensure_ascii=st["ascii"], indent=st["indent"], separators=st["sep"])


def render(obj, fmt, st):
    state = rng.getstate()  # identical choices on every render of this row
    try:
        return to_json(obj, st) if fmt == "json" else to_yaml(obj, st)
    finally:
        rng.setstate(state)


def est_tokens(s):
    """Conservative token count (about 15% over Ettin's on these rows)."""
    a = sum(1 for c in s if ord(c) < 128)
    esc = len(re.findall(r"\\u[0-9a-fA-F]{4}", s))  # \uXXXX from ensure_ascii JSON
    return (a - 6 * esc) / 2.8 + 3 * esc + (len(s) - a) * 1.0


def get(obj, path):
    for p in path:
        obj = obj[p]
    return obj


def put(obj, path, v):
    get(obj, path[:-1])[path[-1]] = v


def shrink(t):
    n = max(40, int(len(t) * 0.6))
    if len(t) <= n:
        return t
    t = t[:n]
    return (t[: t.rfind(" ")] if " " in t[30:] else t).strip()


# ------------------------------------------------------------------- build rows
def harmless_fields(o, out=None):
    out = [] if out is None else out
    if isinstance(o, dict):
        for v in o.values():
            harmless_fields(v, out)
    elif isinstance(o, list):
        for v in o:
            harmless_fields(v, out)
    elif isinstance(o, str) and len(o) >= 40:
        out.append(o)
    return out


SENT = "QZXSENTINELQZX"
SEPS = ["\n\n", "\n", " ", "\n\n---\n", "\n\n"]


def make_row(rtype, fmt, split, attack, builder=None, mirror=None, count=None, modes=("append", "replace"),
             source=None, shrink_fn=None):
    """attack: None for benign, else the attack text. Returns (row, meta).

    The keyword arguments are for later rounds (build_records4.py): builder
    makes one (record, slots), mirror makes the second harmless text of a
    benign "append" row, count draws the number of records, modes are the
    ways the payload goes into the field ("prepend" puts it first)."""
    builder, mirror, shrink_ = builder or TYPES[rtype], mirror or harmless, shrink_fn or shrink
    # Count and payload position are drawn once, so retries don't bias the
    # payload towards the first record.
    n = count() if count else (1 if rng.random() < 0.35 else rng.randrange(2, 6))
    target = rng.randrange(n)
    for _attempt in range(40):
        if _attempt and _attempt % 8 == 0 and target > 0:
            target -= 1  # only when the payload can't be reached at all
        SCHEMA.clear()
        recs = [builder(split) for _ in range(n)]
        records = [r for r, _ in recs]
        slot = rng.choice(recs[target][1])
        if n == 1:
            shape, root, base = "single", records[0], ()
        elif rng.random() < 0.3:
            lk, ck = rng.choice(WRAP_LIST), rng.choice(WRAP_COUNT)
            root = {lk: records, ck: n} if rng.random() < 0.7 else {ck: n, lk: records}
            if rng.random() < 0.25:
                root[pick("next_cursor", "next_page", "page")] = pick(None, rng.randrange(2, 9), f"c_{rng.randrange(16**8):08x}")
            shape, base = "wrapped", (lk,)
        else:
            shape, root, base = "list", records, ()
        path = base + ((target,) if n > 1 else ()) + slot
        orig = get(root, path)
        mode = rng.choice(list(modes))
        sep = rng.choice(SEPS)
        if fmt == "json":
            st = {"ascii": rng.random() < 0.25, "indent": pick(None, None, 2, 2, 4),
                  "sep": None if rng.random() < 0.7 else (",", ":")}
        else:
            st = YStyle()
        # Position check with a sentinel where the payload will start. Records
        # before the target are shortened until the payload is in reach.
        for _ in range(10):
            put(root, path, (orig + sep + SENT) if mode == "append" else SENT)
            txt = render(root, fmt, st)
            if est_tokens(txt[: txt.index(SENT)]) <= TOKEN_BUDGET:
                break
            if mode == "append" and len(orig) > 40:
                orig = shrink_(orig)
            for p in ([(base + ((i,) if n > 1 else ()) + s) for i in range(target + 1) for s in recs[i][1]]):
                if p != path:
                    v = get(root, p)
                    if isinstance(v, str):
                        put(root, p, shrink_(v))
        else:
            continue
        put(root, path, orig)
        fields = harmless_fields(root)  # the external benign text in this row
        if attack is not None:
            payload = clean(attack)
            value = {"append": orig + sep + payload, "prepend": payload + sep + orig}.get(mode, payload)
        else:
            # Mirror the attack rows' field lengths: half the time two
            # harmless texts joined the same way.
            value = {"append": lambda: orig + sep + mirror(split),
                     "prepend": lambda: mirror(split) + sep + orig}.get(mode, lambda: orig)()
        put(root, path, value)
        if attack is None:
            fields = harmless_fields(root)
        text = render(root, fmt, st)
        return ({"text": text, "label": "injection" if attack is not None else "benign", "source": source or f"records_{rtype}"},
                {"type": rtype, "format": fmt, "shape": shape, "split": split, "mode": mode, "n": n,
                 "target": target, "obj": root, "fields": fields, "value": value})
    raise RuntimeError(f"could not place payload for {rtype}/{fmt}")


types = list(TYPE_WEIGHT)
weights = [TYPE_WEIGHT[t] for t in types]
p_held = sum(TYPE_WEIGHT[t] * 0.5 for t, f in HELD_COMBOS) / sum(weights)
row_dev = (DEV_SHARE - p_held) / (1 - p_held)


def plan(n_rows):
    out = []
    for _ in range(n_rows):
        t = rng.choices(types, weights)[0]
        f = rng.choice(["json", "yaml"])
        sp = "dev" if (t, f) in HELD_COMBOS or rng.random() < row_dev else "train"
        out.append((t, f, sp))
    return out


def main():
    benign_plan, attack_plan = plan(N_BENIGN), plan(N_ATTACK)
    sched = {sp: attack_schedule(sp, sum(1 for *_, s in attack_plan if s == sp)) for sp in ("train", "dev")}
    rows, metas = [], []
    for t, f, sp in benign_plan:
        r, m = make_row(t, f, sp, None)
        m["attack_src"] = None
        rows.append(r), metas.append(m)
    for t, f, sp in attack_plan:
        if not sched[sp]:
            continue
        src, a = sched[sp].pop()
        r, m = make_row(t, f, sp, a)
        m["attack_src"], m["attack"] = src, a
        rows.append(r), metas.append(m)

    # ---------------------------------------------------------------------- checks
    errors = []
    try:
        import yaml

        bad = 0
        for r, m in zip(rows, metas):
            if m["format"] == "yaml":
                if yaml.safe_load(r["text"]) != m["obj"]:
                    bad += 1
            elif json.loads(r["text"]) != m["obj"]:
                bad += 1
        if bad:
            errors.append(f"{bad} rows do not parse back to their records")
        print("parse-back check: all rows load as JSON/YAML and equal their records" if not bad else "")
    except ImportError:
        print("pyyaml not installed: skipped the YAML parse-back check")

    # 1. Nothing from dev / dev_ood / dev_agent / held-out in the output.
    exact = [r for r in rows if norm(r["text"]) in eval_norm]
    if exact:
        errors.append(f"{len(exact)} rows equal an eval text")
    field_leaks = sum(1 for m in metas for t in m["fields"] if overlaps_eval(t))
    if field_leaks:
        errors.append(f"{field_leaks} harmless fields overlap an eval text (>= {W + STRIDE - 1} chars)")
    out_blob = "\n".join([norm(r["text"]) for r in rows] + [norm(m["value"]) for m in metas])
    sub_hits = 0
    for t in eval_texts:
        k = norm(t)
        if len(k) >= 200 and k[:200] in out_blob:
            sub_hits += 1
    for m in metas:
        for t in m["fields"] + [m["value"]] + ([clean(m["attack"])] if m["attack_src"] else []):
            k = norm(t)
            if len(k) >= 200 and k[:200] in eval_blob:
                sub_hits += 1
    if sub_hits:
        errors.append(f"{sub_hits} 200-character overlaps between output and eval texts")

    # 2. Every attack text is a train injection row, used only in its own split.
    used = collections.defaultdict(set)
    for m in metas:
        if m["attack_src"]:
            if m["attack"] not in train_inj or clean(m["attack"]) not in m["value"]:
                errors.append("attack text not from train injection rows")
                break
            used[m["attack"]].add(m["split"])
    if any(len(v) > 1 for v in used.values()):
        errors.append("an attack text is in both records_train and records_dev")

    # 3. Counts.
    cnt = collections.Counter(r["label"] for r in rows)
    dev_share = sum(m["split"] == "dev" for m in metas) / len(metas)
    if not (0.95 * N_BENIGN <= cnt["benign"] <= 1.05 * N_BENIGN):
        errors.append(f"benign count {cnt['benign']} out of range")
    if not (0.95 * N_ATTACK <= cnt["injection"] <= 1.05 * N_ATTACK):
        errors.append(f"attack count {cnt['injection']} out of range")
    if not (0.17 <= dev_share <= 0.23):
        errors.append(f"records_dev share {dev_share:.3f} out of range")
    src_share = collections.Counter(m["attack_src"] for m in metas if m["attack_src"])
    if max(src_share.values()) > 0.30 * cnt["injection"]:
        errors.append(f"one attack source dominates: {src_share}")
    for sp in ("train", "dev"):
        labs = {r["label"] for r, m in zip(rows, metas) if m["split"] == sp}
        if labs != {"benign", "injection"}:
            errors.append(f"{sp} split lacks a label")
    if any(m["split"] == "train" and (m["type"], m["format"]) in HELD_COMBOS for m in metas):
        errors.append("held-out combo in train")

    # ---------------------------------------------------------------------- report
    def table(title, key):
        c = collections.Counter()
        for r, m in zip(rows, metas):
            c[(key(m), m["split"], r["label"])] += 1
        print(f"\n{title:<22} {'train ben':>9} {'train inj':>9} {'dev ben':>8} {'dev inj':>8}")
        for k in sorted({k[0] for k in c}, key=str):
            print(f"{str(k):<22} {c[(k, 'train', 'benign')]:>9} {c[(k, 'train', 'injection')]:>9} "
                  f"{c[(k, 'dev', 'benign')]:>8} {c[(k, 'dev', 'injection')]:>8}")


    table("type", lambda m: m["type"])
    table("format", lambda m: m["format"])
    table("type/format", lambda m: f"{m['type']}/{m['format']}")
    table("shape", lambda m: m["shape"])
    table("records in row", lambda m: m["n"])
    table("payload record", lambda m: "only" if m["n"] == 1 else ("first" if m["target"] == 0 else ("last" if m["target"] == m["n"] - 1 else "middle")))
    table("insert mode", lambda m: m["mode"])
    table("attack source", lambda m: m["attack_src"] or "-")
    lens = sorted(len(r["text"]) for r in rows)
    print(f"\nharmless texts drawn: {dict(LANG_DRAWN)} ({LANG_DRAWN['multi'] / sum(LANG_DRAWN.values()):.1%} multilingual Wikipedia)")
    print(f"rendered length: median {lens[len(lens) // 2]}, p90 {lens[int(len(lens) * 0.9)]}, max {lens[-1]}")
    print(f"records_dev share {dev_share:.3f} (held-out combos {p_held:.3f} of draws, row split {row_dev:.3f} of the rest)")

    if errors:
        print("\nFAILED:\n  " + "\n  ".join(errors))
        sys.exit(1)

    # ----------------------------------------------------------------------- write
    os.makedirs(OUT, exist_ok=True)
    new_train = [r for r, m in zip(rows, metas) if m["split"] == "train"]
    dev_rows = [r for r, m in zip(rows, metas) if m["split"] == "dev"]
    all_train = train_rows + new_train
    rng.shuffle(all_train)
    rng.shuffle(dev_rows)
    for name, rs in (("train", all_train), ("records_dev", dev_rows)):
        with open(f"{OUT}/{name}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for s in ("dev", "dev_ood", "dev_agent"):
        shutil.copy(f"{DATA}/{s}.jsonl", f"{OUT}/{s}.jsonl")
    print(f"\nwrote {OUT}/train.jsonl ({len(all_train)} rows, {len(new_train)} new) and "
          f"{OUT}/records_dev.jsonl ({len(dev_rows)} rows); all checks passed")


if __name__ == "__main__":
    main()
