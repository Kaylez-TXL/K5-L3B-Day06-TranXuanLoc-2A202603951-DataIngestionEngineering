"""Validate benchmark structure and exact evidence before retrieval; no model needed."""
import hashlib
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]


def normalized(text):
    return re.sub(r"\s+", " ", text).strip()


def validate_benchmark(repo=REPO):
    path = repo / "benchmark/questions.jsonl"
    questions = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    lock = json.loads((repo / "benchmark/split_lock.json").read_text(encoding="utf-8"))
    pages = {(r["document_id"], r["page"]): r for r in map(json.loads, (repo / "data/processed/pages.jsonl").read_text(encoding="utf-8").splitlines())}
    errors = []
    def require(condition, message):
        if not condition:
            errors.append(message)
    require(hashlib.sha256(path.read_bytes()).hexdigest() == lock["benchmark_sha256"], "Benchmark changed after split lock; review and document before retrieval.")
    require(len(questions) >= 20, "Need at least 20 questions.")
    require(len({q["id"] for q in questions}) == len(questions), "Duplicate question IDs.")
    splits = Counter(q["split"] for q in questions)
    require(splits["dev"] >= 15 and splits["test"] >= 5, "Need at least 15 dev and 5 test.")
    require(set(splits) <= {"dev", "test"}, "Unknown split.")
    require(sum(bool(q.get("filter")) for q in questions) >= 3, "Need at least 3 filter questions.")
    require(any(q.get("filter") and q["split"] == "test" for q in questions), "Need a filter question in test.")
    for split in ["dev", "test"]:
        require([q["id"] for q in questions if q["split"] == split] == lock[f"{split}_ids"], f"{split} IDs differ from lock.")
    for q in questions:
        for key in ["question", "expected_answer", "evidence_quote"]:
            require(bool(q.get(key, "").strip()), f"{q['id']}: empty {key}.")
        golds = [{"source_id": q["gold_source_id"], "page": q["gold_page"], "evidence_quote": q["evidence_quote"]}] + q.get("alternative_gold", [])
        for gold in golds:
            page = pages.get((gold["source_id"], gold["page"]))
            require(page is not None, f"{q['id']}: unknown gold page.")
            if page:
                require(bool(gold["evidence_quote"].strip()) and normalized(gold["evidence_quote"]) in normalized(page["text"]), f"{q['id']}: evidence missing from source page.")
                require(all(page["metadata"].get(k) == v for k, v in q.get("filter", {}).items()), f"{q['id']}: gold contradicts question filter.")
        for key, value in q.get("filter", {}).items():
            require(key == "chapter" and type(value) is int and 1 <= value <= 5, f"{q['id']}: invalid filter schema.")
            require(f"chương {value}" in q["question"].lower(), f"{q['id']}: filter scope must be explicit in question.")
    if errors:
        raise ValueError("\n".join(errors))
    return {"questions": len(questions), "splits": dict(splits), "filter_questions": sum(bool(q.get("filter")) for q in questions), "test_filter_questions": sum(bool(q.get("filter")) and q["split"] == "test" for q in questions), "evidence_checks": "passed", "split_lock": "passed", "human_review": lock.get("author_review", {"status": "pending", "note": "Exact matching does not replace semantic/visual verification."})}




def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Strict full-evidence proxy; semantic alternatives need documented review."""
    golds = [{"source_id": question["gold_source_id"], "page": question["gold_page"],
              "evidence_quote": question["evidence_quote"]}] + question.get("alternative_gold", [])
    labelled = []
    for hit in hits[:3]:
        exact = any(hit["document_id"] == g["source_id"] and hit["page"] == g["page"]
                    and normalized(g["evidence_quote"]) in normalized(hit["text"]) for g in golds)
        labelled.append({**hit, "evidence_hit": exact})
    return {**question, "hits": labelled, "hit1": bool(labelled and labelled[0]["evidence_hit"]),
            "hit3": any(h["evidence_hit"] for h in labelled),
            "scoring": "source/page + complete normalized quote; semantic alternatives pending review"}


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for strategy in sorted({t["strategy"] for t in traces}):
        result[strategy] = {}
        for split in ("dev", "test"):
            group = [t for t in traces if t["strategy"] == strategy and t["split"] == split]
            n = len(group)
            hit1, hit3 = sum(t["hit1"] for t in group), sum(t["hit3"] for t in group)
            result[strategy][split] = {"n": n, "hit1_count": hit1, "hit3_count": hit3,
                "hit1": hit1/n if n else None, "hit3": hit3/n if n else None}
    return result


def summarize_audit(path):
    """Separate provisional AI labels from author-confirmed scores."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for strategy in ("A", "B", "C"):
        result[strategy] = {}
        for kind in ("coherence", "grounding", "metadata"):
            group = [r for r in rows if r["strategy"] == strategy and r["kind"] == kind]
            maximum = 1 if kind == "metadata" else 2
            for row in group:
                if row["score"] and (row["score"] not in [str(i) for i in range(maximum+1)] or not row["reason"]):
                    raise ValueError("Audit requires a valid score and explanation.")
            confirmed = [r for r in group if r["status"] == "confirmed" and r["reviewer"] and r["score"]]
            proposed = [int(r["score"]) for r in group if r["score"]]
            result[strategy][kind] = {"expected": len(group), "confirmed": len(confirmed),
                "confirmed_mean": sum(int(r["score"]) for r in confirmed)/len(confirmed) if confirmed else None,
                "labelled_mean_including_proposals": sum(proposed)/len(proposed) if proposed else None,
                "labelled_distribution_including_proposals": {str(i): proposed.count(i) for i in range(maximum+1)}}
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true", help="Aggregate audit labels; keep author confirmation separate.")
    if parser.parse_args().audit:
        result = summarize_audit(REPO / "results/manual_audit.csv")
        (REPO / "results/audit_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0)
    result = validate_benchmark()
    output = REPO / "results/benchmark_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
