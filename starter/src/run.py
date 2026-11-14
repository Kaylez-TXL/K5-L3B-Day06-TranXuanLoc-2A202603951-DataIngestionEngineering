"""Reproducible E5 + pgvector experiment. Run from starter: python -m src.run --offline."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import time
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

from dotenv import load_dotenv
from .chunking import fixed_size, by_paragraph
from .embeddings import MyEmbedder
from .evaluation import evaluate_query, summarize, summarize_audit, validate_benchmark
from .ingestion import prepare_corpus, write_json
from .models import DocumentPage
from .store import PgVectorStore

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results"
CONFIG = {
    "A": {"chunking": "fixed_size", "size": 500, "overlap": 50, "filter": False},
    "B": {"chunking": "by_paragraph", "size": 500, "overlap": 0, "filter": False},
    "C": {"chunking": "by_paragraph", "size": 500, "overlap": 0, "filter": True},
}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare_audit(chunks_by_strategy, traces):
    lock = json.loads((REPO / "benchmark/split_lock.json").read_text(encoding="utf-8"))
    rows = []
    for strategy, chunks in chunks_by_strategy.items():
        for anchor in lock["coherence_sampling"]["anchors"]:
            candidates = [c for c in chunks if c.document_id == anchor["document_id"] and c.page == anchor["page"]
                          and c.metadata["char_start"] <= anchor["char_offset"] < c.metadata["char_end"]]
            chosen = sorted(candidates, key=lambda c: (-c.metadata["char_end"], c.metadata["char_start"]))[0]
            rows.append({"kind": "coherence", "strategy": strategy, "item_id": f"page-{anchor['page']}",
                         "source": chosen.source, "page": chosen.page, "chunk_ids": chosen.chunk_id,
                         "text": chosen.text, "context": anchor["source_excerpt"]})
    for trace in traces:
        if trace["split"] != "test":
            continue
        rows.append({"kind": "grounding", "strategy": trace["strategy"], "item_id": trace["id"],
                     "text": json.dumps(trace["hits"], ensure_ascii=False),
                     "context": json.dumps({k: trace[k] for k in ["question", "expected_answer", "evidence_quote", "gold_page"]}, ensure_ascii=False)})
        for rank, hit in enumerate(trace["hits"], 1):
            rows.append({"kind": "metadata", "strategy": trace["strategy"], "item_id": f"{trace['id']}-rank{rank}",
                         "source": hit["source"], "page": hit["page"], "chunk_ids": hit["chunk_id"],
                         "text": hit["text"], "context": json.dumps({"metadata": hit["metadata"], "filter": trace["filter"]}, ensure_ascii=False)})
    path = OUT / "manual_audit.csv"
    # Preserve labels keyed to the exact text, even when rerunning the pipeline.
    previous = {}
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as stream:
            previous = {(r["kind"], r["strategy"], r["item_id"]): r for r in csv.DictReader(stream)}
    fields = ["kind", "strategy", "item_id", "source", "page", "chunk_ids", "text", "context", "score", "reason", "reviewer", "status"]
    for row in rows:
        old = previous.get((row["kind"], row["strategy"], row["item_id"]), {})
        keep = old.get("text") == row["text"] and old.get("context") == row["context"]
        row.update({key: old.get(key, "") if keep else "" for key in ["score", "reason", "reviewer"]})
        row["status"] = old.get("status", "pending") if keep else "pending"
    write_csv(path, rows, fields)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="Use the downloaded model without network access.")
    args = parser.parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    OUT.mkdir(exist_ok=True)
    load_dotenv(REPO / "starter/.env")
    url = os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL missing; copy starter/.env.example to starter/.env.")
    started = time.perf_counter()
    prepare_corpus()
    write_json(OUT / "benchmark_validation.json", validate_benchmark())
    pages = [DocumentPage(**{k: row[k] for k in ["document_id", "source", "page", "text", "metadata"]})
             for row in read_jsonl(REPO / "data/processed/pages.jsonl")]
    questions = read_jsonl(REPO / "benchmark/questions.jsonl")
    embedder = MyEmbedder(local_files_only=args.offline)
    store = PgVectorStore(url, embedder.dimensions)
    stats, chunks_by_strategy, traces = {}, {}, []
    try:
        store.create_schema()
        chunks_a = [c for p in pages for c in fixed_size(p, 500, 50)]
        chunks_b = [c for p in pages for c in by_paragraph(p, 500)]
        for name, chunks in [("A", chunks_a), ("B", chunks_b), ("C", chunks_b)]:
            t0 = time.perf_counter()
            if name != "C":
                vectors = embedder.embed_documents([c.text for c in chunks])
            for chunk in chunks:
                chunk.metadata.update({"embedding_model": embedder.model_name, "embedding_revision": embedder.revision})
            store.upsert(name, chunks, vectors)
            first_count = store.count(name)
            ingest_seconds = time.perf_counter() - t0
            t1 = time.perf_counter()
            store.upsert(name, chunks, vectors)
            if store.count(name) != first_count or first_count != len(chunks):
                raise AssertionError("Re-ingestion duplicated or lost chunks.")
            tokens = [len(embedder.model.tokenizer("passage: " + c.text, truncation=False)["input_ids"]) for c in chunks]
            lengths = sorted(len(c.text) for c in chunks)
            stats[name] = {"chunk_count": len(chunks), "characters": sum(lengths), "min_chars": lengths[0],
                "mean_chars": sum(lengths)/len(lengths), "max_chars": lengths[-1], "tokens": sum(tokens), "max_tokens": max(tokens),
                "ingest_seconds": ingest_seconds, "reingest_seconds": time.perf_counter()-t1,
                "reingest_count": first_count, "embedding_reused_from": "B" if name == "C" else None, "api_cost": 0}
            chunks_by_strategy[name] = chunks
            print(f"Stored {name}: {first_count} chunks; duplicate check passed.", flush=True)
        # Verify durability through an independent database connection.
        independent = PgVectorStore(url, embedder.dimensions)
        try:
            assert all(independent.count(s) == stats[s]["chunk_count"] for s in CONFIG)
        finally:
            independent.close()
        selection = None
        query_vectors = {}
        for split in ["dev", "test"]:
            for q in (q for q in questions if q["split"] == split):
                query_vectors[q["id"]] = embedder.embed_query(q["question"])
                for name in CONFIG:
                    filters = q.get("filter", {}) if CONFIG[name]["filter"] else {}
                    t0 = time.perf_counter()
                    hits = store.search(name, query_vectors[q["id"]], filters=filters)
                    trace = evaluate_query(q, hits)
                    trace.update(strategy=name, applied_filter=filters, search_seconds=time.perf_counter()-t0)
                    traces.append(trace)
            if split == "dev":
                dev = summarize(traces)
                selected = max(CONFIG, key=lambda s: (dev[s]["dev"]["hit3"], dev[s]["dev"]["hit1"], -stats[s]["ingest_seconds"]))
                selection = {"strategy": selected, "selected_before_test": True, "rule": "dev Hit@3, dev Hit@1, lower measured ingest time",
                    "dev_metrics": dev, "config": CONFIG, "benchmark_sha256": hashlib.sha256((REPO / "benchmark/questions.jsonl").read_bytes()).hexdigest(),
                    "frozen_at": datetime.now(timezone(timedelta(hours=7))).isoformat()}
                write_json(OUT / "selection.json", selection)
                print(f"Selected on dev: {selected}. Configuration frozen before test.", flush=True)
        with store.connection.cursor() as cur:
            cur.execute("SELECT current_setting('server_version'), (SELECT extversion FROM pg_extension WHERE extname='vector')")
            pg, vector_version = cur.fetchone()
        metrics = {"run_at": datetime.now(timezone(timedelta(hours=7))).isoformat(), "retrieval": summarize(traces), "resources": stats,
            "config": CONFIG, "model": embedder.model_name, "model_revision": embedder.revision, "dimensions": embedder.dimensions,
            "device": "cpu", "document_prefix": "passage: ", "query_prefix": "query: ", "normalization": "unit L2", "max_tokens": embedder.max_tokens,
            "python": platform.python_version(), "packages": {p: importlib.metadata.version(p) for p in ["sentence-transformers", "transformers", "torch", "pypdf", "psycopg", "python-dotenv"]},
            "postgresql": pg, "pgvector": vector_version, "search": "exact cosine; SQL filter before rank; no ANN index",
            "total_seconds": time.perf_counter()-started, "api_cost": 0, "cost_note": "Local CPU inference; excludes electricity, hardware, network and human time.",
            "durability_check": "independent connection counts passed", "manual_audit_status": "pending human review", "selected_strategy": selection["strategy"]}
        # Machine checks support the human audit, but do not replace it.
        page_map = {(p.document_id, p.page): p for p in pages}
        checks = {}
        for name in CONFIG:
            group = [t for t in traces if t["strategy"] == name and t["split"] == "test"]
            records = [h for t in group for h in t["hits"]]
            valid = sum((h["document_id"], h["page"]) in page_map and
                       h["source"] == page_map[(h["document_id"], h["page"])].source and
                       h["metadata"]["chapter"] == page_map[(h["document_id"], h["page"])].metadata["chapter"] for h in records)
            filtered = [t for t in group if t.get("filter")]
            compliant = sum(len(t["hits"]) == 3 and all(all(h["metadata"].get(k) == v for k,v in t["filter"].items()) for h in t["hits"]) for t in filtered)
            checks[name] = {"metadata_auto_correct": valid, "metadata_records": len(records), "filter_questions": len(filtered), "filter_compliant": compliant,
                            "note": "Checks requested filter even for A/B which deliberately do not apply it; manual review still required."}
        metrics["metadata_checks"] = checks
        write_json(OUT / "metrics.json", metrics)
        with (OUT / "traces.jsonl").open("w", encoding="utf-8") as stream:
            for trace in traces:
                stream.write(json.dumps(trace, ensure_ascii=False) + "\n")
        prepare_audit(chunks_by_strategy, traces)
        audit = summarize_audit(OUT / "manual_audit.csv")
        write_json(OUT / "audit_summary.json", audit)
        metrics["manual_audit"] = audit
        confirmed = sum(g["confirmed"] for s in audit.values() for g in s.values())
        expected = sum(g["expected"] for s in audit.values() for g in s.values())
        metrics["manual_audit_status"] = f"{confirmed}/{expected} author-confirmed"
        write_json(OUT / "metrics.json", metrics)
        log_rows = []
        hypotheses = {"A": "Đo baseline fixed-size; overlap có thể giảm mất ngữ cảnh nhưng tăng trùng lặp.", "B": "Ranh giới cấu trúc có thể giữ ý tốt hơn; PDF xuống dòng có thể gây chia chưa đúng.", "C": "Filter chương có thể giảm nhiễu khi câu hỏi chỉ rõ phạm vi; có thể loại bằng chứng ở chương khác."}
        for name, cfg in CONFIG.items():
            r = metrics["retrieval"][name]
            log_rows.append(dict(strategy=name,chunking=cfg["chunking"],chunk_size=500,overlap=cfg["overlap"],metadata="document_id/source/page/chapter",filter="chapter if supplied" if cfg["filter"] else "none",vectorizer=embedder.model_name,hypothesis=hypotheses[name],dev_hit1=r["dev"]["hit1"],dev_hit3=r["dev"]["hit3"],test_hit1=r["test"]["hit1"],test_hit3=r["test"]["hit3"],notes="Strict quote proxy; audit confirmation in results/audit_summary.json; semantic alternatives require explicit source-level review."))
        fields = "strategy,chunking,chunk_size,overlap,metadata,filter,vectorizer,hypothesis,dev_hit1,dev_hit3,test_hit1,test_hit3,notes".split(",")
        write_csv(REPO / "artifacts/experiment_log.csv", log_rows, fields)
        print(json.dumps(metrics["retrieval"], ensure_ascii=False, indent=2), flush=True)
    finally:
        store.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        OUT.mkdir(exist_ok=True)
        # Do not log DATABASE_URL, credentials or raw provider exceptions.
        with (OUT / "errors.log").open("a", encoding="utf-8") as stream:
            stream.write(datetime.now(timezone.utc).isoformat() + f" Pipeline failed: {type(exc).__name__}; inspect local terminal. No secret values logged.\n")
        raise
