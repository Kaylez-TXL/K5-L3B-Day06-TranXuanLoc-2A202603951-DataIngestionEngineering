"""Local E5-small embeddings; identical model and normalization for A/B/C."""
import json
import math
from pathlib import Path
from typing import Protocol

from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """Encode E5 inputs on CPU; reject overflow instead of silently truncating."""

    def __init__(self, model_name: str = MODEL_NAME, *, revision: str = MODEL_REVISION,
                 local_files_only: bool = False) -> None:
        if model_name != MODEL_NAME:
            raise ValueError(f"This adapter implements the E5 contract for {MODEL_NAME} only.")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu", revision=revision,
                                         local_files_only=local_files_only)
        self.dimensions = self.model.get_embedding_dimension()
        if self.dimensions != 384:
            raise ValueError(f"Unexpected E5-small dimensions: {self.dimensions}")
        self.max_tokens = min(512, self.model.max_seq_length)
        self.model.max_seq_length = self.max_tokens
        self.revision = revision

    def prepare_inputs(self, texts: list[str], prefix: str) -> list[str]:
        if prefix not in ("query: ", "passage: "):
            raise ValueError("E5 requires query or passage prefix.")
        inputs = []
        for i, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Empty or invalid embedding input at index {i}.")
            value = prefix + text
            tokens = self.model.tokenizer(value, add_special_tokens=True, truncation=False)["input_ids"]
            if len(tokens) > self.max_tokens:
                raise ValueError(f"Input {i} has {len(tokens)} tokens, limit {self.max_tokens}; "
                                 "split the text before embedding. No truncation was applied.")
            inputs.append(value)
        return inputs

    def _encode(self, texts: list[str], prefix: str) -> list[list[float]]:
        if not texts:
            return []
        inputs = self.prepare_inputs(texts, prefix)
        vectors = self.model.encode(inputs, batch_size=16, normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=False).tolist()
        if len(vectors) != len(texts):
            raise ValueError("Embedding count does not match input count.")
        for vector in vectors:
            if len(vector) != self.dimensions or not all(math.isfinite(x) for x in vector):
                raise ValueError("Invalid embedding dimensions or non-finite values.")
            if not math.isclose(sum(x*x for x in vector), 1.0, abs_tol=1e-4):
                raise ValueError("Embedding is not unit-normalized.")
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, "passage: ")

    def embed_query(self, question: str) -> list[float]:
        return self._encode([question], "query: ")[0]


if __name__ == "__main__":
    import argparse
    from .chunking import fixed_size, by_paragraph
    from .models import DocumentPage

    parser = argparse.ArgumentParser(description="Check E5-small inputs and real embedding output.")
    parser.add_argument("--offline", action="store_true", help="Use only the already downloaded model.")
    embedder = MyEmbedder(local_files_only=parser.parse_args().offline)
    repo = Path(__file__).resolve().parents[2]
    rows = map(json.loads, (repo / "data/processed/pages.jsonl").read_text(encoding="utf-8").splitlines())
    count = 0
    for row in rows:
        page = DocumentPage(**{k: row[k] for k in ("document_id", "source", "page", "text", "metadata")})
        for chunks in (fixed_size(page, 500, 50), by_paragraph(page, 500)):
            embedder.prepare_inputs([chunk.text for chunk in chunks], "passage: ")
            count += len(chunks)
    questions = list(map(json.loads, (repo / "benchmark/questions.jsonl").read_text(encoding="utf-8").splitlines()))
    embedder.prepare_inputs([q["question"] for q in questions], "query: ")
    # This smoke check uses a generic example, not benchmark retrieval/test scores.
    document = embedder.embed_documents(["Hạt nano Au@Ag gồm lõi vàng và vỏ bạc."])[0]
    query = embedder.embed_query("Hạt Au@Ag gồm những kim loại nào?")
    print(f"Model: {embedder.model_name} | revision: {embedder.revision}")
    print(f"Dimensions: {len(document)} | max tokens: {embedder.max_tokens}")
    print(f"Token checks passed: {count} chunks (A + B), {len(questions)} questions; no truncation.")
    print(f"Example cosine similarity: {sum(a*b for a,b in zip(document, query)):.4f}")
    print("Real model smoke check only; no benchmark retrieval or database writes.")
