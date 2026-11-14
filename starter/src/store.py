"""Persistent exact cosine search; strategy ingestion is an atomic snapshot."""
import math
from typing import Any
import psycopg
from psycopg.types.json import Jsonb
from .models import Chunk


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        if not isinstance(dimensions, int) or not 1 <= dimensions <= 16000:
            raise ValueError("Invalid vector dimensions.")
        self.dimensions = dimensions
        self.connection = psycopg.connect(database_url, connect_timeout=10, autocommit=True)

    def close(self):
        self.connection.close()

    def _vector(self, values):
        if len(values) != self.dimensions or not all(math.isfinite(x) for x in values):
            raise ValueError("Vector dimensions or values are invalid.")
        if not any(values):
            raise ValueError("Cosine distance is undefined for zero vectors.")
        return "[" + ",".join(str(float(x)) for x in values) + "]"

    def create_schema(self) -> None:
        with self.connection.transaction(), self.connection.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"""CREATE TABLE IF NOT EXISTS day6_chunks (
                strategy text NOT NULL, chunk_id text NOT NULL, document_id text NOT NULL,
                source text NOT NULL, page integer NOT NULL, text text NOT NULL,
                metadata jsonb NOT NULL, embedding vector({self.dimensions}) NOT NULL,
                PRIMARY KEY(strategy, chunk_id))""")
            cur.execute("SELECT format_type(atttypid, atttypmod) FROM pg_attribute WHERE attrelid='day6_chunks'::regclass AND attname='embedding'")
            if cur.fetchone()[0] != f"vector({self.dimensions})":
                raise ValueError("Existing table uses different vector dimensions; migrate explicitly.")

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Replace the complete strategy snapshot, including removed documents.

        Validate first, then delete/insert in one transaction: failed inserts roll
        back to the old snapshot. Caller must supply the complete strategy corpus.
        """
        if len(chunks) != len(vectors) or len({c.chunk_id for c in chunks}) != len(chunks):
            raise ValueError("Chunk/vector counts differ or chunk IDs are duplicated.")
        rows = [(strategy, c.chunk_id, c.document_id, c.source, c.page, c.text,
                 Jsonb(c.metadata), self._vector(v)) for c, v in zip(chunks, vectors)]
        with self.connection.transaction(), self.connection.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", ("day6:" + strategy,))
            cur.execute("DELETE FROM day6_chunks WHERE strategy=%s", (strategy,))
            if rows:
                cur.executemany("INSERT INTO day6_chunks VALUES (%s,%s,%s,%s,%s,%s,%s,%s::vector)", rows)

    def count(self, strategy):
        with self.connection.cursor() as cur:
            cur.execute("SELECT count(*) FROM day6_chunks WHERE strategy=%s", (strategy,))
            return cur.fetchone()[0]

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        filters = filters or {}
        if set(filters) - {"chapter"} or any(type(v) is not int or not 1 <= v <= 5 for v in filters.values()):
            raise ValueError("Only integer chapter filters 1–5 are supported.")
        vector = self._vector(query_vector)
        with self.connection.cursor() as cur:
            cur.execute("""SELECT chunk_id,document_id,source,page,text,metadata,
                1-(embedding <=> %s::vector) AS score FROM day6_chunks
                WHERE strategy=%s AND metadata @> %s
                ORDER BY embedding <=> %s::vector,chunk_id LIMIT %s""",
                (vector, strategy, Jsonb(filters), vector, top_k))
            keys = ["chunk_id", "document_id", "source", "page", "text", "metadata", "score"]
            return [dict(zip(keys, row)) for row in cur.fetchall()]
