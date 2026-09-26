"""
Sentiment Lite — scoring léxico de polaridad (sin ML pesado / sin deps).
Uso: python sentiment.py "Me encanta este proyecto"
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass


POS = {
    "encanta", "excelente", "genial", "bueno", "buena", "amor", "feliz",
    "perfecto", "gracias", "éxito", "exito", "rápido", "rapido", "claro",
    "útil", "util", "increíble", "increible", "top", "love", "great", "good",
}
NEG = {
    "odio", "malo", "mala", "pésimo", "pesimo", "lento", "error", "falla",
    "horrible", "triste", "difícil", "dificil", "basura", "bug", "crash",
    "hate", "bad", "awful", "slow", "broken",
}


@dataclass
class Result:
    text: str
    score: float
    label: str
    positives: list[str]
    negatives: list[str]
    tokens: int


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-záéíóúñü]+", text.lower())


def analyze(text: str) -> Result:
    tokens = tokenize(text)
    pos_hits = [t for t in tokens if t in POS]
    neg_hits = [t for t in tokens if t in NEG]
    raw = len(pos_hits) - len(neg_hits)
    denom = max(len(tokens), 1)
    score = round(raw / denom, 3)
    if score > 0.05:
        label = "positive"
    elif score < -0.05:
        label = "negative"
    else:
        label = "neutral"
    return Result(text=text, score=score, label=label, positives=pos_hits, negatives=neg_hits, tokens=len(tokens))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentiment léxico ES/EN")
    parser.add_argument("text", nargs="?", help="Texto a analizar")
    parser.add_argument("--file", type=str, help="Archivo con una frase por línea")
    parser.add_argument("--json", action="store_true", help="Salida JSON")
    args = parser.parse_args()

    texts: list[str] = []
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            texts = [ln.strip() for ln in f if ln.strip()]
    elif args.text:
        texts = [args.text]
    else:
        raise SystemExit("Pasa un texto o --file")

    results = [analyze(t) for t in texts]
    if args.json:
        print(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2))
    else:
        for r in results:
            print(f"[{r.label}] score={r.score:+.3f} · tokens={r.tokens}")
            print(f"  {r.text}")
            if r.positives:
                print(f"  + {', '.join(r.positives)}")
            if r.negatives:
                print(f"  - {', '.join(r.negatives)}")


if __name__ == "__main__":
    main()
