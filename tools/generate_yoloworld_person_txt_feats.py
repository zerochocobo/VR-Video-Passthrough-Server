"""Generate fixed YOLO-World text features for person-like prompts.

This utility intentionally avoids PyTorch, transformers, and tokenizers. It uses
the OpenAI CLIP BPE vocabulary/merges plus a CLIP ViT-B/32 text ONNX model once
with ONNX Runtime.

Expected output:

    models/person_txt_feats.npy

with default shape:

    (1, 5, 512)

The 512-wide feature is required by the local YOLO-World ONNX files. The
num_classes axis is dynamic in the local YOLO-World ONNX files, so several
person-like prompts can be used for better recall.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from urllib.error import URLError, HTTPError
from urllib.request import urlretrieve

import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_URL = (
    "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/"
    "onnx/text_model_quantized.onnx"
)
DEFAULT_CACHE_PATH = ROOT / "runtime_cache" / "clip_text_onnx" / "clip-vit-base-patch32_text_model_quantized.onnx"
DEFAULT_VOCAB_URL = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/vocab.json"
DEFAULT_MERGES_URL = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/merges.txt"
DEFAULT_VOCAB_PATH = ROOT / "runtime_cache" / "clip_text_onnx" / "clip-vit-base-patch32_vocab.json"
DEFAULT_MERGES_PATH = ROOT / "runtime_cache" / "clip_text_onnx" / "clip-vit-base-patch32_merges.txt"
DEFAULT_BPE_URL = "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz"
DEFAULT_BPE_PATH = ROOT / "runtime_cache" / "clip_text_onnx" / "bpe_simple_vocab_16e6.txt.gz"
DEFAULT_OUTPUT = ROOT / "models" / "person_txt_feats.npy"
DEFAULT_CLASSES = ["person", "human", "man", "woman", "pedestrian"]


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"[download] {url}")
    urlretrieve(url, tmp)
    tmp.replace(dest)


def _bytes_to_unicode() -> dict[int, str]:
    visible = list(range(ord("!"), ord("~") + 1))
    visible += list(range(161, 173))
    visible += list(range(174, 256))
    chars = visible[:]
    extra = 0
    for byte in range(256):
        if byte not in visible:
            visible.append(byte)
            chars.append(256 + extra)
            extra += 1
    return dict(zip(visible, [chr(n) for n in chars]))


class ClipBpeTokenizer:
    def __init__(self, encoder: dict[str, int], merges: list[tuple[str, str]]) -> None:
        self.encoder = encoder
        self.byte_encoder = _bytes_to_unicode()
        self.bpe_ranks = {pair: idx for idx, pair in enumerate(merges)}
        self.cache: dict[str, str] = {}

    @classmethod
    def from_vocab_merges(cls, vocab_path: Path, merges_path: Path) -> "ClipBpeTokenizer":
        encoder = json.loads(vocab_path.read_text(encoding="utf-8"))
        merge_pairs: list[tuple[str, str]] = []
        for line in merges_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 2:
                merge_pairs.append((parts[0], parts[1]))
        return cls(encoder, merge_pairs)

    @classmethod
    def from_openai_bpe(cls, bpe_path: Path) -> "ClipBpeTokenizer":
        byte_encoder = _bytes_to_unicode()
        vocab = list(byte_encoder.values())
        vocab = vocab + [item + "</w>" for item in vocab]
        lines = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = [tuple(line.split()) for line in lines[1:49152 - 256 - 2 + 1] if line]
        vocab = vocab + ["".join(pair) for pair in merges]
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])
        return cls(dict(zip(vocab, range(len(vocab)))), merges)

    @staticmethod
    def _pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
        return set(zip(word, word[1:]))

    def _bpe(self, token: str) -> str:
        cached = self.cache.get(token)
        if cached is not None:
            return cached
        if not token:
            return ""
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = self._pairs(word)
        if not pairs:
            out = token + "</w>"
            self.cache[token] = out
            return out
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: list[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = self._pairs(word)
        out = " ".join(word)
        self.cache[token] = out
        return out

    def encode(self, text: str) -> list[int]:
        text = " ".join(text.lower().strip().split())
        token_ids: list[int] = []
        for raw in text.split(" "):
            if not raw:
                continue
            token = "".join(self.byte_encoder[b] for b in raw.encode("utf-8"))
            for piece in self._bpe(token).split(" "):
                token_ids.append(int(self.encoder[piece]))
        return token_ids

    def batch_input_ids(self, texts: list[str], context_length: int = 77) -> np.ndarray:
        input_ids = np.zeros((len(texts), context_length), dtype=np.int64)
        for row, text in enumerate(texts):
            seq = [49406, *self.encode(text), 49407]
            if len(seq) > context_length:
                seq = seq[:context_length]
                seq[-1] = 49407
            input_ids[row, :len(seq)] = seq
        return input_ids


def _run_text_model(model_path: Path, input_ids: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_names = {meta.name for meta in session.get_inputs()}
    feeds: dict[str, np.ndarray] = {"input_ids": input_ids}
    if "attention_mask" in input_names:
        attention_mask = (input_ids != 0).astype(np.int64)
        feeds["attention_mask"] = attention_mask

    outputs = session.run(None, feeds)
    output_names = [meta.name for meta in session.get_outputs()]
    if "text_embeds" in output_names:
        embeds = outputs[output_names.index("text_embeds")]
    else:
        candidates = [
            (name, value)
            for name, value in zip(output_names, outputs)
            if getattr(value, "ndim", 0) == 2 and value.shape[-1] == 512
        ]
        if not candidates:
            shapes = [(name, getattr(value, "shape", None)) for name, value in zip(output_names, outputs)]
            raise RuntimeError(f"no 2-D 512-wide text embedding output found; outputs={shapes}")
        print(f"[warn] text_embeds output not found; using {candidates[0][0]}")
        embeds = candidates[0][1]

    embeds = np.asarray(embeds, dtype=np.float32)
    if embeds.ndim != 2 or embeds.shape[0] != input_ids.shape[0] or embeds.shape[1] != 512:
        raise RuntimeError(
            f"unexpected text embedding shape {embeds.shape}; "
            "the local YOLO-World ONNX models require CLIP ViT-B/32 512-wide text features"
        )
    norm = np.linalg.norm(embeds, axis=-1, keepdims=True)
    return (embeds / np.maximum(norm, 1.0e-12))[None, :, :].astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate models/person_txt_feats.npy for YOLO-World.")
    parser.add_argument("--model", default="", help="existing CLIP ViT-B/32 text_model ONNX path")
    parser.add_argument("--url", default=DEFAULT_MODEL_URL, help="download URL used when --model is omitted")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE_PATH), help="download/cache path for the ONNX text model")
    parser.add_argument("--vocab-url", default=DEFAULT_VOCAB_URL, help="download URL for CLIP vocab.json")
    parser.add_argument("--merges-url", default=DEFAULT_MERGES_URL, help="download URL for CLIP merges.txt")
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB_PATH), help="cache path for CLIP vocab.json")
    parser.add_argument("--merges", default=str(DEFAULT_MERGES_PATH), help="cache path for CLIP merges.txt")
    parser.add_argument("--bpe-url", default=DEFAULT_BPE_URL, help="fallback OpenAI CLIP BPE download URL")
    parser.add_argument("--bpe", default=str(DEFAULT_BPE_PATH), help="cache path for OpenAI CLIP BPE gzip")
    parser.add_argument(
        "--classes",
        default=",".join(DEFAULT_CLASSES),
        help="comma-separated text prompts to encode; all are treated as person-like foreground classes",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT), help="output .npy path")
    parser.add_argument("--no-download", action="store_true", help="fail if the ONNX model is not already present")
    args = parser.parse_args(argv)

    model_path = Path(args.model).resolve() if args.model else Path(args.cache).resolve()
    if not model_path.exists():
        if args.no_download:
            raise FileNotFoundError(f"CLIP text ONNX model not found: {model_path}")
        _download(args.url, model_path)

    class_names = [item.strip() for item in args.classes.split(",") if item.strip()]
    if not class_names:
        raise RuntimeError("at least one class prompt is required")
    vocab_path = Path(args.vocab).resolve()
    merges_path = Path(args.merges).resolve()
    bpe_path = Path(args.bpe).resolve()
    tokenizer = None
    if vocab_path.exists() and merges_path.exists():
        tokenizer = ClipBpeTokenizer.from_vocab_merges(vocab_path, merges_path)
    elif not args.no_download:
        try:
            if not vocab_path.exists():
                _download(args.vocab_url, vocab_path)
            if not merges_path.exists():
                _download(args.merges_url, merges_path)
            tokenizer = ClipBpeTokenizer.from_vocab_merges(vocab_path, merges_path)
        except (URLError, HTTPError, TimeoutError, OSError) as exc:
            print(f"[warn] failed to fetch Hugging Face tokenizer files: {exc}")
    if tokenizer is None:
        if not bpe_path.exists():
            if args.no_download:
                raise FileNotFoundError(
                    f"CLIP tokenizer files not found: {vocab_path}, {merges_path}, or {bpe_path}"
                )
            _download(args.bpe_url, bpe_path)
        tokenizer = ClipBpeTokenizer.from_openai_bpe(bpe_path)
    person_tokens = tokenizer.encode("person")
    if person_tokens != [2533]:
        raise RuntimeError(f"CLIP tokenizer self-check failed: person -> {person_tokens}, expected [2533]")
    input_ids = tokenizer.batch_input_ids(class_names)
    txt_feats = _run_text_model(model_path, input_ids)
    if txt_feats.shape != (1, len(class_names), 512):
        raise RuntimeError(f"unexpected txt_feats shape: {txt_feats.shape}")
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, txt_feats)
    print(f"[saved] {out}")
    print(f"[classes] {class_names}")
    print(f"[tokens] {[tokenizer.encode(name) for name in class_names]}")
    print(f"[shape] {txt_feats.shape} dtype={txt_feats.dtype} norms={np.linalg.norm(txt_feats[0], axis=1)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
