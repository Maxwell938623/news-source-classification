from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable, List

import torch
from torch import nn
from tokenizers import Tokenizer
from transformers import DistilBertConfig, DistilBertForSequenceClassification

MAX_LENGTH = 96
VOCAB_BLOB_CAP = 300_000
TOKENIZER_JSON_BLOB_CAP = 1_000_000


class _WordPieceTokenizer:
    def __init__(self, vocab_text: str) -> None:
        tokens = [line.strip() for line in vocab_text.splitlines() if line.strip()]
        self.vocab = {tok: idx for idx, tok in enumerate(tokens)}
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.pad_token = "[PAD]"
        self.unk_id = self.vocab[self.unk_token]
        self.cls_id = self.vocab[self.cls_token]
        self.sep_id = self.vocab[self.sep_token]
        self.pad_id = self.vocab[self.pad_token]

    def _strip_accents(self, text: str) -> str:
        text = unicodedata.normalize("NFD", text)
        return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")

    def _basic_tokenize(self, text: str) -> list[str]:
        text = self._strip_accents(str(text).lower())
        text = re.sub(r"\s+", " ", text).strip()
        return re.findall(r"[a-z0-9]+|[^\w\s]", text)

    def _wordpiece(self, token: str) -> list[str]:
        if len(token) > 100:
            return [self.unk_token]
        pieces: list[str] = []
        start = 0
        while start < len(token):
            end = len(token)
            cur = None
            while start < end:
                sub = token[start:end] if start == 0 else "##" + token[start:end]
                if sub in self.vocab:
                    cur = sub
                    break
                end -= 1
            if cur is None:
                return [self.unk_token]
            pieces.append(cur)
            start = end
        return pieces

    def encode_batch(self, texts: list[str], max_length: int = MAX_LENGTH) -> dict[str, torch.Tensor]:
        all_ids: list[list[int]] = []
        all_masks: list[list[int]] = []
        for text in texts:
            pieces: list[str] = []
            for tok in self._basic_tokenize(text):
                pieces.extend(self._wordpiece(tok))

            ids = [self.cls_id]
            ids.extend(self.vocab.get(piece, self.unk_id) for piece in pieces[: max_length - 2])
            ids.append(self.sep_id)
            mask = [1] * len(ids)

            pad_n = max_length - len(ids)
            if pad_n > 0:
                ids.extend([self.pad_id] * pad_n)
                mask.extend([0] * pad_n)

            all_ids.append(ids[:max_length])
            all_masks.append(mask[:max_length])

        return {
            "input_ids": torch.tensor(all_ids, dtype=torch.long),
            "attention_mask": torch.tensor(all_masks, dtype=torch.long),
        }


class Model(nn.Module):
    """
    DistilBERT submission wrapper.

    The evaluator instantiates this class, loads model.pt as a state_dict, and
    then calls predict(batch). model.pt includes both DistilBERT weights and the
    tokenizer vocabulary bytes used by the lightweight WordPiece tokenizer.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = DistilBertConfig(
            vocab_size=30522,
            max_position_embeddings=512,
            n_layers=6,
            n_heads=12,
            dim=768,
            hidden_dim=3072,
            dropout=0.1,
            attention_dropout=0.1,
            seq_classif_dropout=0.2,
            num_labels=2,
            pad_token_id=0,
        )
        self.model = DistilBertForSequenceClassification(config)
        self.register_buffer("vocab_blob", torch.zeros(VOCAB_BLOB_CAP, dtype=torch.uint8))
        self.register_buffer("vocab_size", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("tokenizer_json_blob", torch.zeros(TOKENIZER_JSON_BLOB_CAP, dtype=torch.uint8))
        self.register_buffer("tokenizer_json_size", torch.tensor(0, dtype=torch.int64))
        self._tokenizer: _WordPieceTokenizer | None = None
        self._hf_tokenizer: Tokenizer | None = None

    def _ensure_hf_tokenizer(self) -> Tokenizer | None:
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer
        n = int(self.tokenizer_json_size.item())
        if n <= 0:
            return None
        tokenizer_json = self.tokenizer_json_blob[:n].cpu().numpy().tobytes().decode("utf-8")
        self._hf_tokenizer = Tokenizer.from_str(tokenizer_json)
        return self._hf_tokenizer

    def _ensure_wordpiece_tokenizer(self) -> _WordPieceTokenizer:
        if self._tokenizer is not None:
            return self._tokenizer
        n = int(self.vocab_size.item())
        if n <= 0:
            raise RuntimeError("Tokenizer vocabulary was not loaded from model.pt.")
        vocab_text = self.vocab_blob[:n].cpu().numpy().tobytes().decode("utf-8")
        self._tokenizer = _WordPieceTokenizer(vocab_text)
        return self._tokenizer

    def eval(self) -> "Model":
        super().eval()
        self.model.eval()
        return self

    def predict(self, batch: Iterable[Any]) -> List[int]:
        texts = [str(x) for x in batch]
        hf_tokenizer = self._ensure_hf_tokenizer()
        if hf_tokenizer is not None:
            encodings = hf_tokenizer.encode_batch(texts)
            all_ids: list[list[int]] = []
            all_masks: list[list[int]] = []
            pad_id = 0
            for enc in encodings:
                ids = enc.ids[:MAX_LENGTH]
                mask = enc.attention_mask[:MAX_LENGTH]
                pad_n = MAX_LENGTH - len(ids)
                if pad_n > 0:
                    ids.extend([pad_id] * pad_n)
                    mask.extend([0] * pad_n)
                all_ids.append(ids)
                all_masks.append(mask)
            encoded = {
                "input_ids": torch.tensor(all_ids, dtype=torch.long),
                "attention_mask": torch.tensor(all_masks, dtype=torch.long),
            }
        else:
            tokenizer = self._ensure_wordpiece_tokenizer()
            encoded = tokenizer.encode_batch(texts, max_length=MAX_LENGTH)
        device = next(self.model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        self.model.eval()
        with torch.no_grad():
            logits = self.model(**encoded).logits
        return logits.argmax(dim=-1).cpu().tolist()


def get_model() -> Model:
    return Model()
