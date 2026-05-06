from __future__ import annotations

from typing import Any, Iterable, List

import torch
from tokenizers import Tokenizer
from torch import nn
from transformers import ModernBertConfig, ModernBertForSequenceClassification

MAX_LENGTH = 64
TOKENIZER_JSON_BLOB_CAP = 4_000_000
PAD_TOKEN_ID = 50283


class Model(nn.Module):
    """
    ModernBERT submission wrapper.

    The evaluator instantiates this class, loads model.pt as a state_dict, then
    calls predict(batch). model.pt contains both ModernBERT weights and the
    tokenizer JSON bytes needed for inference.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = ModernBertConfig(
            vocab_size=50368,
            hidden_size=768,
            intermediate_size=1152,
            num_hidden_layers=22,
            num_attention_heads=12,
            max_position_embeddings=8192,
            local_attention=128,
            global_attn_every_n_layers=3,
            global_rope_theta=160000.0,
            local_rope_theta=10000.0,
            hidden_activation="gelu",
            classifier_activation="gelu",
            classifier_pooling="mean",
            classifier_dropout=0.0,
            classifier_bias=False,
            attention_dropout=0.0,
            embedding_dropout=0.0,
            mlp_dropout=0.0,
            attention_bias=False,
            mlp_bias=False,
            norm_bias=False,
            norm_eps=1e-5,
            layer_norm_eps=1e-5,
            decoder_bias=True,
            pad_token_id=PAD_TOKEN_ID,
            bos_token_id=50281,
            cls_token_id=50281,
            eos_token_id=50282,
            sep_token_id=50282,
            num_labels=2,
            id2label={0: "FoxNews", 1: "NBC"},
            label2id={"FoxNews": 0, "NBC": 1},
        )
        self.model = ModernBertForSequenceClassification(config)
        self.register_buffer("tokenizer_json_blob", torch.zeros(TOKENIZER_JSON_BLOB_CAP, dtype=torch.uint8))
        self.register_buffer("tokenizer_json_size", torch.tensor(0, dtype=torch.int64))
        self._tokenizer: Tokenizer | None = None

    def _ensure_tokenizer(self) -> Tokenizer:
        if self._tokenizer is not None:
            return self._tokenizer
        n = int(self.tokenizer_json_size.item())
        if n <= 0:
            raise RuntimeError("Tokenizer JSON was not loaded from model.pt.")
        tokenizer_json = self.tokenizer_json_blob[:n].cpu().numpy().tobytes().decode("utf-8")
        self._tokenizer = Tokenizer.from_str(tokenizer_json)
        return self._tokenizer

    def eval(self) -> "Model":
        super().eval()
        self.model.eval()
        return self

    def predict(self, batch: Iterable[Any]) -> List[int]:
        texts = [str(x) for x in batch]
        tokenizer = self._ensure_tokenizer()
        encodings = tokenizer.encode_batch(texts)

        all_ids: list[list[int]] = []
        all_masks: list[list[int]] = []
        for enc in encodings:
            ids = enc.ids[:MAX_LENGTH]
            mask = enc.attention_mask[:MAX_LENGTH]
            pad_n = MAX_LENGTH - len(ids)
            if pad_n > 0:
                ids.extend([PAD_TOKEN_ID] * pad_n)
                mask.extend([0] * pad_n)
            all_ids.append(ids)
            all_masks.append(mask)

        device = next(self.model.parameters()).device
        inputs = {
            "input_ids": torch.tensor(all_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(all_masks, dtype=torch.long, device=device),
        }
        self.model.eval()
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return logits.argmax(dim=-1).cpu().tolist()


def get_model() -> Model:
    return Model()
