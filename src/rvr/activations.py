"""Model loading and activation extraction (§4.1, §4.5).

Read positions are recorded by the generator as CHARACTER offsets, because
tokenization is a property of the model, not of the stimulus. This module maps
them to token indices with the tokenizer's offset mapping, so the same rendered
context can be read at the same semantic position under any tokenizer.

Hardware note. On a 16GB T4 the 8B model does not fit in fp16 (~16.1GB of
weights before activations), so `load_model` defaults to 8-bit. int8 perturbs
the activations the probes read far less than 4-bit does, but it is still
quantization: state it as a limitation, and keep the quantization mode in
results.json so numbers from different runs are never silently pooled.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

import numpy as np

DEFAULT_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
Quant = Literal["8bit", "4bit", "fp16", "bf16"]


# --------------------------------------------------------------------------
# read positions
# --------------------------------------------------------------------------


def char_to_token(offsets: Sequence[tuple[int, int]], char_offset: int) -> int:
    """Index of the last token beginning at or before `char_offset`.

    The primary read position is the end of an assistant turn, immediately
    before a tool call (§4.5), so we want the final token of that span rather
    than the first token of whatever follows.
    """
    lo, hi, best = 0, len(offsets) - 1, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if offsets[mid][0] <= char_offset:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def resolve_positions(text: str, read_positions: dict, tokenizer,
                      want: Iterable[str] = ("primary", "final")) -> dict[str, int | list[int]]:
    """Map a context's char-offset read positions to token indices."""
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    n = len(offsets)
    out: dict[str, int | list[int]] = {}
    for key in want:
        if key not in read_positions:
            continue
        v = read_positions[key]
        if isinstance(v, list):
            out[key] = [min(char_to_token(offsets, c), n - 1) for c in v]
        else:
            out[key] = min(char_to_token(offsets, v), n - 1)
    out["n_tokens"] = n
    return out


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------


def load_model(model_name: str = DEFAULT_MODEL, quant: Quant = "8bit"):
    """Load model and tokenizer. Returns (model, tokenizer, info)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs: dict = {"low_cpu_mem_usage": True}
    if quant in ("8bit", "4bit"):
        from transformers import BitsAndBytesConfig

        if quant == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        kwargs["device_map"] = "auto"
    else:
        # T4 is Turing: fp16 only, bf16 unsupported.
        kwargs["torch_dtype"] = torch.float16 if quant == "fp16" else torch.bfloat16
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    info = {
        "model": model_name,
        "quantization": quant,
        "n_layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "device": str(next(model.parameters()).device),
        "torch_dtype": str(getattr(model.config, "torch_dtype", "n/a")),
    }
    return model, tok, info


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------


@dataclass
class ActivationSet:
    """Activations at one read position, for every layer.

    X has shape (n_items, n_layers + 1, hidden). Index 0 on the layer axis is
    the embedding output, so layer L lives at index L.
    """

    X: np.ndarray
    keys: list[str] = field(default_factory=list)     # item identifiers
    read_position: str = "primary"
    info: dict = field(default_factory=dict)

    def layer(self, i: int) -> np.ndarray:
        return self.X[:, i, :]

    @property
    def n_layers(self) -> int:
        return self.X.shape[1]


def extract(model, tokenizer, texts: Sequence[str], token_indices: Sequence[int],
            keys: Sequence[str] | None = None, read_position: str = "primary",
            batch_size: int = 4, max_length: int = 4096,
            progress: bool = True) -> ActivationSet:
    """Run forward passes and keep the hidden state at one token per item.

    Hidden states are reduced to the read position immediately: keeping the full
    sequence for every layer would be ~135MB per 500-token item on an 8B model.
    """
    import torch

    n = len(texts)
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            chunk = list(texts[start:start + batch_size])
            idxs = list(token_indices[start:start + batch_size])
            enc = tokenizer(chunk, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_length,
                            add_special_tokens=False)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            res = model(**enc, output_hidden_states=True)
            # hidden_states: tuple(n_layers+1) of (batch, seq, hidden)
            hs = torch.stack(res.hidden_states, dim=1)     # (b, L+1, seq, hidden)
            for b, ti in enumerate(idxs):
                # left-padding would shift indices; tokenizer pads right by default,
                # and indices were computed on the unpadded text, so clamp only.
                t = min(ti, hs.shape[2] - 1)
                out.append(hs[b, :, t, :].float().cpu().numpy())
            del res, hs, enc
            gc.collect()
            torch.cuda.empty_cache()
            if progress:
                done = min(start + batch_size, n)
                print(f"\r  extract {done}/{n}", end="", flush=True)
    if progress:
        print()
    X = np.stack(out)
    return ActivationSet(X=X, keys=list(keys or range(n)), read_position=read_position)
