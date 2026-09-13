from __future__ import annotations

import math
from typing import Iterator
import torch
import torch.nn.functional as F

from .model import NexusModel
from .tokenizer import SimpleTokenizer


def apply_repetition_penalty(
    logits: torch.Tensor,
    context_tokens: torch.Tensor,
    penalty: float = 1.0,
) -> torch.Tensor:
    """Applies multiplicative repetition penalty to previously generated tokens."""
    if penalty == 1.0 or context_tokens.numel() == 0:
        return logits

    for b in range(logits.shape[0]):
        unique_tokens = torch.unique(context_tokens[b])
        scores = logits[b, unique_tokens]
        # Standard formula: if score < 0, score * penalty; if score > 0, score / penalty
        penalized = torch.where(scores < 0.0, scores * penalty, scores / penalty)
        logits[b, unique_tokens] = penalized
    return logits


def apply_no_repeat_ngram(
    logits: torch.Tensor,
    context_tokens: torch.Tensor,
    no_repeat_ngram_size: int = 0,
) -> torch.Tensor:
    """Prevents exact repetition of n-grams of a given size."""
    if no_repeat_ngram_size <= 0:
        return logits

    for b in range(logits.shape[0]):
        tokens = context_tokens[b].tolist()
        if len(tokens) < no_repeat_ngram_size:
            continue
        prefix = tuple(tokens[-(no_repeat_ngram_size - 1):])
        banned = set()
        for i in range(len(tokens) - no_repeat_ngram_size + 1):
            ngram = tuple(tokens[i : i + no_repeat_ngram_size])
            if ngram[:-1] == prefix:
                banned.add(ngram[-1])
        if banned:
            banned_tensor = torch.tensor(list(banned), dtype=torch.long, device=logits.device)
            logits[b, banned_tensor] = -float("inf")
    return logits


def apply_top_k(logits: torch.Tensor, top_k: int | None = None) -> torch.Tensor:
    """Filters logits to retain only the top_k most likely tokens."""
    if top_k is None or top_k <= 0 or top_k >= logits.size(-1):
        return logits
    k = min(top_k, logits.size(-1))
    top_values, _ = torch.topk(logits, k, dim=-1)
    min_value = top_values[..., -1, None]
    return torch.where(logits < min_value, torch.tensor(-float("inf"), device=logits.device), logits)


def apply_top_p(logits: torch.Tensor, top_p: float | None = None) -> torch.Tensor:
    """Filters logits using Nucleus (top-p) sampling."""
    if top_p is None or top_p <= 0.0 or top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above top_p (keep the first token exceeding it)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
    return logits.masked_fill(indices_to_remove, -float("inf"))


def apply_min_p(logits: torch.Tensor, min_p: float | None = None) -> torch.Tensor:
    """Filters logits using Min-P sampling (removes tokens with prob < min_p * max_prob)."""
    if min_p is None or min_p <= 0.0 or min_p >= 1.0:
        return logits

    probs = torch.softmax(logits, dim=-1)
    max_probs, _ = torch.max(probs, dim=-1, keepdim=True)
    threshold = min_p * max_probs
    mask = probs < threshold
    return logits.masked_fill(mask, -float("inf"))


def sample_next_token(
    logits: torch.Tensor,
    context_tokens: torch.Tensor,
    temperature: float = 0.7,
    top_k: int | None = 50,
    top_p: float | None = 0.9,
    min_p: float | None = 0.05,
    repetition_penalty: float = 1.15,
    no_repeat_ngram_size: int = 0,
    do_sample: bool = True,
) -> torch.Tensor:
    """Processes logits and samples the next token ID."""
    # Ensure logits are 2D: (batch_size, vocab_size)
    if logits.dim() == 3:
        logits = logits[:, -1, :]

    # Greedy decoding when temperature is very low or sampling is disabled
    if not do_sample or temperature <= 1e-5:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # 1. Repetition penalty
    if repetition_penalty > 1.0:
        logits = apply_repetition_penalty(logits, context_tokens, repetition_penalty)

    # 2. No-repeat n-grams
    if no_repeat_ngram_size > 0:
        logits = apply_no_repeat_ngram(logits, context_tokens, no_repeat_ngram_size)

    # 3. Temperature scaling
    logits = logits / max(temperature, 1e-5)

    # 4. Top-K filtering
    if top_k is not None and top_k > 0:
        logits = apply_top_k(logits, top_k)

    # 5. Top-P (Nucleus) filtering
    if top_p is not None and top_p < 1.0:
        logits = apply_top_p(logits, top_p)

    # 6. Min-P filtering
    if min_p is not None and min_p > 0.0:
        logits = apply_min_p(logits, min_p)

    # 7. Probability distribution & multinomial sampling
    probs = torch.softmax(logits, dim=-1)

    # Fallback to greedy if any probabilities are NaN / invalid
    if torch.isnan(probs).any() or (probs.sum(dim=-1) <= 0).any():
        return torch.argmax(logits, dim=-1, keepdim=True)

    return torch.multinomial(probs, num_samples=1)


def generate_stream(
    model: NexusModel,
    tokenizer: SimpleTokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    min_new_tokens: int = 0,
    temperature: float = 0.7,
    top_k: int | None = 50,
    top_p: float | None = 0.9,
    min_p: float | None = 0.05,
    repetition_penalty: float = 1.15,
    no_repeat_ngram_size: int = 0,
    do_sample: bool = True,
    use_cache: bool = True,
    stop_strings: list[str] | None = None,
    device: torch.device | str | None = None,
) -> Iterator[str]:
    """Yields generated text incrementally token-by-token (streaming mode with KV-caching)."""
    model.eval()

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = [tokenizer.bos_token_id]

    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    all_tokens = input_ids.clone()
    generated_ids: list[int] = []

    if stop_strings is None:
        stop_strings = ["\nПользователь:", "\nUser:", "\nHuman:", "\n###", "<eos>"]

    past_key_values = None

    with torch.no_grad():
        for step in range(max_new_tokens):
            if use_cache:
                if past_key_values is None:
                    # Prefill prompt
                    out = model(input_ids, use_cache=True)
                    logits, past_key_values = out
                else:
                    # Single new token forward with cached KV
                    out = model(
                        input_ids[:, -1:],
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    logits, past_key_values = out
            else:
                logits = model(all_tokens)

            next_token = sample_next_token(
                logits=logits,
                context_tokens=all_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                do_sample=do_sample,
            )

            token_id = next_token.item()
            generated_ids.append(token_id)
            all_tokens = torch.cat([all_tokens, next_token], dim=1)
            input_ids = next_token

            # Check EOS token
            if token_id == tokenizer.eos_token_id and step >= min_new_tokens:
                break

            # Current decoded string
            current_text = tokenizer.decode(generated_ids)

            # Check stop strings
            stopped = False
            for stop_str in stop_strings:
                if stop_str in current_text:
                    current_text = current_text.split(stop_str)[0]
                    stopped = True
                    break

            yield current_text

            if stopped:
                break


def generate_text(
    model: NexusModel,
    tokenizer: SimpleTokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    min_new_tokens: int = 0,
    temperature: float = 0.7,
    top_k: int | None = 50,
    top_p: float | None = 0.9,
    min_p: float | None = 0.05,
    repetition_penalty: float = 1.15,
    no_repeat_ngram_size: int = 0,
    do_sample: bool = True,
    use_cache: bool = True,
    stop_strings: list[str] | None = None,
    return_full_text: bool = False,
    device: torch.device | str | None = None,
) -> str:
    """Generate text from a prompt with KV-cache and modern sampling strategies."""
    final_output = ""
    for chunk in generate_stream(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        do_sample=do_sample,
        use_cache=use_cache,
        stop_strings=stop_strings,
        device=device,
    ):
        final_output = chunk

    if return_full_text:
        return prompt + final_output
    return final_output
