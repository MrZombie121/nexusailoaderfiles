from __future__ import annotations

import torch

from .model import NexusModel
from .tokenizer import SimpleTokenizer


def generate_text(
    model: NexusModel,
    tokenizer: SimpleTokenizer,
    prompt: str,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> str:
    """Generate text from a prompt using greedy or top-k sampling."""
    model.eval()
    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = [tokenizer.bos_token_id]

    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            logits = logits[:, -1, :] / temperature

            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, top_k)
                min_values = values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_values, torch.tensor(-float("Inf"), device=logits.device), logits)

            probabilities = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(input_ids[0].tolist())
