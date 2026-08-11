from __future__ import annotations

import torch
import torch.nn as nn

from .embedding import PositionalEmbedding, TokenEmbedding
from .transformer import TransformerBlock


class NexusModel(nn.Module):
    """Decoder-only transformer model with embedding and unmasked causal attention."""

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 512,
        intermediate_size: int = 2048,
        num_layers: int = 8,
        num_heads: int = 8,
        num_key_value_heads: int | None = None,
        max_position_embeddings: int = 2048,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.token_embedding = TokenEmbedding(vocab_size, hidden_size)
        self.position_embedding = PositionalEmbedding(max_position_embeddings, hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(hidden_size, intermediate_size, num_heads, num_key_value_heads, dropout) for _ in range(num_layers)]
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Tie weights: связываем веса token_embedding и lm_head (стандарт для моделей типа Llama)
        # Это экономит vocab_size * hidden_size параметров (например, ~131 МБ для 6B модели)
        self.lm_head.weight = self.token_embedding.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_length = input_ids.size(1)
        x = self.token_embedding(input_ids) + self.position_embedding(seq_length, input_ids.device)

        if self.gradient_checkpointing and self.training:
            if x.is_floating_point() and not x.requires_grad:
                x.requires_grad_(True)

        from torch.utils.checkpoint import checkpoint

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        x = self.layer_norm(x)
        return self.lm_head(x)

    def generate(
        self,
        tokenizer: "SimpleTokenizer",
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> str:
        from .tokenizer import SimpleTokenizer

        if not isinstance(tokenizer, SimpleTokenizer):
            raise ValueError("tokenizer must be an instance of SimpleTokenizer")

        self.eval()
        tokens = tokenizer.encode(prompt)
        if not tokens:
            tokens = [tokenizer.bos_token_id]

        input_ids = torch.tensor(tokens, dtype=torch.long, device=next(self.parameters()).device).unsqueeze(0)
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self(input_ids)
                logits = logits[:, -1, :] / temperature

                if top_k is not None and top_k > 0:
                    top_values, _ = torch.topk(logits, top_k)
                    min_value = top_values[:, -1].unsqueeze(-1)
                    logits = torch.where(logits < min_value, torch.tensor(-float("Inf"), device=logits.device), logits)

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)

                if next_token.item() == tokenizer.eos_token_id:
                    break

        return tokenizer.decode(input_ids[0].tolist())


NexusModel.__annotations__["SimpleTokenizer"] = "SimpleTokenizer"
