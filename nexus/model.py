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

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
        position_ids: torch.Tensor | None = None,
        start_pos: int = 0,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        seq_length = input_ids.size(1)
        device = input_ids.device

        if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None and start_pos == 0:
            start_pos = past_key_values[0][0].shape[2]

        x = self.token_embedding(input_ids) + self.position_embedding(
            seq_length=seq_length,
            start_pos=start_pos,
            position_ids=position_ids,
            device=device,
        )

        if self.gradient_checkpointing and self.training:
            if x.is_floating_point() and not x.requires_grad:
                x.requires_grad_(True)

        present_key_values = [] if use_cache else None

        from torch.utils.checkpoint import checkpoint

        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training and not use_cache:
                x = checkpoint(layer, x, use_reentrant=False, preserve_rng_state=False)
            else:
                if use_cache:
                    x, layer_present = layer(x, past_key_value=layer_past, use_cache=True, attn_mask=attn_mask)
                    present_key_values.append(layer_present)
                else:
                    x = layer(x, past_key_value=None, use_cache=False, attn_mask=attn_mask)

        x = self.layer_norm(x)
        logits = self.lm_head(x)

        if use_cache:
            return logits, present_key_values
        return logits

    def generate(
        self,
        tokenizer: "SimpleTokenizer",
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int | None = 50,
        top_p: float | None = 0.9,
        min_p: float | None = 0.05,
        repetition_penalty: float = 1.15,
        no_repeat_ngram_size: int = 0,
        do_sample: bool = True,
        use_cache: bool = True,
        stop_strings: list[str] | None = None,
        return_full_text: bool = True,
    ) -> str:
        from .generation import generate_text

        return generate_text(
            model=self,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            do_sample=do_sample,
            use_cache=use_cache,
            stop_strings=stop_strings,
            return_full_text=return_full_text,
        )


NexusModel.__annotations__["SimpleTokenizer"] = "SimpleTokenizer"
