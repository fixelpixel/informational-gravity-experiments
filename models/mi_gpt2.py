import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

class MIAttention(nn.Module):
    """
    Простая обёртка: добавляет MI-бонус к attention logits
    """
    def __init__(self, original_attn, mi_matrix, alpha=1.0):
        super().__init__()
        self.original_attn = original_attn
        self.mi_matrix = mi_matrix
        self.alpha = alpha

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
        **kwargs
    ):
        # стандартное GPT2 внимание
        query, key, value = self.original_attn.c_attn(hidden_states).split(
            self.original_attn.split_size, dim=2
        )

        query = self.original_attn._split_heads(query, self.original_attn.num_heads, self.original_attn.head_dim)
        key = self.original_attn._split_heads(key, self.original_attn.num_heads, self.original_attn.head_dim)
        value = self.original_attn._split_heads(value, self.original_attn.num_heads, self.original_attn.head_dim)

        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        attn_weights = attn_weights / (self.original_attn.head_dim ** 0.5)

        # добавляем MI-bias (симметричный)
        if self.mi_matrix is not None:
            vocab_size = self.mi_matrix.shape[0]
            if vocab_size > 0:
                mi_bias = self.mi_matrix.mean()  # простой глобальный сдвиг
                attn_weights = attn_weights + self.alpha * mi_bias

        # causal mask
        if not self.original_attn.is_cross_attention:
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.original_attn.bias[:, :, key_length - query_length:key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = self.original_attn._merge_heads(attn_output, self.original_attn.num_heads, self.original_attn.head_dim)
        attn_output = self.original_attn.c_proj(attn_output)
        attn_output = self.original_attn.resid_dropout(attn_output)

        outputs = (attn_output, layer_past)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


def inject_mi_attention(model: GPT2LMHeadModel, mi_matrix: torch.Tensor, alpha: float = 1.0, layer_idx: int = 6):
    """
    Внедряем MI-внимание в указанный слой GPT2
    """
    print(f"Injecting MI-attention into layer {layer_idx} (alpha={alpha})")
    target_layer = model.transformer.h[layer_idx]
    original_attn = target_layer.attn
    mi_attn = MIAttention(original_attn, mi_matrix, alpha)
    target_layer.attn = mi_attn
    return model
