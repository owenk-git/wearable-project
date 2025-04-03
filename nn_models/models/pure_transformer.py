import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------------------
# Helpers for Rotary Positional Embedding (RoPE)
# -------------------------------------------------------------------------
def get_rotary_embedding(seq_len, dim, device):
    """
    Returns cos_emb and sin_emb of shape [1, 1, seq_len, dim],
    used to perform the RoPE transformation on Q/K.
    """
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # [seq_len, dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)       # [seq_len, dim]
    cos_emb = emb.cos()[None, None, :, :]         # [1, 1, seq_len, dim]
    sin_emb = emb.sin()[None, None, :, :]         # [1, 1, seq_len, dim]
    return cos_emb, sin_emb

def rotate_every_two(x):
    """
    Splits x into even/odd channels, rotates them:
    x[..., ::2], x[..., 1::2].
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    # Rotate: (x1, x2) -> ( -x2, x1 )
    x_rotated = torch.stack((-x2, x1), dim=-1)
    # Merge back into last dimension
    return x_rotated.flatten(-2)

def apply_rope(x, cos, sin):
    """
    x: [batch, num_heads, seq_len, head_dim]
    cos/sin: [1, 1, seq_len, head_dim]
    """
    return x * cos + rotate_every_two(x) * sin


# -------------------------------------------------------------------------
# Multi-Head Attention with RoPE
# -------------------------------------------------------------------------
class MultiHeadAttentionRoPE(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        B, S, _ = x.size()
        qkv = self.qkv_proj(x)  # [B, S, 3*d_model]
        qkv = qkv.reshape(B, S, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: [B, num_heads, S, head_dim]

        # RoPE embeddings
        device = x.device
        cos_emb, sin_emb = get_rotary_embedding(S, self.head_dim, device)
        q = apply_rope(q, cos_emb, sin_emb)
        k = apply_rope(k, cos_emb, sin_emb)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)  # [B, num_heads, S, S]
        context = torch.matmul(attn_probs, v)        # [B, num_heads, S, head_dim]

        # Merge heads
        context = context.transpose(1, 2).reshape(B, S, self.d_model)
        return self.out_proj(context)


# -------------------------------------------------------------------------
# A single Transformer Encoder layer with MHA + FF
# -------------------------------------------------------------------------
class TransformerEncoderLayerRoPE(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=256, dropout=0.1, max_seq_len=200):
        super().__init__()
        self.self_attn = MultiHeadAttentionRoPE(d_model, num_heads, max_seq_len)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # Self-Attn + Residual
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # FFN + Residual
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


# -------------------------------------------------------------------------
# Final Transformer class recognized by "model_type == 'transformer'"
# -------------------------------------------------------------------------
class transformer(nn.Module):
    """
    Minimally replicates your RoPE-based Transformer, 
    so that 'model_type' = 'transformer' can be used.
    """
    def __init__(self,
                 inp_size=[42],
                 outp_size=[18],
                 num_layers=4,
                 d_model=96,
                 num_heads=6,
                 dim_feedforward=256,
                 dropout=0.1,
                 max_seq_len=200,
                 prediction='angle',
                 **kwargs):
        super().__init__()
        # Use the sum of inp_size[0] as the actual input dimension (like conv/lstm do).
        self.input_dim = inp_size[0]
        self.output_dim = outp_size[0]
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_seq_len = max_seq_len

        # 1) Input projection
        self.input_linear = nn.Linear(self.input_dim, d_model)

        # 2) Stacked Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayerRoPE(d_model, num_heads, dim_feedforward, dropout, max_seq_len)
            for _ in range(num_layers)
        ])
        # 3) Output projection
        self.out_linear = nn.Linear(d_model, self.output_dim)

    def forward(self, x):
        """
        x: [batch_size, seq_len, input_dim]
        Returns: [batch_size, seq_len, output_dim]
        """
        out = self.input_linear(x)
        for layer in self.layers:
            out = layer(out)
        out = self.out_linear(out)
        return out