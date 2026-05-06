import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration

# =============================================================================
# SET TRANSFORMER ENCODER
# =============================================================================
class MAB(nn.Module):
    """Multihead Attention Block."""
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.ln0 = nn.LayerNorm(dim_V) if ln else nn.Identity()
        self.ln1 = nn.LayerNorm(dim_V) if ln else nn.Identity()

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / torch.sqrt(torch.tensor(self.dim_V, dtype=torch.float32)), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = self.ln0(O)
        O = O + self.fc_o(O).relu()
        return self.ln1(O)

class ISAB(nn.Module):
    """Induced Set Attention Block."""
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class SetTransformerEncoder(nn.Module):
    """Permutation-invariant encoder using Set Transformer blocks."""
    def __init__(self, dim_input, dim_model, num_heads, num_inds, num_layers, dropout=0.0, max_seq_len=128):
        super().__init__()
        self.input_proj = nn.Linear(dim_input, dim_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_len, dim_model))
        self.layers = nn.ModuleList([
            ISAB(dim_model, dim_model, num_heads, num_inds, ln=True) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input_proj(x) + self.pos_encoder[:, :x.size(1), :]
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        return x

# =============================================================================
# SLOT BOTTLENECK
# =============================================================================
class SlotBottleneck(nn.Module):
    """Cross-attention bottleneck that compresses set features into fixed slots."""
    def __init__(self, dim_model, num_slots, num_heads, dropout=0.0):
        super().__init__()
        self.slots_init = nn.Parameter(torch.zeros(1, num_slots, dim_model))
        nn.init.xavier_uniform_(self.slots_init)
        self.cross_attn = MAB(dim_model, dim_model, dim_model, num_heads, ln=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        slots = self.slots_init.repeat(x.size(0), 1, 1)
        slots = self.cross_attn(slots, x)
        return self.dropout(slots)

# =============================================================================
# T5 CROSS-ATTENTION DECODER WRAPPER
# =============================================================================
class SlotCrossProjector(nn.Module):
    """Projects slots to T5 d_model space for native cross-attention."""
    def __init__(self, slot_dim, t5_dim, dropout=0.0):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(slot_dim, t5_dim),
            nn.LayerNorm(t5_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, slots):
        return self.proj(slots)

class SlotCrossDecoder(nn.Module):
    """Wraps T5 to use projected slots as encoder_outputs."""
    def __init__(self, t5_name="t5-small", slot_dim=192, freeze_t5=True):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_name)
        self.t5_dim = self.t5.config.d_model
        self.projector = SlotCrossProjector(slot_dim, self.t5_dim, dropout=0.0)
        
        if freeze_t5:
            for param in self.t5.parameters():
                param.requires_grad = False
            self.t5.eval()

    def forward(self, slots, decoder_input_ids=None, labels=None, attention_mask=None):
        encoder_hidden = self.projector(slots)
        encoder_outputs = (encoder_hidden,)
        
        outputs = self.t5(
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            return_dict=True
        )
        return outputs.logits, slots

# =============================================================================
# MAIN MODEL
# =============================================================================
class EncodeThought(nn.Module):
    """SetEncoder -> SlotBottleneck -> T5 Cross-Attention Decoder."""
    def __init__(self, input_dim, dim_model, num_heads, num_encoder_layers, num_inds,
                 num_slots, max_seq_len, t5_name="t5-small", dropout_slots=0.0, freeze_t5=True):
        super().__init__()
        self.encoder = SetTransformerEncoder(
            dim_input=input_dim, dim_model=dim_model, num_heads=num_heads,
            num_inds=num_inds, num_layers=num_encoder_layers, dropout=0.0, max_seq_len=max_seq_len
        )
        self.bottleneck = SlotBottleneck(dim_model, num_slots, num_heads, dropout=0.0)
        self.slot_dropout = nn.Dropout(dropout_slots)
        self.decoder = SlotCrossDecoder(t5_name, slot_dim=dim_model, freeze_t5=freeze_t5)
        self.num_slots = num_slots

    def forward(self, x, decoder_input_ids=None, labels=None, attention_mask=None):
        encoded = self.encoder(x)
        slots = self.slot_dropout(self.bottleneck(encoded))
        logits, slots = self.decoder(slots, decoder_input_ids=decoder_input_ids, labels=labels, attention_mask=attention_mask)
        return logits, slots

    @torch.no_grad()
    def get_slots(self, x):
        return self.bottleneck(self.encoder(x))
