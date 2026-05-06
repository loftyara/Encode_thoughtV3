import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import gc
import numpy as np
from transformers import T5Tokenizer
from peft import LoraConfig, get_peft_model
from model import EncodeThought
from dataset import StoryEmbeddingDataset, prepare_pinned_dataset

# ================= CONFIGURATION =================
CHECKPOINT_DIR = "../checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best.pt")
DATA_DIR = "../data/processed"
MODEL_FILTER = "bert-mini"
T5_NAME = "t5-small"

MAX_TRAIN_SAMPLES = 20000
MAX_VAL_SAMPLES = 2000
MAX_SEQ_LEN = 128
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 2
EPOCHS = 120
LEARNING_RATE = 1.5e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

NUM_SLOTS = 16
DIM_MODEL = 192

NOISE_START_EPOCH = 3
MAX_NOISE_ALPHA = 0.12
SLOT_ALIGNMENT_WEIGHT = 0.5
EARLY_STOP_PATIENCE = 8

DEVICE = "cuda"
assert torch.cuda.is_available(), "CUDA is required. No CPU fallback."
torch.backends.cudnn.benchmark = True

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ================= SLOT ALIGNMENT LOSS =================
class SlotAlignmentLoss(nn.Module):
    """Aligns projected slots with the semantic mean of target embeddings in T5 space."""
    def __init__(self): super().__init__()
    def forward(self, projected_slots, target_embeds, mask):
        mask_exp = mask.unsqueeze(-1).float()
        valid_count = mask_exp.sum(dim=1).clamp(min=1.0)
        target_mean = (target_embeds * mask_exp).sum(dim=1) / valid_count
        slot_mean = projected_slots.mean(dim=1)
        pred_norm = nn.functional.normalize(slot_mean, p=2, dim=-1)
        tgt_norm = nn.functional.normalize(target_mean, p=2, dim=-1)
        return (1.0 - torch.sum(pred_norm * tgt_norm, dim=-1)).mean()

# ================= MAIN =================
def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"[Device] {DEVICE}")
    
    tokenizer = T5Tokenizer.from_pretrained(T5_NAME)
    
    train_ds = StoryEmbeddingDataset(DATA_DIR, MODEL_FILTER, "train", MAX_TRAIN_SAMPLES, preload=True)
    val_ds = StoryEmbeddingDataset(DATA_DIR, MODEL_FILTER, "val", MAX_VAL_SAMPLES, preload=True)
    X_train, Ids_train, Mask_train = prepare_pinned_dataset(train_ds, MAX_TRAIN_SAMPLES, MAX_SEQ_LEN, tokenizer)
    X_val, Ids_val, Mask_val = prepare_pinned_dataset(val_ds, MAX_VAL_SAMPLES, MAX_SEQ_LEN, tokenizer)
    input_dim = X_train.shape[-1]
    print(f"[Data] Train: {X_train.shape}, Val: {X_val.shape}")
    
    model = EncodeThought(
        input_dim=input_dim, dim_model=DIM_MODEL, num_heads=1,
        num_encoder_layers=1, num_inds=4, num_slots=NUM_SLOTS,
        max_seq_len=MAX_SEQ_LEN, t5_name=T5_NAME,
        dropout_slots=0.0, freeze_t5=True
    ).to(DEVICE)
    print(f"[Model] Initialized. Slots={NUM_SLOTS}, Dim={DIM_MODEL}. Applying Cross-LoRA.")
    
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, target_modules=["q", "k", "v", "o"],
        lora_dropout=0.05, bias="none", task_type="SEQ_2_SEQ_LM"
    )
    model.decoder.t5 = get_peft_model(model.decoder.t5, lora_cfg)
    model.decoder.t5.print_trainable_parameters()
    
    ce_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    align_criterion = SlotAlignmentLoss()
    best_val_loss = float('inf')
    patience_counter = 0
    scaler = torch.amp.GradScaler('cuda')
    gc.disable()
    
    for epoch in range(EPOCHS):
        start = time.time()
        
        noise_alpha = 0.0
        if epoch >= NOISE_START_EPOCH:
            noise_alpha = min(MAX_NOISE_ALPHA, (epoch - NOISE_START_EPOCH) / (EPOCHS - NOISE_START_EPOCH) * MAX_NOISE_ALPHA)
            
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        model.train()
        total_loss, n = 0.0, 0
        indices = torch.randperm(X_train.size(0))
        optimizer.zero_grad(set_to_none=True)
        
        for i in range(0, X_train.size(0), BATCH_SIZE):
            idx = indices[i:i+BATCH_SIZE]
            x_b = X_train[idx].to(DEVICE, non_blocking=True)
            ids_b = Ids_train[idx].to(DEVICE, non_blocking=True)
            m_b = Mask_train[idx].to(DEVICE, non_blocking=True)
            
            decoder_input_ids = ids_b[:, :-1].clone()
            labels = ids_b[:, 1:].clone()
            attn_mask = m_b[:, :-1]
            
            slots = model.get_slots(x_b)
            projected_slots = model.decoder.projector(slots)
            target_embeds = model.decoder.t5.get_input_embeddings()(ids_b)
            
            if noise_alpha > 0.0:
                with torch.no_grad():
                    logits_pred, _ = model(x_b, decoder_input_ids=decoder_input_ids, attention_mask=attn_mask)
                    pred_ids = torch.argmax(logits_pred, dim=-1)
                    
                noise_mask = torch.rand_like(decoder_input_ids, dtype=torch.float) < noise_alpha
                noise_mask = noise_mask & attn_mask
                decoder_input_ids = torch.where(noise_mask, pred_ids, decoder_input_ids)
            
            with torch.amp.autocast('cuda'):
                logits, _ = model(x_b, decoder_input_ids=decoder_input_ids, labels=None, attention_mask=attn_mask)
                loss_ce = ce_criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                
            loss_align = align_criterion(projected_slots, target_embeds, m_b)
            loss = loss_ce + SLOT_ALIGNMENT_WEIGHT * loss_align
            
            loss = loss / GRAD_ACCUM_STEPS
            scaler.scale(loss).backward()
            
            if (i // BATCH_SIZE + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
            total_loss += loss.item() * GRAD_ACCUM_STEPS
            n += 1
            
        train_loss = total_loss / max(n, 1)
        
        model.eval()
        val_loss, val_steps = 0.0, 0
        with torch.no_grad():
            for i in range(0, X_val.size(0), BATCH_SIZE * 2):
                x_v = X_val[i:i+BATCH_SIZE*2].to(DEVICE, non_blocking=True)
                ids_v = Ids_val[i:i+BATCH_SIZE*2].to(DEVICE, non_blocking=True)
                m_v = Mask_val[i:i+BATCH_SIZE*2].to(DEVICE, non_blocking=True)
                
                dec_in = ids_v[:, :-1]
                lbls = ids_v[:, 1:]
                attn_v = m_v[:, :-1]
                
                with torch.amp.autocast('cuda'):
                    logits_v, _ = model(x_v, decoder_input_ids=dec_in, labels=None, attention_mask=attn_v)
                    val_loss += ce_criterion(logits_v.reshape(-1, logits_v.size(-1)), lbls.reshape(-1)).item()
                val_steps += 1
        val_loss /= max(val_steps, 1)
        
        elapsed = time.time() - start
        mode_tag = "Noise" if noise_alpha > 0.0 else "TF"
        print(f"Epoch {epoch+1:02d} | {mode_tag}({noise_alpha:.2f}) | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Time: {elapsed:.1f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': {
                    'num_slots': NUM_SLOTS, 'dim_model': DIM_MODEL, 'input_dim': input_dim,
                    'num_encoder_layers': 1, 'num_inds': 4, 'num_heads': 1,
                    'dropout_slots': 0.0, 't5_name': T5_NAME, 'max_seq_len': MAX_SEQ_LEN
                }
            }, CHECKPOINT_PATH)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"[Early Stop] Val loss plateaued after {EARLY_STOP_PATIENCE} epochs.")
                break
            
    gc.enable()
    print("[Training] Complete. Final checkpoint saved.")

if __name__ == "__main__":
    main()
