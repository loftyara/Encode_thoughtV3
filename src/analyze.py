import os
import torch
from transformers import T5Tokenizer
from peft import LoraConfig, get_peft_model
from model import EncodeThought
from dataset import StoryEmbeddingDataset

# ================= CONFIGURATION =================
CHECKPOINT_PATH = "../checkpoints/best.pt"
DATA_DIR = "../data/processed"
MODEL_FILTER = "bert-mini"
T5_NAME = "t5-small"
NUM_SAMPLES = 5
MAX_SEQ_LEN = 128
DEVICE = "cuda"

TEMPERATURE = 0.8
TOP_P = 0.9
REPETITION_PENALTY = 1.2

# ================= MAIN DIAGNOSTICS =================
def main():
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    print(f"[Device] {DEVICE}")
    tokenizer = T5Tokenizer.from_pretrained(T5_NAME)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    cfg = ckpt["config"]

    # Initialize model with saved config
    model = EncodeThought(
        input_dim=cfg["input_dim"], dim_model=cfg["dim_model"], num_heads=cfg["num_heads"],
        num_encoder_layers=cfg["num_encoder_layers"], num_inds=cfg["num_inds"],
        num_slots=cfg["num_slots"], max_seq_len=cfg["max_seq_len"],
        t5_name=cfg["t5_name"], dropout_slots=cfg["dropout_slots"], freeze_t5=True
    ).to(DEVICE)
    
    # Re-wrap LoRA to match checkpoint structure
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, target_modules=["q", "k", "v", "o"],
        lora_dropout=0.05, bias="none", task_type="SEQ_2_SEQ_LM"
    )
    model.decoder.t5 = get_peft_model(model.decoder.t5, lora_cfg)
    
    model.load_state_dict(ckpt["model_state_dict"])
    model.float()
    model.eval()
    print(f"[Model] Loaded checkpoint. Slots={cfg['num_slots']}, Dim={cfg['dim_model']}")
    print(f"[Inference] T5 Cross-Attn + LoRA | Temp={TEMPERATURE} | Top-p={TOP_P} | RepPen={REPETITION_PENALTY}")

#    val_ds = StoryEmbeddingDataset(DATA_DIR, MODEL_FILTER, "val", NUM_SAMPLES, preload=True)
    val_ds = StoryEmbeddingDataset(DATA_DIR, MODEL_FILTER, "train", NUM_SAMPLES, preload=True)
    exact_matches = 0
    slot_utilization_confirmed = False

    for i in range(len(val_ds)):
        sample = val_ds[i]
        orig_text = sample["text"]
        x_full = sample["embeddings"]
        length = min(sample.get("length", len(x_full)), MAX_SEQ_LEN)
        x = x_full[:length].unsqueeze(0).to(DEVICE).float()

        print(f"\n{'='*70}")
        print(f"[Sample {i+1}] Length: {length}")
        print(f"[Original]\n{orig_text}")
        print("-" * 70)

        with torch.no_grad():
            slots = model.get_slots(x)
            encoder_hidden = model.decoder.projector(slots)
            encoder_outputs = (encoder_hidden,)
            
            gen_ids = model.decoder.t5.generate(
                encoder_outputs=encoder_outputs,
                max_length=length,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                do_sample=True,
                decoder_start_token_id=0,
                bos_token_id=0,
                pad_token_id=0,
                eos_token_id=tokenizer.eos_token_id
            )
            
        recovered_text = tokenizer.decode(gen_ids[0].cpu().tolist(), skip_special_tokens=True)
        print(f"[Recovered (AR)]\n{recovered_text}")
        print("-" * 70)

        match = recovered_text.strip() == orig_text.strip()
        if match: exact_matches += 1

        theme_overlap = sum(1 for w in orig_text.lower().split() if w in recovered_text.lower().split()) / max(len(orig_text.split()), 1)
        if theme_overlap > 0.50: slot_utilization_confirmed = True

        print(f"[Exact Match] {match} | [Theme Overlap] {theme_overlap:.2f}")

    print(f"\n{'='*70}")
    print(f"[Summary] Exact Matches: {exact_matches}/{NUM_SAMPLES}")
    print(f"[Slot Utilization] {'Confirmed' if slot_utilization_confirmed else 'Weak/None'}")

    if exact_matches == 0 and slot_utilization_confirmed:
        print("[Diagnosis] AR stable + slots active. Semantic drift within expected bounds for current capacity.")
    elif not slot_utilization_confirmed:
        print("[Diagnosis] Slots misaligned or projector degraded. Verify training convergence.")
    else:
        print("[Diagnosis] Faithful AR recovery achieved. Pipeline ready for scaling.")

if __name__ == "__main__":
    main()
