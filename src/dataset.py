import os
import torch
from torch.utils.data import Dataset

# =============================================================================
# DATASET LOADER
# =============================================================================
class StoryEmbeddingDataset(Dataset):
    """Loads precomputed story embeddings and raw text from flat chunked .pt files."""
    def __init__(self, data_dir, model_filter, split, max_samples=None, preload=True):
        self.data_dir = data_dir
        self.preload = preload
        self.samples = []

        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Match existing naming convention: embeddings_{model_filter}_{split}_chunk_XXXX.pt
        prefix = f"embeddings_{model_filter}_{split}_chunk_"
        chunk_files = sorted([f for f in os.listdir(self.data_dir) if f.startswith(prefix) and f.endswith('.pt')])

        if not chunk_files:
            raise FileNotFoundError(f"No chunk files found matching pattern '{prefix}*.pt' in {self.data_dir}")

        print(f"Found {len(chunk_files)} chunk files for {split}.")

        if preload:
            print(f"Mode: PRELOAD (Loading data into RAM)...")
            for cf in chunk_files:
                chunk_data = torch.load(os.path.join(self.data_dir, cf), map_location='cpu')
                
                # Normalize dict format {"embeddings": [...], "texts": [...]} to list of dicts
                if isinstance(chunk_data, dict):
                    embeddings = chunk_data.get("embeddings") or chunk_data.get("embeddings ")
                    texts = chunk_data.get("texts") or chunk_data.get("texts ")
                    for emb, txt in zip(embeddings, texts):
                        self.samples.append({"embeddings": emb, "text": txt})
                else:
                    self.samples.extend(chunk_data)

                if max_samples and len(self.samples) >= max_samples:
                    self.samples = self.samples[:max_samples]
                    break
            print(f"Loaded {len(self.samples)} samples into RAM.")
        else:
            # Lazy loading fallback
            self.chunk_files = chunk_files
            self.total_len = 0
            for cf in chunk_files:
                chunk_data = torch.load(os.path.join(self.data_dir, cf), map_location='cpu')
                if isinstance(chunk_data, dict):
                    self.total_len += len(chunk_data.get("embeddings", chunk_data.get("embeddings ", [])))
                else:
                    self.total_len += len(chunk_data)
            self.total_len = min(self.total_len, max_samples or self.total_len)

    def __len__(self):
        return len(self.samples) if self.preload else self.total_len

    def __getitem__(self, idx):
        if self.preload:
            return self.samples[idx]
        
        current_offset = 0
        for cf in self.chunk_files:
            chunk_data = torch.load(os.path.join(self.data_dir, cf), map_location='cpu')
            if isinstance(chunk_data, dict):
                chunk_len = len(chunk_data.get("embeddings", chunk_data.get("embeddings ", [])))
            else:
                chunk_len = len(chunk_data)
                
            if current_offset <= idx < current_offset + chunk_len:
                if isinstance(chunk_data, dict):
                    return {
                        "embeddings": chunk_data["embeddings"][idx - current_offset],
                        "text": chunk_data["texts"][idx - current_offset]
                    }
                return chunk_data[idx - current_offset]
            current_offset += chunk_len
        raise IndexError("Dataset index out of bounds")

# =============================================================================
# TENSOR PREPARATION
# =============================================================================
def prepare_pinned_dataset(dataset, limit, max_len, tokenizer):
    """Converts dataset to contiguous RAM tensors for fast GPU transfer."""
    limit = min(limit, len(dataset))
    dim = dataset[0]["embeddings"].shape[-1]
    
    x = torch.zeros(limit, max_len, dim, dtype=torch.float32)
    target_ids = torch.full((limit, max_len), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros(limit, max_len, dtype=torch.bool)
    
    print(f"Preparing {limit} samples (BERT embeds + T5 targets)...")
    for i in range(limit):
        item = dataset[i]
        emb = item.get("embeddings")
        length = min(emb.shape[0], max_len)
        x[i, :length] = emb[:length]
        mask[i, :length] = True
        
        text = item.get("text", "")
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=max_len)
        target_ids[i, :len(tokens)] = torch.tensor(tokens[:max_len], dtype=torch.long)
        
    return x, target_ids, mask
