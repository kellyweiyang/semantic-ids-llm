#!/usr/bin/env python3
"""
Generate embeddings for product items using Qwen3-Embedding model.
Reads pre-tokenized data and generates embeddings optimized for GPU execution.

First run src/tokenize_items.py to pre-process data on CPU.
"""

import os
import time
from pathlib import Path

# Disable tokenizers parallelism to avoid forking warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel

from src.logger import setup_logger

logger = setup_logger("embed-items", log_to_file=True)

# Data settings
CATEGORY = "Video_Games"  # Product category to process
NUM_ROWS = None  # Number of rows to process (None = all)
DATA_DIR = Path("data")  # Data directory path

# Model settings
#MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"  # HuggingFace model name
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # HuggingFace model name
BATCH_SIZE = 64  # Batch size for processing
TARGET_DIM = 384  # Target embedding dimension

# Other settings
VERIFY_CONSISTENCY = False  # Verify single vs batch embedding consistency
LOG_FREQ = 1000  # Log progress every N items


class TokenizedDataset(Dataset):
    """Dataset for pre-tokenized data with pinned memory support."""

    def __init__(self, input_ids: np.ndarray, attention_mask: np.ndarray):
        """Initialize with numpy arrays and convert to pinned tensors."""
        # Convert to tensors and pin memory for faster GPU transfers
        #self.input_ids = torch.from_numpy(input_ids).pin_memory()
        #self.attention_mask = torch.from_numpy(attention_mask).pin_memory()
        self.input_ids = torch.from_numpy(input_ids)
        self.attention_mask = torch.from_numpy(attention_mask)
        self.length = len(input_ids)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "attention_mask": self.attention_mask[idx]}


def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        # Enable TF32 for better performance
        torch.set_float32_matmul_precision("high")

        # Set memory fraction to avoid OOM
        torch.cuda.set_per_process_memory_fraction(0.95)
        torch.cuda.empty_cache()

        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Extract embeddings using last token pooling."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def generate_embeddings(
    model: AutoModel,
    device: str,
    pretokenized_batch: dict,
    target_dim: int = 384,
) -> np.ndarray:
    """
    Generate embeddings for a batch of pre-tokenized inputs using last token pooling.
    Returns L2-normalized embeddings, optionally truncated to target dimension.
    """
    # Move to device
    encoded = {k: v.to(device) for k, v in pretokenized_batch.items()}

    # Generate embeddings
    with torch.no_grad():
        outputs = model(**encoded)

        # Use last token pooling
        embeddings = last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])

        # Truncate to target dimension if specified
        if target_dim and target_dim < embeddings.shape[1]:
            embeddings = embeddings[:, :target_dim]

        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().numpy()


def verify_embedding_consistency(model: AutoModel, device: str, pretokenized_data: dict) -> None:
    """
    Verify that embedding a single item produces the same result as embedding it in a batch.
    This checks for potential issues with padding or batch processing.
    """
    logger.info("Verifying embedding consistency...")

    # Get first batch
    batch_size = min(BATCH_SIZE, pretokenized_data["n_items"])

    # Generate embeddings for first batch
    batch_input_ids = torch.from_numpy(pretokenized_data["input_ids"][:batch_size])
    batch_attention_mask = torch.from_numpy(pretokenized_data["attention_mask"][:batch_size])
    batch_tokens = {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
    }

    batch_embeddings = generate_embeddings(model, device, batch_tokens, TARGET_DIM)

    # Generate embedding for first item only
    single_input_ids = torch.from_numpy(pretokenized_data["input_ids"][:1])
    single_attention_mask = torch.from_numpy(pretokenized_data["attention_mask"][:1])
    single_tokens = {
        "input_ids": single_input_ids,
        "attention_mask": single_attention_mask,
    }

    single_embedding = generate_embeddings(model, device, single_tokens, TARGET_DIM)

    # Compare embeddings
    are_similar = np.allclose(single_embedding[0], batch_embeddings[0], rtol=1e-6, atol=1e-6)
    diff = np.abs(single_embedding[0] - batch_embeddings[0])

    logger.info(f"Embeddings are similar: {are_similar}")
    logger.info(f"Max difference: {diff.max():.2e}")
    logger.info(f"Mean difference: {diff.mean():.2e}")
    logger.info(f"Single embedding: {single_embedding[0]}")
    logger.info(f"First batch embedding: {batch_embeddings[0]}")

    if not are_similar:
        logger.warning("Embeddings differ more than expected!")
        logger.warning("This may indicate issues with padding or batch processing.")
    else:
        logger.info("✓ Embedding consistency verified")


def embed_items():
    # Device selection
    device = get_device()
    logger.info(f"Device: {device}, Model: {MODEL_NAME}, Batch: {BATCH_SIZE}")

    # Setup paths
    input_path = DATA_DIR / "output" / f"{CATEGORY}_items_updated.parquet"
    output_path = DATA_DIR / "output" / f"{CATEGORY}_items_with_embeddings.parquet"

    # Tokenized data path
    suffix = f"_{NUM_ROWS}" if NUM_ROWS else ""
    tokenized_path = DATA_DIR / "output" / f"{CATEGORY}_tokenized{suffix}.npz"

    # Load data
    logger.info(f"Loading data from {input_path}")
    item_df = pl.read_parquet(input_path)
    logger.info(f"Loaded {len(item_df):,} items from {input_path}")
    logger.info("Sample of item_contexts:")
    pl.Config.set_fmt_str_lengths(2000)
    logger.info(item_df["item_context"].head(5))

    # Apply row limit if specified
    if NUM_ROWS is not None:
        logger.info(f"Limiting to {NUM_ROWS} rows for testing")
        item_df = item_df.head(NUM_ROWS)

    total_items = len(item_df)
    logger.info(f"Processing {total_items:,} items")

    # Load model for embedding generation
    logger.info(f"Loading model: {MODEL_NAME}")
    # Don't use flash attention as it only support bf16 and makes embeddings less precise
    model = AutoModel.from_pretrained(MODEL_NAME)

    # Move model to device
    model = model.to(device)
    model.eval()

    use_compile = True
    if device == "cuda" and use_compile:  # Only compile for CUDA, not for MPS or CPU
        model = torch.compile(model)

    logger.info(f"Model hidden size: {model.config.hidden_size}, Target dim: {TARGET_DIM}")

    # Load pre-tokenized data
    if not tokenized_path.exists():
        raise FileNotFoundError(
            f"Pre-tokenized data not found at {tokenized_path}. Please run src/tokenize_items.py first."
        )

    logger.info(f"Loading pre-tokenized data from {tokenized_path}")
    pretokenized_data = np.load(tokenized_path)

    # Verify data matches
    if pretokenized_data["n_items"] != total_items:
        raise ValueError(
            f"Pre-tokenized data has {pretokenized_data['n_items']} items, but current data has {total_items}"
        )

    logger.info(f"Loaded pre-tokenized data: shape {pretokenized_data['input_ids'].shape}")

    # Verify embedding consistency if enabled
    if VERIFY_CONSISTENCY:
        verify_embedding_consistency(model, device, pretokenized_data)

    # Create dataset and dataloader
    logger.info("Creating DataLoader with pinned memory...")
    dataset = TokenizedDataset(pretokenized_data["input_ids"], pretokenized_data["attention_mask"])

    # DataLoader with optimizations
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # Keep order for consistent results
        num_workers=4,  # Parallel data loading
        pin_memory=False,  # Already pinned in dataset
        prefetch_factor=2,  # Prefetch next batches
        persistent_workers=True,  # Keep workers alive
    )

    # Pre-allocate output array
    logger.info(f"Pre-allocating output array for {total_items} embeddings with dim {TARGET_DIM}...")
    all_embeddings = np.zeros((total_items, TARGET_DIM), dtype=np.float32)

    # Process batches
    start_time = time.time()
    current_idx = 0

    with tqdm(total=total_items, desc="Generating embeddings") as pbar:
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Get batch size (last batch might be smaller)
                batch_size = batch["input_ids"].size(0)

                # Generate embeddings
                batch_embeddings = generate_embeddings(model, device, batch, TARGET_DIM)

                # Write directly to pre-allocated array
                all_embeddings[current_idx : current_idx + batch_size] = batch_embeddings
                current_idx += batch_size

                # Update progress
                pbar.update(batch_size)

                # Log progress every LOG_FREQ items
                if current_idx % LOG_FREQ == 0 or current_idx == total_items:
                    elapsed = time.time() - start_time
                    items_per_sec = current_idx / elapsed
                    eta = (total_items - current_idx) / items_per_sec if current_idx < total_items else 0
                    logger.info(
                        f"Processed {current_idx:,}/{total_items:,} items "
                        f"({items_per_sec:.1f} items/sec, ETA: {eta / 60:.1f} min)"
                    )

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                raise

    # Final timing
    total_time = time.time() - start_time
    logger.info("Embedding generation complete!")
    logger.info(f"Total time: {total_time:.1f} seconds ({total_time / 60:.1f} minutes)")
    logger.info(f"Average time per item: {total_time / total_items * 1000:.1f} ms")

    # Use the pre-allocated array directly
    embeddings_array = all_embeddings
    logger.info(f"Embeddings shape: {embeddings_array.shape}")

    # Verify embeddings are normalized
    norms = np.linalg.norm(embeddings_array, axis=1)
    logger.info(f"Embedding L2 norms - Mean: {norms.mean():.6f}, Std: {norms.std():.6f}")

    # Add embeddings to dataframe
    embeddings_list = embeddings_array.tolist()
    item_df_with_embeddings = item_df.with_columns(pl.Series("embedding", embeddings_list, dtype=pl.List(pl.Float32)))

    # Save results
    logger.info(f"Saving embeddings to: {output_path}")
    item_df_with_embeddings.write_parquet(output_path)

    # Final statistics
    logger.info("Final statistics:")
    logger.info(f"- Total items with embeddings: {len(item_df_with_embeddings):,}")
    logger.info(f"- Embedding dimension: {embeddings_array.shape[1]}")
    logger.info(f"- Output file size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    logger.info(f"- Processing rate: {total_items / total_time:.1f} items/sec")

    # GPU memory stats if available
    if device == "cuda":
        logger.info(f"- Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")


if __name__ == "__main__":
    embed_items()
