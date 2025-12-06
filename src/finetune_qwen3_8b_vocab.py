#!/usr/bin/env python3
"""
Fine-tune Qwen3-8B model with extended vocabulary for semantic IDs.
Stage 1: Embedding initialization - trains only new token embeddings.
"""

# Unsloth should be imported before trl, transformers, peft to ensure all optimizations are applied.
from unsloth import FastLanguageModel, is_bfloat16_supported, add_new_tokens  # isort: skip

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

import wandb
from src.device_manager import DeviceManager
from src.logger import setup_logger
from src.test_prompts import REC_TEST_PROMPTS, SYSTEM_PROMPT

logger = setup_logger("finetune-qwen3-vocab", log_to_file=True)


@dataclass
class FineTuneConfig:
    """Configuration for Stage 1: Embedding initialization."""

    # Model settings
    model_name: str = "unsloth/Qwen3-8B"
    max_seq_length: int = 512
    dtype: Optional[torch.dtype] = None  # None for auto detection
    load_in_4bit: bool = False  # Must be False for embedding training (quantized models can't be trained)
    load_in_8bit: bool = False
    random_state: int = 1368
    num_proc: int = 32
    enable_thinking: bool = False

    # Semantic ID vocabulary extension
    extend_vocabulary: bool = True
    codebook_levels: int = 4  # Number of hierarchical levels
    codebook_size: int = 256  # Number of codes per codebook
    num_semantic_tokens: int = 1024  # <|sid_0|> to <|sid_1023|>
    system_prompt: str = SYSTEM_PROMPT

    # Data settings
    category: str = "Video_Games"
    data_dir: Path = Path("data")
    max_training_samples: int = 32000  # Sample size for embedding init

    # Training params
    learning_rate: float = 1e-3
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_steps: int = 1000
    num_train_epochs: int = 1  # Train for 1 epoch
    warmup_steps: int = 100
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"

    # Memory optimization
    gradient_checkpointing: bool = False  # Disable to use more memory but slightly faster

    # Optimizer settings
    optim: str = "adamw_8bit"

    # Output settings
    output_dir: Path = Path("models/qwen3_8b_vocab_extended")
    steps_per_val_log: int = 250  # Validate and checkpoint every N steps
    save_steps: int = 5000  # Save checkpoints during training

    # Computed paths (set in __post_init__)
    train_path: Optional[Path] = None
    val_path: Optional[Path] = None

    def __post_init__(self):
        """Post-initialization setup and validation."""
        if self.dtype is None:
            self.dtype = torch.float16 if not is_bfloat16_supported() else torch.bfloat16

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set and validate data paths
        self.train_path = self.data_dir / "output" / f"{self.category}_conversations_train.parquet"
        self.val_path = self.data_dir / "output" / f"{self.category}_conversations_val.parquet"

        # Validate that training data exists
        if not self.train_path.exists():
            raise FileNotFoundError(
                f"Training data not found at {self.train_path}. "
                f"Please ensure you have run the data preparation scripts for category '{self.category}'."
            )

        # Validation data is optional, just log if missing
        if not self.val_path.exists():
            logger.warning(f"Validation data not found at {self.val_path}. Training without validation set.")

    def log_config(self):
        """Log all configuration parameters."""
        logger.info("=== Qwen3-8B Vocabulary Extension Configuration ===")
        logger.info("Stage 1: Embedding Initialization")

        # Model settings
        logger.info("Model Settings:")
        logger.info(f"  model_name: {self.model_name}")
        logger.info(f"  max_seq_length: {self.max_seq_length}")
        logger.info(f"  dtype: {self.dtype}")
        logger.info(f"  load_in_4bit: {self.load_in_4bit}")
        logger.info(f"  gradient_checkpointing: {self.gradient_checkpointing}")
        logger.info(f"  random_state: {self.random_state}")

        # Vocabulary extension
        logger.info("Vocabulary Extension:")
        logger.info(f"  extend_vocabulary: {self.extend_vocabulary}")
        logger.info(f"  codebook_levels: {self.codebook_levels}")
        logger.info(f"  codebook_size: {self.codebook_size}")
        logger.info(f"  num_semantic_tokens: {self.num_semantic_tokens}")
        logger.info(f"  Total new tokens: {self.num_semantic_tokens + 2} (including <|sid_start|> and <|sid_end|>)")

        # Data settings
        logger.info("Data Settings:")
        logger.info(f"  category: {self.category}")
        logger.info(f"  data_dir: {self.data_dir}")
        logger.info(f"  train_path: {self.train_path}")
        logger.info(f"  val_path: {self.val_path}")
        logger.info(f"  max_training_samples: {self.max_training_samples}")

        # Training parameters
        logger.info("Training Parameters (Stage 1):")
        logger.info(f"  learning_rate: {self.learning_rate} (high for embedding init)")
        logger.info(f"  batch_size: {self.batch_size}")
        logger.info(f"  gradient_accumulation_steps: {self.gradient_accumulation_steps}")
        logger.info(f"  effective_batch_size: {self.batch_size * self.gradient_accumulation_steps}")
        logger.info(f"  max_steps: {self.max_steps}")
        logger.info(f"  num_train_epochs: {self.num_train_epochs}")
        logger.info(f"  warmup_steps: {self.warmup_steps}")
        logger.info(f"  weight_decay: {self.weight_decay}")
        logger.info(f"  lr_scheduler_type: {self.lr_scheduler_type}")
        logger.info(f"  optim: {self.optim}")

        # Output settings
        logger.info("Output Settings:")
        logger.info(f"  output_dir: {self.output_dir}")
        logger.info(f"  steps_per_train_log: {self.steps_per_train_log}")
        logger.info(f"  steps_per_val_log: {self.steps_per_val_log}")
        logger.info(f"  save_steps: {self.save_steps}")
        logger.info("============================================")


def extend_tokenizer(model, tokenizer, config: FineTuneConfig):
    """
    Add semantic ID tokens to the tokenizer using Unsloth's add_new_tokens.

    Returns the number of new tokens added.
    """
    logger.info("=== Extending tokenizer with semantic ID tokens ===")

    # Log initial state
    original_vocab_size = len(tokenizer)
    original_embedding_size = model.get_input_embeddings().weight.shape[0]
    original_lm_head_size = model.get_output_embeddings().weight.shape[0]

    logger.info(
        f"Before - Vocab size: {original_vocab_size:,}, Embedding matrix: {original_embedding_size:,}, LM head matrix: {original_lm_head_size:,}"
    )

    # Fix size mismatch if model has more embeddings than tokenizer vocab
    if original_embedding_size > original_vocab_size:
        logger.warning(
            f"⚠ Model has {original_embedding_size - original_vocab_size} more embeddings than tokenizer tokens."
        )
        logger.info("Resizing model to match tokenizer before adding new tokens")
        model.resize_token_embeddings(original_vocab_size)

        # Update sizes after resize
        original_embedding_size = model.get_input_embeddings().weight.shape[0]
        original_lm_head_size = model.get_output_embeddings().weight.shape[0]
        logger.info(
            f"After resize - Embedding matrix: {original_embedding_size:,}, LM head matrix: {original_lm_head_size:,}"
        )

    # Add special tokens
    new_tokens = ["<|rec|>", "<|sid_start|>", "<|sid_end|>"]

    # Add tokens for semantic IDs: <|sid_0|> through <|sid_1023|>
    # Token mapping: <|sid_X|> where X = level * 256 + value
    # Level 0: <|sid_0|> through <|sid_255|>
    # Level 1: <|sid_256|> through <|sid_511|>
    # Level 2: <|sid_512|> through <|sid_767|>
    # Level 3: <|sid_768|> through <|sid_1023|>
    for i in range(config.num_semantic_tokens):
        new_tokens.append(f"<|sid_{i}|>")

    logger.info(f"Adding {len(new_tokens)} new tokens")
    logger.info("  Special tokens: <|rec|>, <|sid_start|>, <|sid_end|>")
    logger.info(f"  Semantic ID tokens: <|sid_0|> through <|sid_{config.num_semantic_tokens - 1}|>")

    # Add new tokens using Unsloth's method (uses mean initialization by default)
    add_new_tokens(model, tokenizer, new_tokens=new_tokens)

    # Log final state
    new_vocab_size = len(tokenizer)
    new_embedding_size = model.get_input_embeddings().weight.shape[0]
    new_lm_head_size = model.get_output_embeddings().weight.shape[0]

    logger.info(
        f"After - Vocab size: {new_vocab_size:,}, Embedding matrix: {new_embedding_size:,}, LM head matrix: {new_lm_head_size:,}"
    )

    # Verify consistency - CRITICAL CHECK
    if new_vocab_size != new_embedding_size:
        logger.error(f"❌ CRITICAL: Tokenizer size ({new_vocab_size}) != Embedding size ({new_embedding_size})")
        logger.info("Attempting to force resize model embeddings")
        model.resize_token_embeddings(new_vocab_size)
        new_embedding_size = model.get_input_embeddings().weight.shape[0]
        new_lm_head_size = model.get_output_embeddings().weight.shape[0]
        logger.info(f"After forced resize - Embedding: {new_embedding_size}, LM head: {new_lm_head_size}")

    if new_vocab_size != new_lm_head_size:
        logger.error(f"❌ CRITICAL: Tokenizer size ({new_vocab_size}) != LM head size ({new_lm_head_size})")
        logger.error("Model will NOT be able to generate new tokens!")
        # Try to fix by resizing
        model.resize_token_embeddings(new_vocab_size)
        new_lm_head_size = model.get_output_embeddings().weight.shape[0]
        logger.info(f"After forced resize - LM head: {new_lm_head_size}")

    # Final verification
    if new_vocab_size == new_embedding_size == new_lm_head_size:
        logger.info("✅ Model dimensions verified: All layers properly sized")
    else:
        logger.error("❌ Model dimension mismatch persists - this will cause generation issues!")

    num_added = new_vocab_size - original_vocab_size
    logger.info(f"\n✓ Successfully added {num_added} new tokens")
    logger.info("=" * 50)

    return num_added


def prepare_model(model, tokenizer, config: FineTuneConfig, num_new_tokens: int):
    """
    Prepare model for embedding-only training by freezing all parameters except new embeddings.
    Note: Model embeddings should already be resized by add_new_tokens().
    """
    logger.info("=== Preparing model for embedding-only training ===")

    # Get original vocab size before new tokens
    current_vocab_size = len(tokenizer)
    current_embedding_size = model.get_input_embeddings().weight.shape[0]

    logger.info(
        f"Current - Vocab size: {current_vocab_size:,}, Embedding matrix: {current_embedding_size:,}, New tokens: {num_new_tokens:}"
    )

    # Verify embeddings are already properly sized (should be done by add_new_tokens)
    assert current_embedding_size == current_vocab_size, "Embedding size mismatch!"

    # Get the original vocab size (before adding new tokens)
    original_vocab_size = current_vocab_size - num_new_tokens

    # Freeze all parameters first
    for _, param in model.named_parameters():
        param.requires_grad = False

    # Unfreeze BOTH input and output embeddings for NEW tokens only
    embedding_layer = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()

    # Unfreeze entire embedding layers for simplicity
    embedding_layer.weight.requires_grad = True
    mask = torch.zeros(current_vocab_size, 1)
    mask[original_vocab_size:] = 1.0  # Unfreeze new tokens
    embedding_layer.weight.register_hook(lambda grad: grad * mask)

    if output_embeddings is not None:
        output_embeddings.weight.requires_grad = True
        logger.info("✅ Unfroze both input and output embedding layers for training")
    else:
        logger.error("❌ Could not access output embeddings - only input embeddings will be trained!")
        logger.error("This will cause generation issues!")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    logger.info(f"Percentage trainable: {100 * trainable_params / total_params:.4f}%")

    # Check initialization of new embeddings (already initialized by add_new_tokens with mean)
    original_vocab_size = len(tokenizer) - num_new_tokens
    with torch.no_grad():
        new_embeddings = embedding_layer.weight[original_vocab_size:]
        logger.info("New embeddings statistics (initialized by Unsloth with mean):")
        logger.info(f"  Shape: {new_embeddings.shape}")
        logger.info(f"  Mean: {new_embeddings.mean().item():.6f}")
        logger.info(f"  Std: {new_embeddings.std().item():.6f}")
        logger.info(f"  Min: {new_embeddings.min().item():.6f}")
        logger.info(f"  Max: {new_embeddings.max().item():.6f}")

    # Note: Gradient checkpointing can be enabled but may have compatibility issues with Unsloth/Qwen3
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled for memory efficiency")
    else:
        model.gradient_checkpointing_disable()
        logger.info("Gradient checkpointing disabled")

    # Set cache based on gradient checkpointing
    model.config.use_cache = not config.gradient_checkpointing

    logger.info("=== Model preparation complete ===")

    return model


def load_sid_dataset(config: FineTuneConfig, tokenizer, split="train"):
    """
    Load and prepare the conversation dataset with semantic IDs.

    Args:
        config: Configuration object
        tokenizer: Tokenizer to apply chat template
        split: "train" or "val" to load respective dataset
    """
    logger.info(f"Loading semantic ID conversation dataset ({split})")

    # Select the appropriate path based on split
    if split == "train":
        data_path = config.train_path
    elif split == "val":
        data_path = config.val_path
    logger.info(f"Loading from: {data_path}")

    dataset = load_dataset("parquet", data_files=str(data_path), split="train")
    logger.info(f"Loaded {len(dataset)} conversations")

    # For validation, use fewer samples
    if split == "val":
        num_samples = min(len(dataset), 500)  # Max 500 for validation
    else:
        num_samples = min(len(dataset), config.max_training_samples)

    logger.info(f"Sampling {num_samples} examples for {split}")
    dataset = dataset.shuffle(seed=config.random_state).select(range(num_samples))

    # Apply chat template using map
    def apply_chat_template(example):
        text = tokenizer.apply_chat_template(
            example["conversations"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=config.enable_thinking,
        )
        return {"text": text}

    logger.info("Applying chat template to conversations")
    dataset = dataset.map(apply_chat_template, remove_columns=dataset.column_names, num_proc=config.num_proc)

    logger.info(f"Created dataset with {len(dataset)} examples")

    # Verify semantic IDs are present (only for train)
    if split == "train" and len(dataset) > 0:
        sample_text = dataset[0]["text"]
        if "<|sid_start|>" in sample_text and "<|sid_end|>" in sample_text:
            logger.info("✓ Verified: Semantic ID tokens found in dataset")
            sid_count = sample_text.count("<|sid_start|>")
            logger.info(f"  Sample contains {sid_count} semantic ID(s)")

            # Log a sample of the chat template output
            logger.info("=" * 60)
            logger.info(f"Sample chat template output ({split}): {sample_text}")
            logger.info("=" * 60)

        else:
            logger.warning("⚠ Warning: No semantic ID tokens found in sample")

    return dataset


class DataInspectionCallback(TrainerCallback):
    """Inspect training data and tokenization at each logging step."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.trainer = None  # Will be set later
        self.first_batch_inspected = False

    def set_trainer(self, trainer):
        """Set the trainer after it's been created."""
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        """Inspect first batch at training start."""
        if not self.first_batch_inspected:
            self.first_batch_inspected = True
            logger.info("\n" + "=" * 60)
            logger.info("=== Initial Training Data Inspection ===")
            logger.info("=" * 60)

            try:
                if self.trainer is None:
                    logger.info("Trainer not yet set, skipping inspection")
                    return

                train_dataloader = self.trainer.get_train_dataloader()

                # Get first batch
                for batch in train_dataloader:
                    logger.info(f"Batch keys: {batch.keys()}")
                    logger.info(f"Input IDs shape: {batch['input_ids'].shape}")
                    logger.info(f"Attention mask shape: {batch['attention_mask'].shape}")

                    # Inspect first example in batch
                    first_example = batch["input_ids"][0]
                    decoded = self.tokenizer.decode(first_example, skip_special_tokens=False)

                    # Show full token list
                    logger.info(f"Tokens (first example): {first_example.tolist()}")
                    logger.info(f"Decoded: {decoded}")

                    # Count SID tokens
                    token_list = first_example.tolist()
                    sid_tokens = sum(1 for t in token_list if 151672 <= t <= 152695)
                    logger.info(f"Number of SID tokens: {sid_tokens}")

                    break  # Just check first batch

            except Exception as e:
                logger.info(f"Could not inspect first batch: {e}")

            logger.info("=" * 60 + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Inspect data at each logging step."""
        # Only log every logging_steps
        if state.global_step > 0 and state.global_step % args.logging_steps == 0:
            logger.info("=" * 60)
            logger.info(f"=== Data Inspection at Step {state.global_step} ===")
            logger.info("=" * 60)

            try:
                if self.trainer is None:
                    logger.info("Trainer not yet set, skipping inspection")
                    return

                train_dataloader = self.trainer.get_train_dataloader()

                # Get one batch from current position
                for i, batch in enumerate(train_dataloader):
                    # Just get the first batch we encounter
                    logger.info(f"Batch shape: {batch['input_ids'].shape}")

                    # Show first example
                    first_example = batch["input_ids"][0]
                    token_list = first_example.tolist()

                    # Count SID tokens
                    sid_tokens = sum(1 for t in token_list if 151672 <= t <= 152695)

                    logger.info(f"First example - SID tokens: {sid_tokens}, Total tokens: {len(token_list)}")
                    logger.info(f"Token IDs: {token_list}")

                    # Show decoded version (truncated for readability)
                    decoded = self.tokenizer.decode(first_example, skip_special_tokens=False)
                    logger.info(f"Decoded: {decoded}")

                    break  # Just check first batch

            except Exception as e:
                logger.info(f"Could not inspect batch at step {state.global_step}: {e}")
            logger.info("=" * 60 + "\n")


class EmbeddingMonitorCallback(TrainerCallback):
    """Monitor embedding statistics and log to W&B."""

    def __init__(self, tokenizer, num_new_tokens, monitor_interval=100):
        self.tokenizer = tokenizer
        self.num_new_tokens = num_new_tokens
        self.monitor_interval = monitor_interval
        self.original_vocab_size = len(tokenizer) - num_new_tokens
        self.initial_embeddings = None
        self.prev_embeddings = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Capture initial embedding state."""
        embeddings = model.get_input_embeddings().weight
        self.fixed_embeddings = embeddings[: self.original_vocab_size].clone().detach()
        self.initial_embeddings = embeddings[self.original_vocab_size :].clone().detach()
        self.prev_embeddings = self.initial_embeddings.clone()

        # Log initial statistics
        mean = self.initial_embeddings.mean().item()
        std = self.initial_embeddings.std().item()

        wandb.log(
            {
                "embeddings/initial_mean": mean,
                "embeddings/initial_std": std,
                "embeddings/initial_norm": self.initial_embeddings.norm(dim=-1).mean().item(),
            }
        )

        logger.info(f"Initial embeddings - Mean: {mean:.4f}, Std: {std:.4f}")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Monitor embedding changes and log."""
        if state.global_step % self.monitor_interval == 0 and state.global_step > 0:
            embeddings = model.get_input_embeddings().weight
            old_embeddings = embeddings[: self.original_vocab_size]
            new_embeddings = embeddings[self.original_vocab_size :]

            # Calculate changes
            change_from_fixed = (old_embeddings - self.fixed_embeddings).abs().mean().item()
            if change_from_fixed > 1e-6:
                logger.warning(
                    f"⚠ Warning: Original embeddings changed during training! "
                    f"Change from fixed: {change_from_fixed:.6f}"
                )
            change_from_init = (new_embeddings - self.initial_embeddings).abs().mean().item()
            change_from_prev = (new_embeddings - self.prev_embeddings).abs().mean().item()

            # Calculate statistics
            mean = new_embeddings.mean().item()
            std = new_embeddings.std().item()
            norm = new_embeddings.norm(dim=-1).mean().item()

            # Calculate per-level statistics (for semantic ID levels)
            level_stats = {}
            tokens_per_level = config.codebook_size
            for level in range(4):
                start_idx = level * tokens_per_level
                end_idx = min((level + 1) * tokens_per_level, self.num_new_tokens - 2)  # -2 for <sid>, </sid>
                if start_idx < self.num_new_tokens - 2:
                    level_embeddings = new_embeddings[start_idx:end_idx]
                    level_stats[f"embeddings/level_{level}_mean"] = level_embeddings.mean().item()
                    level_stats[f"embeddings/level_{level}_std"] = level_embeddings.std().item()
                    level_stats[f"embeddings/level_{level}_norm"] = level_embeddings.norm(dim=-1).mean().item()

            # Check gradients if available
            grad_norm = 0.0
            grad_max = 0.0
            if embeddings.grad is not None:
                grad = embeddings.grad[self.original_vocab_size :]
                grad_norm = grad.norm().item()
                grad_max = grad.abs().max().item()

            wandb_log = {
                "embeddings/change_from_init": change_from_init,
                "embeddings/change_from_prev": change_from_prev,
                "embeddings/mean": mean,
                "embeddings/std": std,
                "embeddings/norm": norm,
                "embeddings/grad_norm": grad_norm,
                "embeddings/grad_max": grad_max,
                "step": state.global_step,
                **level_stats,  # Add per-level stats
            }

            wandb.log(wandb_log)

            # Console logging
            logger.info(
                f"Step {state.global_step} - Embeddings: "
                f"Change(init)={change_from_init:.4f}, Change(prev)={change_from_prev:.6f}, "
                f"Mean={mean:.4f}, Std={std:.4f}, Norm={norm:.4f}"
            )

            self.prev_embeddings = new_embeddings.clone().detach()


class SemanticIDGenerationCallback(TrainerCallback):
    """Test semantic ID generation and log to W&B."""

    def __init__(self, tokenizer, test_interval=200):
        self.tokenizer = tokenizer
        self.test_interval = test_interval
        self.test_messages = REC_TEST_PROMPTS

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Test generation at specified intervals."""
        if state.global_step % self.test_interval == 0 and state.global_step > 0:
            self.test_generation(model, state.global_step)

    def test_generation(self, model, step):
        """Test if model can generate with semantic IDs and log to W&B."""
        logger.info("=" * 60)
        logger.info(f"Testing semantic ID generation at step {step}")
        logger.info("=" * 60)

        training_mode = model.training
        model.eval()

        successful_generations = 0
        generation_results = []

        for i, messages in enumerate(self.test_messages, 1):
            # Apply chat template to format the messages properly
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)

            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, max_new_tokens=100, temperature=0.7, min_p=0.01, top_p=0.8, top_k=20
                    )

                generated_full = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
                generated_new = generated_full[len(prompt) :]

                # Analysis
                has_sid_tags = "<|sid_start|>" in generated_new or "<|sid_end|>" in generated_new
                sid_tokens = [t for t in generated_new.split() if t.startswith("<|sid_") and t.endswith("|>")]
                uses_semantic_ids = has_sid_tags or len(sid_tokens) > 0

                if uses_semantic_ids:
                    successful_generations += 1

                # Get user message for cleaner logging
                user_message = messages[-1]["content"]

                # Store for table
                generation_results.append([step, user_message, generated_new, uses_semantic_ids, len(sid_tokens)])

                # Log results
                logger.info(f"\nTest {i}: {user_message}")
                logger.info(f"  Generated: {generated_new}")
                logger.info(f"  ✓ Uses SIDs: {uses_semantic_ids} (tags={has_sid_tags}, tokens={len(sid_tokens)})")

            except Exception as e:
                user_message = messages[-1]["content"]
                logger.warning(f"Generation failed for prompt {i}: {e}")
                generation_results.append([step, user_message[:50], f"[Error: {e}]", False, 0])

        success_rate = successful_generations / len(self.test_messages)

        wandb.log(
            {
                "generation/success_rate": success_rate,
                "generation/successful_count": successful_generations,
                "generation/total_prompts": len(self.test_messages),
                "generation/examples": wandb.Table(
                    columns=["Step", "User_Message", "Generated", "Uses_SID", "Num_Tokens"], data=generation_results
                ),
                "step": step,
            }
        )

        logger.info(
            f"\nSummary: {successful_generations}/{len(self.test_messages)} "
            f"({success_rate:.0%}) prompts generated semantic IDs"
        )

        model.train(training_mode)
        logger.info("=" * 60 + "\n")


def train_embeddings(model, tokenizer, config: FineTuneConfig, num_new_tokens: int):
    """
    Train only the new token embeddings using SFTTrainer with comprehensive monitoring.

    Args:
        model: The model with extended vocabulary
        tokenizer: The extended tokenizer
        config: Training configuration
        num_new_tokens: Number of new tokens added to vocabulary
    """
    logger.info("Starting Stage 1: Embedding initialization training")

    train_dataset = load_sid_dataset(config, tokenizer, split="train")
    val_dataset = load_sid_dataset(config, tokenizer, split="val")
    wandb.log(
        {
            "dataset/train_size": len(train_dataset) if train_dataset else 0,
            "dataset/val_size": len(val_dataset) if val_dataset else 0,
            "dataset/vocabulary_size": len(tokenizer),
            "dataset/new_tokens": num_new_tokens,
        }
    )

    # Create SFT configuration for embedding training with W&B reporting
    sft_config = SFTConfig(
        dataset_text_field="text",
        dataset_num_proc=config.num_proc,  # Increase parallel tokenization processes
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps if config.max_steps > 0 else -1,  # Use max_steps if > 0, else -1 for epoch-based
        num_train_epochs=config.num_train_epochs if config.max_steps <= 0 else 1,  # Use epochs if max_steps <= 0
        learning_rate=config.learning_rate,
        logging_steps=config.steps_per_train_log,
        optim=config.optim,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        seed=config.random_state,
        output_dir=str(config.output_dir),
        save_steps=config.save_steps,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        report_to="wandb",  # Enable W&B reporting
        save_strategy="steps",
        gradient_checkpointing=config.gradient_checkpointing,  # Use config setting
        # Evaluation settings
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=config.steps_per_val_log if val_dataset else None,
        per_device_eval_batch_size=config.batch_size,
        metric_for_best_model="eval_loss" if val_dataset else None,
        greater_is_better=False,
        load_best_model_at_end=True if val_dataset else False,
        save_total_limit=2,
    )

    # Create data inspection callback because we need to set the trainer afterwards
    data_inspection_callback = DataInspectionCallback(tokenizer)

    # Create other callbacks
    callbacks = [
        data_inspection_callback,  # Add data inspection callback
        EmbeddingMonitorCallback(tokenizer, num_new_tokens, monitor_interval=config.steps_per_val_log),
        SemanticIDGenerationCallback(tokenizer, test_interval=config.steps_per_val_log),
    ]

    # Create trainer with callbacks
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=sft_config,
        callbacks=callbacks,
    )

    # Set trainer in the data inspection
    data_inspection_callback.set_trainer(trainer)

    # Show current memory stats (if CUDA available)
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        logger.info(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    # Show final memory and time stats (if CUDA available)
    if torch.cuda.is_available():
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_training = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        training_percentage = round(used_memory_for_training / max_memory * 100, 3)
        logger.info(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        logger.info(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
        logger.info(f"Peak reserved memory = {used_memory} GB.")
        logger.info(f"Peak reserved memory for training = {used_memory_for_training} GB.")
        logger.info(f"Peak reserved memory % of max memory = {used_percentage} %.")
        logger.info(f"Peak reserved memory for training % of max memory = {training_percentage} %.")

    # Log final summary
    wandb.summary["final_loss"] = trainer_stats.training_loss if hasattr(trainer_stats, "training_loss") else None
    wandb.summary["total_steps"] = (
        trainer_stats.global_step if hasattr(trainer_stats, "global_step") else config.max_steps
    )
    wandb.summary["training_time_seconds"] = trainer_stats.metrics["train_runtime"]

    logger.info("Stage 1 embedding initialization completed!")
    return trainer_stats


def save_model_and_tokenizer(model, tokenizer, config: FineTuneConfig):
    """
    Save the model with initialized embeddings and extended tokenizer.
    """
    logger.info("Saving model and tokenizer")

    # CRITICAL: Verify dimensions before saving
    input_size = model.get_input_embeddings().weight.shape[0]
    output_size = model.get_output_embeddings().weight.shape[0]
    vocab_size = len(tokenizer)

    logger.info("=== Pre-save verification ===")
    logger.info(f"Model input embedding size: {input_size}")
    logger.info(f"Model output embedding size: {output_size}")
    logger.info(f"Tokenizer vocabulary size: {vocab_size}")

    if input_size != vocab_size or output_size != vocab_size:
        logger.error("❌ CRITICAL: Size mismatch detected before save!")
        logger.error(f"  Input: {input_size}, Output: {output_size}, Tokenizer: {vocab_size}")
        logger.error("This will cause generation issues in Stage 2!")
        # Try to fix one more time
        model.resize_token_embeddings(vocab_size)
        input_size = model.get_input_embeddings().weight.shape[0]
        output_size = model.get_output_embeddings().weight.shape[0]
        logger.info(f"After final resize - Input: {input_size}, Output: {output_size}")

    # Final verification
    assert input_size == vocab_size, f"Input embeddings size mismatch: {input_size} != {vocab_size}"
    assert output_size == vocab_size, f"Output embeddings size mismatch: {output_size} != {vocab_size}"
    logger.info("✅ All dimensions verified - safe to save")

    # Save to final directory
    final_save_path = config.output_dir / "final"
    final_save_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving to: {final_save_path}")

    # Save model
    model.save_pretrained(str(final_save_path))

    # Save tokenizer
    tokenizer.save_pretrained(str(final_save_path))

    logger.info("Model and tokenizer saved successfully!")
    logger.info(f"Checkpoint location: {final_save_path}")

    config_dict = {
        "stage": "vocab_extension",
        "model_name": config.model_name,
        "num_semantic_tokens": config.num_semantic_tokens,
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "category": config.category,
        "vocabulary_size": len(tokenizer),
    }

    with open(final_save_path / "training_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    logger.info("Training configuration saved")


if __name__ == "__main__":
    config = FineTuneConfig()
    device_manager = DeviceManager(logger)
    device = device_manager.device

    run_name = f"qwen3-8b-embed-{config.category}-lr{config.learning_rate}"
    run = wandb.init(project="semantic-id-vocab-extension", name=run_name, config=config.__dict__)
    config.log_config()

    logger.info("Loading base model")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit,
    )

    num_new_tokens = 0
    if config.extend_vocabulary:
        num_new_tokens = extend_tokenizer(model, tokenizer, config)
        model = prepare_model(model, tokenizer, config, num_new_tokens)

    train_stats = train_embeddings(model, tokenizer, config, num_new_tokens)

    logger.info("Saving embeddings as W&B artifact")
    embeddings = model.get_input_embeddings().weight
    new_embeddings = embeddings[len(tokenizer) - num_new_tokens :].detach().cpu()

    artifact = wandb.Artifact(
        f"semantic_embeddings_{config.category}",
        type="embeddings",
        description=f"Trained semantic ID embeddings for {config.category}",
        metadata={
            "num_tokens": num_new_tokens,
            "model": config.model_name,
            "steps": train_stats.global_step if hasattr(train_stats, "global_step") else config.max_steps,
            "final_loss": train_stats.training_loss if hasattr(train_stats, "training_loss") else None,
        },
    )

    embeddings_path = config.output_dir / "semantic_embeddings.npy"
    np.save(embeddings_path, new_embeddings.float().numpy())
    artifact.add_file(str(embeddings_path))
    wandb.log_artifact(artifact)

    save_model_and_tokenizer(model, tokenizer, config)

    wandb.finish()

    logger.info("=" * 50)
    logger.info("Stage 1: Embedding initialization complete!")
    logger.info(f"Initialized {config.num_semantic_tokens + 2} new semantic ID tokens")
    logger.info(f"Model saved to: {config.output_dir / 'final'}")
    logger.info("=" * 50)
