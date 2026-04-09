import torch
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from transformers.image_utils import load_image
from pathlib import Path
import random
from peft import MissConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
import peft
from typing import Any, Dict, Tuple
from PIL import Image
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from lvm_utils.utils import upscale_image


target_modules=[
    # Vision and Language Attention
    "q_proj", "k_proj", "v_proj", "out_proj",
    
    # Vision MLP
    "fc1", "fc2",
    
    # Multimodal Projector (essential for aligning the new vision features to text)
    "linear_1", "linear_2",
    
    # Optional: Language Model Feed Forward / Conv if you still want to train the LM heavily
    "w1", "w2", "w3", "in_proj" 
]

default_peft_config = MissConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    mini_r=8,
    miss_dropout=0.05,
    bias="none",
    target_modules=target_modules
)
# default_peft_config = peft.IA3Config(
#     task_type=TaskType.SEQ_CLS, target_modules=["k_proj", "v_proj", "down_proj", "o_proj"], feedforward_modules=["down_proj"]
# )

# default_peft_config = peft.LoraConfig(
#     r=16,
#     lora_alpha=8,
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM"
# )
default_quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # NormalFloat4 is optimal for normally distributed weights
    bnb_4bit_compute_dtype=torch.bfloat16, # Compute dtype for the dequantization overhead
    bnb_4bit_use_double_quant=True, # Quantizes the quantization constants for extra VRAM savings
    llm_int8_skip_modules=["patch_embedding", "lm_head"] # Skip the patch embedding from integer quantization to prevent input cast errors!
)
    
model_id = "LiquidAI/LFM2-VL-450M"

def load_model_id(model_id : str = model_id, peft_config : Any = default_peft_config, quantization_config : Any = default_quantization_config, cache = True, load_peft = True, quantize = False, torch_dtype = torch.bfloat16, **kwargs) -> Tuple[AutoModelForImageTextToText, AutoProcessor]:
    """
    Load a model and processor with PEFT configuration. If cache is True, it will save the original model to disk and load from there in subsequent calls to speed up loading.
    
    Args:
        model_id (str): The Hugging Face model ID to load.
        peft_config (Any): The PEFT configuration to apply to the model.
        quantization_config (Any): The quantization configuration to apply to the model.
        cache (bool): Whether to cache the original model to disk for faster loading in subsequent calls
        load_peft (bool): Whether to load the PEFT configuration.
        quantize (bool): Whether to apply quantization to the model.

    Returns:
        model (AutoModelForImageTextToText): The loaded model with PEFT applied.
        processor (AutoProcessor): The corresponding processor for the model.
    """
    
    cached_path = Path("./" + model_id + "-original")
    if cached_path.exists() and cache:
        model = None
        if quantize:
            model = AutoModelForImageTextToText.from_pretrained(
                cached_path.absolute(),
                device_map="auto",
                torch_dtype=torch_dtype,  # compute dtype
                quantization_config=quantization_config,
                **kwargs
            )
        else:
            model = AutoModelForImageTextToText.from_pretrained(
                cached_path.absolute(),
                device_map="auto",
                torch_dtype=torch_dtype,
                **kwargs
            )
        processor = AutoProcessor.from_pretrained(cached_path.absolute())
        model = prepare_model_for_kbit_training(model) if quantize else model
        if load_peft:
            model = get_peft_model(model, peft_config)
        
        return model, processor
    else:
        model = None
        if quantize:
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch_dtype,
                quantization_config=quantization_config,
                **kwargs
            )
        else:
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch_dtype,
                **kwargs
            )
        processor = AutoProcessor.from_pretrained(model_id)
        model = prepare_model_for_kbit_training(model) if quantize else model
        if cache == True:
            model.save_pretrained(cached_path.absolute())
            processor.save_pretrained(cached_path.absolute())
        if load_peft:
            model = get_peft_model(model, peft_config)
        return model, processor

def first_stage(image : Image.Image, processor : AutoProcessor, model : AutoModelForImageTextToText, max_tokens = 300) -> Dict[str, Any]:
    """Run the first stage of the pipeline, which generates a detailed description of the image and then asks the model to make a guess about what the image is of.
    
    Args:
        image (Image.Image): The input image.
        processor (AutoProcessor): The processor for the model.
        model (AutoModelForImageTextToText): The model to use.
        max_tokens (int): The maximum number of tokens to generate.
    
    Returns
        Dict[str, Any]: The conversation history.
    """
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": upscale_image(image, 7)},
                {"type": "text", "text": "Describe the image in great detail."},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens)
        assistant_text =  processor.batch_decode(outputs, skip_special_tokens=True)[0]
    conversation.append({"role": "assistant", "content": [{"type": "text", "text": assistant_text}]})
    conversation.append({"role": "user",
        "content": [
            {"type": "text", "text": "You must make a guess, make your best guess, even if you are not sure. What is the image of? One word"}
        ]})
    return conversation


def drop_intermediate_reasoning_from_conversation(
    conversation,
    guess_prompt_substring: str = "You must make a guess",
):
    """Drop assistant reasoning turns that occur before the final guess prompt.

    This expects the conversation shape:
    user(image + prompt) -> assistant(reasoning) -> user(final guess prompt).
    If the final guess prompt is present, assistant turns before that prompt are removed.
    """
    if not isinstance(conversation, list):
        return conversation

    guess_user_idx = None
    for i, msg in enumerate(conversation):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        text_parts = [part.get("text", "") for part in content if part.get("type") == "text"]
        combined_text = " ".join(text_parts)
        if guess_prompt_substring in combined_text:
            guess_user_idx = i
            break

    if guess_user_idx is None:
        return conversation

    return [
        msg
        for i, msg in enumerate(conversation)
        if not (i < guess_user_idx and msg.get("role") == "assistant")
    ]


def stochastic_drop_intermediate_reasoning_batch(
    conversations,
    drop_probability: float = 0.0,
    guess_prompt_substring: str = "You must make a guess",
    rng=None,
):
    """Stochastically remove intermediate reasoning turns per conversation in batch."""
    if not conversations or drop_probability <= 0.0:
        return conversations
    if drop_probability >= 1.0:
        return [
            drop_intermediate_reasoning_from_conversation(c, guess_prompt_substring=guess_prompt_substring)
            for c in conversations
        ]

    rng = rng if rng is not None else random
    return [
        drop_intermediate_reasoning_from_conversation(c, guess_prompt_substring=guess_prompt_substring)
        if rng.random() < drop_probability
        else c
        for c in conversations
    ]

def get_batch_conversation_embeddings(
    model,
    processor,
    conversations,
    normalize=True,
    append_token_id=None,
    pad_token_id=None,
    intermediate_reasoning_drop_probability=0.0,
    guess_prompt_substring: str = "You must make a guess",
    intermediate_reasoning_rng=None,
):
    """
    Batched embedding extraction for multimodal conversations.

    Procedure:
    1) Tokenize conversations with processor chat template (with padding).
    2) Compute true prompt length per sample from attention_mask.
    3) Append one known token (default: eos) right after each sample's true prompt.
    4) Forward pass with output_hidden_states=True.
    5) Gather hidden state at appended token position for each sample.

    Returns:
        emb: [B, H]
        prompt_lens: [B] (position index where appended token was placed)
    """
    device = next(model.parameters()).device

    conversations = stochastic_drop_intermediate_reasoning_batch(
        conversations,
        drop_probability=intermediate_reasoning_drop_probability,
        guess_prompt_substring=guess_prompt_substring,
        rng=intermediate_reasoning_rng,
    )

    # 1) Process whole batch of conversations
    # conversations should be a list, each item is one conversation list-of-messages.
    batch = processor.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={
            "padding": True,
        }
    )

    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

    input_ids = batch["input_ids"]          # [B, Lpad]
    attention_mask = batch["attention_mask"]  # [B, Lpad]

    # 2) True per-sample prompt lengths (number of non-pad tokens)
    prompt_lens = attention_mask.sum(dim=1).long()  # [B]

    # 3) Pick known ids. In training loops, pass cached ids to avoid repeated tokenizer lookups.
    if append_token_id is None:
        tok = processor.tokenizer
        append_token_id = tok.eos_token_id
        if append_token_id is None:
            append_token_id = tok.sep_token_id
        if append_token_id is None:
            append_token_id = tok.pad_token_id
        if append_token_id is None:
            raise ValueError("No eos/sep/pad token id available to append.")

    if pad_token_id is None:
        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0

    # Build variable-length sequences: true prompt + one appended token
    seqs = []
    for i in range(input_ids.size(0)):
        n = int(prompt_lens[i].item())
        seq_i = input_ids[i, :n]
        append_tok = seq_i.new_full((1,), append_token_id)
        seq_i = torch.cat([seq_i, append_tok], dim=0)
        seqs.append(seq_i)

    # Re-pad after appending so appended token is at index prompt_lens[i]
    input_ids2 = pad_sequence(seqs, batch_first=True, padding_value=pad_token_id)  # [B, L2]
    attention_mask2 = (input_ids2 != pad_token_id).long()

    # Keep multimodal tensors (pixel_values, image_grid_thw, etc.)
    extra = {k: v for k, v in batch.items() if k not in ("input_ids", "attention_mask")}

    # TEMPORARY LM_HEAD OVERRIDE FOR VRAM EFFICIENCY:
    # 1. output_hidden_states=True stores ALL layer activations, defeating gradient checkpointing.
    # 2. Extracting logits takes enormous VRAM for large vocab sizes (e.g. 150k+).
    # Solution: We replace the lm_head with Identity to get the final hidden state directly.
    target_model = model.base_model.model if hasattr(model, "base_model") else model
    original_lm_head = target_model.lm_head
    target_model.lm_head = torch.nn.Identity()

    try:
        # 4) Forward pass
        out = model(
            input_ids=input_ids2,
            attention_mask=attention_mask2,
            return_dict=True,
            **extra,
        )
        
        # 'logits' is now actually the final hidden state!
        last_hidden = out.logits  # [B, L2, H]
    finally:
        # Restore the original lm_head
        target_model.lm_head = original_lm_head

    # 5) Gather appended-token embeddings
    # appended token position per sample is prompt_lens[i] (0-based index)
    batch_idx = torch.arange(last_hidden.size(0), device=device)
    emb = last_hidden[batch_idx, prompt_lens, :]  # [B, H]

    if normalize:
        emb = F.normalize(emb, p=2, dim=-1)

    return emb, prompt_lens


@dataclass(frozen=True)
class ConversationEmbeddingConfig:
    """Reusable token-id config for batched conversation embeddings."""

    append_token_id: int
    pad_token_id: int


def build_conversation_embedding_config(processor, append_token_id=None, pad_token_id=None):
    """Resolve token ids once and reuse inside a training loop."""
    tok = processor.tokenizer

    resolved_append = append_token_id
    if resolved_append is None:
        resolved_append = tok.eos_token_id
    if resolved_append is None:
        resolved_append = tok.sep_token_id
    if resolved_append is None:
        resolved_append = tok.pad_token_id
    if resolved_append is None:
        raise ValueError("No eos/sep/pad token id available to append.")

    resolved_pad = pad_token_id
    if resolved_pad is None:
        resolved_pad = tok.pad_token_id
    if resolved_pad is None:
        resolved_pad = 0

    return ConversationEmbeddingConfig(
        append_token_id=int(resolved_append),
        pad_token_id=int(resolved_pad),
    )


def get_batch_conversation_embeddings_with_config(
    model,
    processor,
    conversations,
    config: ConversationEmbeddingConfig,
    normalize=True,
    intermediate_reasoning_drop_probability=0.0,
    guess_prompt_substring: str = "You must make a guess",
    intermediate_reasoning_rng=None,
):
    """Training-loop friendly wrapper using precomputed token ids."""
    return get_batch_conversation_embeddings(
        model=model,
        processor=processor,
        conversations=conversations,
        normalize=normalize,
        append_token_id=config.append_token_id,
        pad_token_id=config.pad_token_id,
        intermediate_reasoning_drop_probability=intermediate_reasoning_drop_probability,
        guess_prompt_substring=guess_prompt_substring,
        intermediate_reasoning_rng=intermediate_reasoning_rng,
    )

def get_batch_conversation_embeddings_with_config_2(
    model,
    processor,
    conversations,
    config: ConversationEmbeddingConfig,
    normalize=True,
):
    """Training-loop friendly wrapper using precomputed token ids."""
    return get_batch_conversation_embeddings_2(
        model=model,
        processor=processor,
        conversations=conversations,
        normalize=normalize,
        append_token_id=config.append_token_id,
        pad_token_id=config.pad_token_id,
    )

def get_batch_conversation_embeddings_2(
    model,
    processor,
    conversations,
    normalize=True,
    append_token_id=None,
    pad_token_id=None,
):
    device = next(model.parameters()).device

    # 1) Process conversations. 
    # add_generation_prompt=False prevents appending unclosed assistant headers 
    # before our explicit pooling token.
    batch = processor.apply_chat_template(
        conversations,
        add_generation_prompt=False, 
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={
            "padding": True,
        }
    )

    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    input_ids = batch["input_ids"]          
    attention_mask = batch["attention_mask"]  

    prompt_lens = attention_mask.sum(dim=1).long() 

    # 2) Token ID resolution
    if append_token_id is None:
        tok = processor.tokenizer
        append_token_id = getattr(tok, "eos_token_id", None) or \
                          getattr(tok, "sep_token_id", None) or \
                          getattr(tok, "pad_token_id", None)
        if append_token_id is None:
            raise ValueError("No eos/sep/pad token id available to append.")

    if pad_token_id is None:
        pad_token_id = getattr(processor.tokenizer, "pad_token_id", 0)

    # 3) Build variable-length sequences
    seqs = []
    for i in range(input_ids.size(0)):
        n = int(prompt_lens[i].item())
        seq_i = input_ids[i, :n]
        append_tok = torch.tensor([append_token_id], dtype=seq_i.dtype, device=device)
        seq_i = torch.cat([seq_i, append_tok], dim=0)
        seqs.append(seq_i)

    # Re-pad sequences
    input_ids2 = pad_sequence(seqs, batch_first=True, padding_value=pad_token_id)
    attention_mask2 = (input_ids2 != pad_token_id).long()

    extra = {k: v for k, v in batch.items() if k not in ("input_ids", "attention_mask")}

    # 4) Architectural Bypass: Target the base model directly
    # This executes the transformer layers and final LayerNorm, skipping lm_head.
    # Safe for FSDP, DDP, and torch.compile.
    base_model = model.model if hasattr(model, "model") else model.base_model.model
    
    out = base_model(
        input_ids=input_ids2,
        attention_mask=attention_mask2,
        return_dict=True,
        **extra,
    )
    
    last_hidden = out.last_hidden_state  # [B, L2, H]

    # 5) Gather appended-token embeddings
    batch_idx = torch.arange(last_hidden.size(0), device=device)
    emb = last_hidden[batch_idx, prompt_lens, :]  

    if normalize:
        emb = F.normalize(emb, p=2, dim=-1)

    return emb, prompt_lens
def first_stage_batch(images : list[Image.Image], processor : AutoProcessor, model : AutoModelForImageTextToText, max_tokens = 300) -> list[Dict[str, Any]]:
    conversations = []
    for image in images:
        conversations.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Describe the image in great detail."},
                    ],
                },
            ]
        )

    orig_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"

    inputs = processor.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={
            "padding": True,
        }
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens)
        new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        assistant_texts = processor.batch_decode(new_tokens, skip_special_tokens=True)

    processor.tokenizer.padding_side = orig_padding_side

    for i in range(len(conversations)):
        conversations[i].append({"role": "assistant", "content": [{"type": "text", "text": assistant_texts[i].strip()}]})
        conversations[i].append({"role": "user",
            "content": [
                {"type": "text", "text": "You must make a guess, make your best guess, even if you are not sure. What is the image of? One word"}
            ]})
    
    return conversations

