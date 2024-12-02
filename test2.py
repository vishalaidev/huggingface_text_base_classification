import torch
from datasets import load_dataset
from peft import get_peft_model, LoraConfig
from transformers import Trainer, TrainingArguments
from transformers import PaliGemmaProcessor, BitsAndBytesConfig
from transformers import PaliGemmaForConditionalGeneration

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "google/paligemma-3b-pt-224"

# Load dataset
ds = load_dataset('merve/vqav2-small')
split_ds = ds["validation"].train_test_split(test_size=0.05)
train_ds = split_ds["train"]
test_ds = split_ds["test"]

# Load processor
processor = PaliGemmaProcessor.from_pretrained(model_id)

# Define collate function
def collate_fn(examples):
    texts = [f"<image> <bos> answer {example['question']}" for example in examples]
    labels = [example['multiple_choice_answer'] for example in examples]
    images = [example["image"].convert("RGB") for example in examples]
    tokens = processor(
        text=texts,
        images=images,
        suffix=labels,
        return_tensors="pt",
        padding="longest"
    )
    tokens = tokens.to(torch.bfloat16).to(device)
    return tokens

# Load model with quantization and device mapping
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",  # Automatically map layers to GPU/CPU
    torch_dtype=torch.bfloat16
)

# Freeze specific parameters to save memory
for param in model.vision_tower.parameters():
    param.requires_grad = False
for param in model.multi_modal_projector.parameters():
    param.requires_grad = False

# Apply LoRA (Low-Rank Adaptation) for fine-tuning
lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Print trainable parameter info

args = TrainingArguments(
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    learning_rate=2e-5,
    weight_decay=1e-6,
    adam_beta2=0.999,
    logging_steps=100,
    optim="adamw_hf",
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=1,
    output_dir="finetuned_paligemma_vqav2_small",
    bf16=torch.cuda.is_available(),
    dataloader_pin_memory=False,
    report_to=[],
    remove_unused_columns=False  # Add this to bypass column checking
)


# Trainer setup
trainer = Trainer(
    model=model,
    train_dataset=train_ds,
    data_collator=collate_fn,
    args=args
)

# Start training
trainer.train()
