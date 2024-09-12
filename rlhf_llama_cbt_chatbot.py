
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from trl import PPOTrainer, PPOConfig
import wandb
from datetime import datetime
import transformers

# Initialize FullyShardedDataParallelPlugin
# fsdp_plugin = FullyShardedDataParallelPlugin(
#     state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
#     optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
# )
# accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

# Load and prepare both datasets
supervised_dataset = load_dataset('json', data_files='supervised_data.jsonl', split='train')
rl_dataset = load_dataset('json', data_files='Self_Labelled_data.jsonl', split='train')

max_length = 256

def format_supervised(example):
    # Formats the supervised dataset for training
    text = f"prompt: {example['input']} response: {example['output']}"
    return tokenizer(text, truncation=True, padding="max_length", max_length=256)

def format_rl(example):
    # Formats the RL dataset for RLHF training
    return {
        "prompt": example['Prompt'],
        "response": example['Response'],
        "score": example['Score']
    }

def formatting_func(example):
    text = f"### prompt: {example['input']}\n ### response: {example['output']}"
    return text

def generate_and_tokenize_prompt2(prompt):
    result = tokenizer(formatting_func(prompt), truncation=True, max_length=max_length, padding="max_length")
    result["labels"] = result["input_ids"].copy()
    return result

# Initialize tokenizer and model
base_model_id = "google/flan-t5-base" #"meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(base_model_id, padding_side="left", add_eos_token=True, add_bos_token=True)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# # Lora Configuration
# config = LoraConfig(
#     r=32, lora_alpha=64, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
#     bias="none", lora_dropout=0.05, task_type="CAUSAL_LM",
# )
# model = get_peft_model(model, config)

# Tokenize and format the datasets
tokenized_supervised_dataset = supervised_dataset.map(format_supervised)
tokenized_rl_dataset = rl_dataset.map(format_rl)

print("Dataset size:", len(tokenized_supervised_dataset))
print("Sample data:", tokenized_supervised_dataset[0])


# Supervised Training
training_args = TrainingArguments(
    output_dir="./model_output",
    warmup_steps=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    max_steps=500,
    learning_rate=2.5e-5,
    bf16=True,
    optim="paged_adamw_8bit",
    logging_dir="./logs",
    save_strategy="steps",
    save_steps=50,
    report_to="none"  # This line disables wandb
)
trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_supervised_dataset, data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False))
trainer.train()

# Reward Function
def calculate_reward(generated_response, ground_truth_score):
    max_score = 3
    normalized_score = ground_truth_score / max_score
    return torch.tensor([normalized_score])

# PPO Training
ppo_config = PPOConfig(batch_size=1, learning_rate=1e-5)
ppo_trainer = PPOTrainer(ppo_config, model, model, tokenizer)

for example in tokenized_rl_dataset:
    query_tensor = torch.tensor([example['input_ids']])
    model.eval()
    with torch.no_grad():
        response_tensor = model.generate(query_tensor, max_new_tokens=256)
    ground_truth_score = example['score']
    reward = calculate_reward(tokenizer.decode(response_tensor[0]), ground_truth_score)
    train_stats = ppo_trainer.step(query_tensor, response_tensor, reward)
