import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

#os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]
# Model and tokenizer names
base_model_name = "NousResearch/Llama-2-7b-chat"
base_model_name = "meta-llama/Meta-Llama-3-8B"
new_model_name = "llama-3-8b-enhanced" #You can give your own name for fine tuned model

# Tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, 
                                                cache_dir="llama-tokenizers-cache", 
                                                trust_remote_code=True)

#,local_files_only=True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"

# Model
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, 
                                                  torch_dtype=torch.float16,
                                                  cache_dir="llama-models-cache",
                                                  trust_remote_code=True
                                                  )



base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

# Data set
#data_name = "mlabonne/guanaco-llama2-1k"
#training_data = load_dataset(data_name, split="train", cache_dir="data_cache")

from prompt_utils import get_datasets
train_dataset, test_dataset = get_datasets("hypothesis_train.jsonl", "hypothesis_test.jsonl")


# check the data
#print(training_data.shape)
# #11 is a QA sample in English
#print(training_data[11])

# Training Params
train_params = SFTConfig(
    output_dir="./results_modified",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    #optim="paged_adamw_32bit",
    save_steps=50,
    logging_steps=50,
    learning_rate=4e-5,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard",
    dataset_text_field="text",
    deepspeed="ds_config.json"
)

from peft import get_peft_model
# LoRA Config
peft_parameters = LoraConfig(
    lora_alpha=8,
    lora_dropout=0.1,
    r=8,
    bias="none",
    task_type="CAUSAL_LM"
)
#model = get_peft_model(base_model, peft_parameters)
#base_model.print_trainable_parameters()

print("configuring trainer")
#torch.dist.barrier()
# Trainer with LoRA configuration
fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    peft_config=peft_parameters,
    processing_class=llama_tokenizer,
    args=train_params
)

print("configured trainer")

# Training
fine_tuning.train()
#base_model.save_pretrained("finetuned_llama3_8B_hypothesis")
