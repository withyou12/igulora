#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.
import copy
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import pdb
import datasets
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from datasets import load_dataset
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    HfArgumentParser,
    TrainingArguments,
    set_seed, DataCollatorForSeq2Seq, GenerationConfig, get_scheduler, AutoTokenizer, AutoConfig, AutoModelForCausalLM,AutoModelForSequenceClassification
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version


sys.path.append("./")
from iglora.rank_allocator import RankAllocator

from iglora.lora_module import SVDLoRAModule

from iglora.calc_gen_scores import calc_scores
from iglora.io_utils import dump_json, dump_jsonl

os.environ["WANDB_MODE"] = "disabled"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "where to store the cached data."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    data_cache_dir: Optional[str] = field(default="./", metadata={"help": "The datasets processed stored"})

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

@dataclass
class MyTrainingArguments(TrainingArguments):

    # lora
    lora_rank : Optional[int] = field(default=16)
    lora_dropout : Optional[float] = field(default=0.3)
    lora_alpha : Optional[float] = field(default=32.)
    peft_path : Optional[str] = field(default=None)
    # training
    apply_lora: Optional[bool] = field(default=False)
    do_train: Optional[bool] = field(default=False)
    do_generation : Optional[bool] = field(default=False)
    learning_rate: Optional[float] = field(default=1e-5)
    num_train_epochs: Optional[int] = field(default=2)
    eval_steps: Optional[int] = field(default=100)
    max_patience: Optional[int] = field(default=4)
    dropout_prob: Optional[float] = field(default=0.1)
    tunable_param_names: Optional[str] = field(
        default=None,
        metadata={"help": "separate by comma; keywords for filtering tunable adapter/lora params"},
    )
    # hyper-params about lora pruning
    start_prune_steps: Optional[int] = field(default=100)
    end_prune_steps: Optional[int] = field(default=100)
    prune_every_steps: Optional[int] = field(default=50)
    num_to_masks_per_layer: Optional[int] = field(default=1)
    lora_rank_target: Optional[float] = field(default=8.0)
    target_total_rank: Optional[float] = field(default=10.0)

logger = logging.getLogger(__name__)

def apply_lora_to_model(model, config=None):

    for layer_idx in range(config.num_hidden_layers):
        q_proj = model.model.layers[layer_idx].self_attn.q_proj
        q_proj_with_lora = SVDLoRAModule(
            linear_module=q_proj,
            in_features=q_proj.in_features,
            out_features=q_proj.out_features,
            r=config.lora_rank,
            lora_dropout=0.3
        )
        model.model.layers[layer_idx].self_attn.q_proj = q_proj_with_lora

        k_proj = model.model.layers[layer_idx].self_attn.k_proj
        k_proj_with_lora = SVDLoRAModule(
            linear_module=k_proj,
            in_features=k_proj.in_features,
            out_features=k_proj.out_features,
            r=config.lora_rank,
            lora_dropout=0.3
        )
        model.model.layers[layer_idx].self_attn.k_proj = k_proj_with_lora

        v_proj = model.model.layers[layer_idx].self_attn.v_proj
        v_proj_with_lora = SVDLoRAModule(
            linear_module=v_proj,
            in_features=v_proj.in_features,
            out_features=v_proj.out_features,
            r=config.lora_rank,
            lora_dropout=0.3
        )
        model.model.layers[layer_idx].self_attn.v_proj = v_proj_with_lora

        o_proj = model.model.layers[layer_idx].self_attn.o_proj
        o_proj_with_lora = SVDLoRAModule(
            linear_module=o_proj,
            in_features=o_proj.in_features,
            out_features=o_proj.out_features,
            r=config.lora_rank,
            lora_dropout=0.3
        )
        model.model.layers[layer_idx].self_attn.o_proj = o_proj_with_lora

        gate_proj = model.model.layers[layer_idx].mlp.gate_proj
        gate_proj_with_lora = SVDLoRAModule(
            linear_module=gate_proj,
            in_features=gate_proj.in_features,
            out_features=gate_proj.out_features,
            r=config.lora_rank,
            lora_dropout=0.3
        )
        model.model.layers[layer_idx].mlp.gate_proj = gate_proj_with_lora

        up_proj = model.model.layers[layer_idx].mlp.up_proj
        up_proj_with_lora = SVDLoRAModule(
            linear_module=up_proj,
            in_features=up_proj.in_features,
            out_features=up_proj.out_features,
            r=config.lora_rank,
            lora_dropout=0.3
        )
        model.model.layers[layer_idx].mlp.up_proj = up_proj_with_lora

        down_proj = model.model.layers[layer_idx].mlp.down_proj
        down_proj_with_lora = SVDLoRAModule(
            linear_module=down_proj,
            in_features=down_proj.in_features,
            out_features=down_proj.out_features,
            r=config.lora_rank,
            lora_dropout=0.3
        )
        model.model.layers[layer_idx].mlp.down_proj = down_proj_with_lora

def compute_svd_orth_regu(model):
    regu_loss = 0.0
    num_param = 0
    for n, p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            para_cov = p @ p.T if "lora_A" in n else p.T @ p
            num_param += 1
            # print("para_cov: ", para_cov.shape)
            # print("torch.diag(para_cov): ", torch.diag(para_cov).shape)
            #
            # print("para_cov: ", para_cov)
            # print("torch.diag(torch.diag(para_cov)): ", torch.diag(torch.diag(para_cov)))
            regu_loss += torch.norm(
                para_cov - torch.diag(torch.diag(para_cov)).to("cuda"),
                p="fro"
            )
    return regu_loss/num_param


def save_lora_modules(model, config=None, to_path=None):

    state_dict = model.state_dict()
    for key_ in copy.copy(list(state_dict.keys())):
        if "lora" not in key_:
            state_dict.pop(key_)
    torch.save(state_dict, to_path)


def sample_coeff():
    # Generate 20 uniformly distributed numbers between 0 and 1
    numbers = [w/20 for w in range(1, 21)]
    # Randomly select a value
    selected_value = random.choice(numbers)
    return selected_value

def set_lora_weight_coeffs(model, config=None, weight_coeff_value=1.0):
    for layer_idx in range(config.num_hidden_layers):
        for module_idx, module_name in enumerate(
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        ):
            if module_name == "q_proj":
                weight_coeff_ = model.model.layers[layer_idx].self_attn.q_proj.weight_coeff
            elif module_name == "k_proj":
                weight_coeff_ = model.model.layers[layer_idx].self_attn.k_proj.weight_coeff
            elif module_name == "v_proj":
                weight_coeff_ = model.model.layers[layer_idx].self_attn.v_proj.weight_coeff
            elif module_name == "o_proj":
                weight_coeff_ = model.model.layers[layer_idx].self_attn.o_proj.weight_coeff
            elif module_name == "gate_proj":
                weight_coeff_ = model.model.layers[layer_idx].mlp.gate_proj.weight_coeff
            elif module_name == "up_proj":
                weight_coeff_ = model.model.layers[layer_idx].mlp.up_proj.weight_coeff
            elif module_name == "down_proj":
                weight_coeff_ = model.model.layers[layer_idx].mlp.down_proj.weight_coeff
            else:
                raise ValueError

            # Modify weight_coeff_
            weight_coeff_[0] = weight_coeff_value
            if module_name == "q_proj":
                model.model.layers[layer_idx].self_attn.q_proj.weight_coeff.data.copy_(weight_coeff_)
            elif module_name == "k_proj":
                model.model.layers[layer_idx].self_attn.k_proj.weight_coeff.data.copy_(weight_coeff_)
            elif module_name == "v_proj":
                model.model.layers[layer_idx].self_attn.v_proj.weight_coeff.data.copy_(weight_coeff_)
            elif module_name == "o_proj":
                model.model.layers[layer_idx].self_attn.o_proj.weight_coeff.data.copy_(weight_coeff_)
            elif module_name == "gate_proj":
                model.model.layers[layer_idx].mlp.gate_proj.weight_coeff.data.copy_(weight_coeff_)
            elif module_name == "up_proj":
                model.model.layers[layer_idx].mlp.up_proj.weight_coeff.data.copy_(weight_coeff_)
            elif module_name == "down_proj":
                model.model.layers[layer_idx].mlp.down_proj.weight_coeff.data.copy_(weight_coeff_)
            else:
                raise ValueError




def eval_model(model, eval_dataloader):
    logger.info("开始验证")
    model.eval()
    losses = []
    total_loss = 0.0
    num_batches = 0
    total_samples = 0
    correct = 0
    total = 0
    for step, batch in tqdm(enumerate(eval_dataloader)):
        bsz = batch["input_ids"].shape[0]
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.to(torch.float32).cpu().numpy().tolist() * bsz
        num_batches += 1
        total_samples += bsz
    try:
        eval_loss = total_loss / total_samples
        perplexity = math.exp(eval_loss)
        print("eval_loss: ", eval_loss)
    except OverflowError:
        perplexity = float("inf")
        eval_loss = 1000000000
        print("eval_loss: ", eval_loss)
    return eval_loss

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)
    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,  # if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=[logging.StreamHandler(sys.stdout)],)
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # transformers.tokenization_utils.logging.set_verbosity_warning()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    # record training args
    tb_writter = SummaryWriter(log_dir=os.path.join(training_args.output_dir, "tensorboard"))

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        # "num_labels":2
    }
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    config.lora_rank = training_args.lora_rank
    config.lora_dropout = training_args.lora_dropout
    config.lora_alpha = training_args.lora_alpha
    config.dropout_prob = training_args.dropout_prob
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, **tokenizer_kwargs
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
    generation_config = GenerationConfig.from_dict(
        {
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "do_sample": False,
            "top_k": 0,
            "top_p": 0.0,
            "num_beams": 5,
            "repetition_penalty": 1.05,
            "max_new_tokens": 128
        }
    )

    def tokenize_function(example):
        max_seq_length = data_args.block_size
        sentence = example["sentence"]
        target_text = "positive" if example["label"] == 1 else "negative"
        prompt = f"Classify the sentiment of the following sentence as positive or negative:\nSentence: {sentence}\nSentiment:"
        input_enc = tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt"
        )
        target_enc = tokenizer(
            target_text,
            add_special_tokens=False,   
            truncation=True,
            max_length=4,               
            return_tensors="pt"
        ).input_ids[0]
        target_length = len(target_enc)
        labels = input_enc["input_ids"].clone()
        if input_enc["input_ids"].shape[1] > target_length:
            labels[:, :-target_length] = -100
        else:
            labels[:] = -100

        for i, token_id in enumerate(target_enc):
            labels[:, -target_length + i] = token_id

        assert input_enc["input_ids"].shape == labels.shape, \
            f"Shape mismatch: input {input_enc['input_ids'].shape}, labels {labels.shape}"
        return {
            "input_ids": input_enc["input_ids"][0],
            "attention_mask": input_enc["attention_mask"][0],
            "labels": labels[0],
        }

    raw_datasets = load_dataset(
        "parquet",
        data_files={
            "train": os.path.join(data_args.dataset_name, "train.parquet"),
            "dev": os.path.join(data_args.dataset_name, "dev.parquet"),
            # "test": os.path.join(data_args.dataset_name, "test.jsonl"),
        }
    )
    os.makedirs(os.path.join(training_args.output_dir, f"cache/"), exist_ok=True)
    tokenized_dataset = raw_datasets.map(
                tokenize_function,
                batched=False,
                num_proc=1,
                remove_columns=raw_datasets["train"].column_names,
                load_from_cache_file=True,
                cache_file_names={k: os.path.join(training_args.output_dir, f'cache/tokenized_{k}.arrow') for k in raw_datasets},
                desc="Running tokenizer on dataset",
            )
    print("tokenized_dataset: ", tokenized_dataset)
    print("input_ids: ", tokenized_dataset["train"][6]['input_ids'])
    print("labels: ", tokenized_dataset["train"][1]['labels'])
    lm_datasets = tokenized_dataset
    # if training_args.do_train:
    train_dataset = lm_datasets['train']
    # print("train_dataset: ", train_dataset)
    logger.info(f"Num train_samples  {len(train_dataset)}")
    # if training_args.do_eval:
    eval_dataset = lm_datasets["dev"]
    # dataset_list = eval_dataset.train_test_split(test_size=0.3,seed=42)
    # eval_dataset = dataset_list["train"]
    # test_dataset = dataset_list["test"]
    test_dataset = lm_datasets["dev"]
    # print("test_dataset: ", test_dataset)
    # logger.info(f"Num eval_samples  {len(eval_dataset)},num test:{len(test_dataset)}")

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        # from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        torch_dtype=torch.bfloat16,
    )
    # model.lm_head = torch.nn.Linear(model.config.hidden_size, 2) 

    rank_allocator = None
    if training_args.apply_lora:
        with torch.no_grad():
            apply_lora_to_model(model, config=config)
        # rank allocator
        rank_allocator = RankAllocator(
            model,
            lora_r=training_args.lora_rank,
            target_rank=None,
            init_warmup=training_args.start_prune_steps,
            final_warmup=training_args.end_prune_steps,
            mask_interval=training_args.prune_every_steps,
            beta1=0.85,
            beta2=0.85,
            target_total_rank=int(training_args.target_total_rank),
            tb_writter=tb_writter,
            tb_writter_loginterval=training_args.prune_every_steps,
        )

    model = model.to(torch.bfloat16)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
        padding="longest"
    )
    # Initialize our Trainer
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=training_args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=data_collator, batch_size=training_args.per_device_eval_batch_size,
        shuffle=False
    )
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    tunable = training_args.tunable_param_names.strip().split(",")
    # print([n for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
    #                    and any(nd in n for nd in tunable) ])
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                        ],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                       ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    training_args.max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch
    overrode_max_train_steps = True
    rank_allocator.set_total_step(training_args.max_train_steps)
    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps * training_args.gradient_accumulation_steps,
        num_training_steps=training_args.max_train_steps * training_args.gradient_accumulation_steps,
    )
    # accelerator
    accelerator_log_kwargs = {}
    accelerator_log_kwargs["project_dir"] = training_args.output_dir
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        **accelerator_log_kwargs
    )
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
        os.makedirs(training_args.output_dir, exist_ok=True)
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    training_args.max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    training_args.num_train_epochs = math.ceil(training_args.max_train_steps / num_update_steps_per_epoch)
    if training_args.do_train:
        # Train!
        total_batch_size = training_args.per_device_train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {training_args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(training_args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0
        steps = 0
        starting_epoch = 0
        # Training stage 1:
        # Phase 1: Preliminary training of LORA parameters
        total_model_params = 0
        num_trained_params = 0
        for n, p in model.named_parameters():
            # if ("lora_vector" in n):
            # if ("lora_vector" in n) or ("lora_a" in n) or ("lora_b" in n) or ("adapter" in n):
            # if ("lora_a" in n) or ("lora_b" in n) or ("adapter" in n):
            if any(nd in n for nd in tunable):
                p.requires_grad = True
            else:
                p.requires_grad = False
            if p.requires_grad:
                num_trained_params += p.numel()
            else:
                total_model_params += p.numel()
            # print(n, p.requires_grad, p.numel())
        logger.info("Total Model Parameters: {}, "
                    "Trainable Parameters: {}".format(
            total_model_params, num_trained_params))
        # time.sleep(10)
        # training loop
        best_loss = 1000000000000
        best_steps = None
        max_patience = training_args.max_patience
        patience = 0
        for epoch in range(starting_epoch, training_args.num_train_epochs):
            total_loss = 0
            active_dataloader = train_dataloader
            for step, batch in enumerate(active_dataloader):
                steps += 1
                # print("batch: ", batch["input_ids"].shape)
                # print("batch: ", batch)
                model.train()
                # set weight coeff
                weight_coeff_value = sample_coeff()
                set_lora_weight_coeffs(model, config=config, weight_coeff_value=weight_coeff_value)
                # print("input_ids" , batch["input_ids"].shape)
                # print("label" , batch["labels"].shape)
                # for k, v in batch.items():
                #     print(f"{k}: {v.shape}")
                outputs = model(**batch)
                # logger.info("outputs:{}".format(outputs.logits.shape))
                # pdb.set_trace() 
                loss_1 = outputs.loss / training_args.gradient_accumulation_steps
                loss_2 = compute_svd_orth_regu(model)
                # pred = model.generate(torch.tensor(batch["input_ids"]).to(torch.device("cuda:0")),attention_mask=torch.tensor(batch["attention_mask"]).to(torch.device("cuda:0")),generation_config=generation_config)
                # print("input_ids: ", len(batch["input_ids"]))
                # print("pred: ", pred[0].shape)
                # response_text = tokenizer.decode(pred[0],skip_special_tokens=True).strip().lower()
                # print("response_text: ", response_text)
                loss = loss_1 + loss_2
                # print("loss_1: ", loss_1)
                # print("loss_2: ", loss_2)
                loss.backward()
                set_lora_weight_coeffs(model, config=config, weight_coeff_value=1.0)
                if tb_writter is not None:
                    tb_writter.add_scalar("train/loss", loss_1, completed_steps)
                    # tb_writter.add_scalar("train/regu_loss", loss_2.to(torch.float32).detach(), completed_steps)
                if random.uniform(0, 1) < 0.01:
                    print("loss: ", loss)
                # if random.uniform(0, 1) < 0.05:
                #     for n, p in model.named_parameters():
                #         if "lora" in n and p.requires_grad:
                #             print(n, p.grad[0])
                if steps % training_args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    # print("steps: ", steps)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        training_args.max_grad_norm
                    )
                    completed_steps += 1
                    # print("completed_steps: ", completed_steps)
                    optimizer.step()
                    # Perform pruning
                    if rank_allocator is not None:
                        curr_rank, mask_threshold = rank_allocator.update_and_mask(model, completed_steps)
                        # TODO: Record the schemes where the rank is masked
                        rank_pattern_ = rank_allocator.get_rank_pattern()
                        ranks_to_mask = []
                        num_ranks_masked = 0.0
                        for k_, v_ in rank_pattern_.items():
                            # print(v_)
                            # model.layers.10.mlp.up_proj.lora_E
                            for rank_idx, m in enumerate(v_):
                                if m == 1:
                                    continue
                                layer_idx = int(k_.split(".")[2])
                                module_name = k_.split(".")[4]
                                module_idx = {
                                    "q_proj": 0,
                                    "k_proj": 1,
                                    "v_proj": 2,
                                    "o_proj": 3,
                                    "gate_proj": 4,
                                    "up_proj": 5,
                                    "down_proj": 6,
                                }[module_name]
                                ranks_to_mask.append(
                                    (layer_idx, module_idx, rank_idx)
                                )
                                if module_idx > 3:
                                    num_ranks_masked += (float(os.getenv("DIM_MULTIPLIER")) + 1) / 2
                                else:
                                    num_ranks_masked += 1
                        # print("num_ranks_masked: ", num_ranks_masked)
                        dump_json(
                            ranks_to_mask,
                            os.path.join(training_args.output_dir, "ranks_to_mask.json")
                        )
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    if completed_steps % training_args.eval_steps == 0 and completed_steps > 0  \
                        and completed_steps > training_args.max_train_steps - training_args.end_prune_steps:
                        # eval model with structural drop
                        eval_loss = eval_model(
                            model,
                            eval_dataloader,
                            # tokenizer
                        )
                        logger.info(f"completed_steps: {completed_steps}; eval loss: {eval_loss}")
                        if eval_loss < best_loss:
                            best_loss = eval_loss
                            best_steps = completed_steps
                            accelerator.wait_for_everyone()
                            unwrapped_model = accelerator.unwrap_model(model)
                            with torch.no_grad():
                                save_lora_modules(
                                    unwrapped_model,
                                    config,
                                    to_path=os.path.join(training_args.output_dir, "lora_state_dict.bin"),
                                )
                            if accelerator.is_main_process:
                                tokenizer.save_pretrained(training_args.output_dir)
                            patience = 0
                        else:
                            patience += 1
                        logger.info(f"current best_loss: {best_loss}; best_steps: {best_steps}")

                        if patience >= max_patience:
                            print(f"patience:{patience},")
                            break
            if completed_steps >= training_args.max_train_steps:
                break
            if patience >= max_patience:
                break
        logger.info("*" * 50)
        logger.info(f"best steps: {best_steps}; best loss: {best_loss}")
        logger.info("*" * 50)

    if training_args.do_generation:
        with torch.no_grad():
            lora_state_dict = torch.load(
                os.path.join(training_args.output_dir, "lora_state_dict.bin")
            )
            model.load_state_dict(lora_state_dict, strict=False)
            model = model.to(torch.bfloat16)
            # apply_trained_lora_to_model(model, config=config, lora_state_dict=lora_state_dict)
        model.eval()
        eval_loss = eval_model(
            model,
            eval_dataloader,
            tokenizer
        )
        # logger.info("*" * 50)
        # logger.info(f"eval_loss: {eval_loss};")
        # logger.info("*" * 50)
        #################
        # Inference on the test set
        #################
        print("start inference on test set...")   
        list_predicted_samples = []
        for samp in test_dataset:
            # print(samp)
            input_ids = samp["input_ids"]
            attention_mask = samp["attention_mask"]
            input_length = len(input_ids[0])
            # print("samp: ", samp)
            outputs = model.generate(
                torch.LongTensor(input_ids).to(torch.device("cuda:0")),
                attention_mask=torch.LongTensor(attention_mask).to(torch.device("cuda:0")),
                generation_config=generation_config,
            )
            # logits = outputs.logits
            # predictions = torch.argmax(outputs, dim=1)
            # response = outputs[0]
            # print("response: ", response)
            # eod_token_idx = None
            # for i in range(len(response)):
            #     if response[i] in [tokenizer.eos_token_id]:
            #         eod_token_idx = i
            #         break
            # if eod_token_idx is None:
            #     eod_token_idx = len(response) - 1
            # response = response[: eod_token_idx]
            response_text = tokenizer.decode(
                outputs[0],skip_special_tokens=True
            ).strip().lower()
            print("response_text: ", response_text)
            samp_copy = copy.deepcopy(samp)
            # samp_copy["pred"] = response_text
            # list_predicted_samples.append(
            #     samp_copy
            # )
            # samp_copy["pred"] = predictions.cpu().numpy().tolist()
            samp_copy["target"] = samp["labels"]
            list_predicted_samples.append(samp_copy)
            dump_jsonl(
                list_predicted_samples,
                os.path.join(training_args.output_dir, "test_predictions.json")
            )
            if len(list_predicted_samples) % 20 == 0 or len(list_predicted_samples) == len(test_dataset):
                with open(os.path.join(training_args.output_dir, "test_predictions.json"), "w", encoding="utf-8") as f:
                    for samp in list_predicted_samples:
                        f.write(
                            json.dumps(samp, ensure_ascii=False) + "\n"
                        )
                scores = calc_scores(os.path.join(training_args.output_dir, "test_predictions.json"))
                print("*" * 50)
                print("scores: ", scores)
                print("*" * 50)

if __name__ == "__main__":
    main()

