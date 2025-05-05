import os
import csv
import copy
import json
import logging
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from typing import Optional, Dict, Sequence, Tuple, List

import torch
import random
import sklearn
import scipy
import transformers

import numpy as np
from torch.utils.data import Dataset
import pdb


os.environ["WANDB_DISABLED"] = "true"


from transformers import Trainer, TrainingArguments, BertTokenizer, RobertaTokenizer,EsmTokenizer, EsmModel, AutoConfig, AutoModel, EarlyStoppingCallback, AutoTokenizer
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_path)
sys.path.append(parent_dir)

from model.nanobert.modeling_nanobert import NanoBertForAminoAcidLevel
from model.vhhbert.modeling_vhhbert import VHHBertForAminoAcidLevel
from model.antiberty.modeling_antiberty import AntiBERTyForAminoAcidLevel
from model.igbert.modeling_igbert import IgBertForAminoAcidLevel
from model.ablang_h.modeling_ablang_h import AbLangHForAminoAcidLevel
from model.ablang_l.modeling_ablang_l import AbLangLForAminoAcidLevel
from model.antiberta2.modeling_antiberta2 import Antiberta2ForAminoAcidLevel
from model.antiberta2.modeling_antiberta2_cssp import Antiberta2CSSPForAminoAcidLevel
from model.protbert.modeling_protbert import ProtBertForAminoAcidLevel
from model.esm2.modeling_esm import ESMForAminoAcidLevel


early_stopping = EarlyStoppingCallback(early_stopping_patience=20)
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    use_alibi: bool = field(default=True, metadata={"help": "whether to use alibi"})
    use_features: bool = field(default=True, metadata={"help": "whether to use alibi"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})
    tokenizer_name_or_path: Optional[str] = field(default="")
    freeze: bool = field(default=True, metadata={"help": "whether to freeze the model"})

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer."})
    data_train_path: str = field(default=None, metadata={"help": "Path to the training data."})
    data_val_path: str = field(default=None, metadata={"help": "Path to the training data."})
    data_test_path: str = field(default=None, metadata={"help": "Path to the test data. is list"})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    evaluation_strategy: str = field(default="steps"),
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=1)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)
    fp16: bool = field(default=False)
    metric_for_best_model: str = field(default="token_accuracy")
    stage: str = field(default='0')
    model_type: str = field(default='protein')
    train_from_scratch: bool = field(default=False)
    log_dir: str = field(default="output")
    attn_implementation: str = field(default="eager")
    dataloader_num_workers: int = field(default=4)
    dataloader_prefetch_factor: int = field(default=2)
    report_to: str = field(default="none")

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # TODO
    # if torch.distributed.get_rank() >= 0:
    #     print("!!!!!!!!!!!!!", "Yes")
    #     torch.cuda.manual_seed_all(args.seed)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 data_path: str, args,
                 tokenizer: transformers.PreTrainedTokenizer, 
                 ):

        super(SupervisedDataset, self).__init__()

        # load data from the disk        
        with open(data_path, "r") as f:
            reader = csv.DictReader(f)
            logging.warning("Perform single sequence classification...")
            data = [
                    (row["seq"].replace("-", tokenizer.mask_token), row["label"])
                    for row in reader
            ]
        
        texts, labels = zip(*data)
        text = texts[0]
        
        # ensure tokenier
        print(type(texts[0]))
        print(texts[0])
        test_example = tokenizer.tokenize(texts[0], add_special_tokens=False)
        print(test_example)
        print(len(test_example))
        print(tokenizer(texts[0]))
        self.labels = labels
        self.num_labels = len(tokenizer)
        # print(f'labels: {self.labels}')
        print(f'num_labels: {self.num_labels}')
        self.texts = texts

    def __len__(self):
       return len(self.texts)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.texts[i], labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        seqs, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        
        output = self.tokenizer(seqs, padding='longest', add_special_tokens=False,max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        label_tokens = self.tokenizer(labels, padding='longest',add_special_tokens=False, truncation=True, return_tensors='pt')["input_ids"]
        
        input_ids = output["input_ids"]
        attention_mask = output["attention_mask"]
        labels = label_tokens.clone()
        labels[input_ids != self.tokenizer.mask_token_id] = -100
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )


def blosum_score(aa1: str, aa2: str) -> float:
    """return token id a and b's blosum score"""
    import blosum as bl
    blosum62 = bl.BLOSUM(62)
    try:
        return blosum62[aa1][aa2]
    except KeyError:
        try:
            return blosum62[aa2][aa1]
        except KeyError:
            return -4.0 

def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray, id_to_token:dict):

    predictions = np.argmax(logits, axis=-1)
    
    token_correct = 0
    token_total = 0
    cdr_correct = [0, 0, 0]  # CDR1, CDR2, CDR3 token-level correct
    cdr_total = [0, 0, 0]

    cdr_reconstruct_success = [0, 0, 0]

    blosum_total = 0
    blosum_count = 0
    cdr_blosum_score = [0, 0, 0]
    cdr_blosum_count = [0, 0, 0]

    num_samples = labels.shape[0]
    for i in range(num_samples):
        label_row = labels[i]
        pred_row = predictions[i]

        valid_positions = np.where(label_row != -100)[0]
        if len(valid_positions) == 0:
            continue

        boundaries = np.split(valid_positions, np.where(np.diff(valid_positions) != 1)[0] + 1)

        if len(boundaries) != 3:
            continue 

        for j, cdr_pos in enumerate(boundaries):
            cdr_label = label_row[cdr_pos]
            cdr_pred = pred_row[cdr_pos]
            cdr_correct[j] += np.sum(cdr_label == cdr_pred)
            cdr_total[j] += len(cdr_pos)
            if np.all(cdr_label == cdr_pred):
                cdr_reconstruct_success[j] += 1
            for pos in cdr_pos:
                a = label_row[pos]
                b = pred_row[pos]
                score = blosum_score(id_to_token[a], id_to_token[b])
                cdr_blosum_score[j] += score
                cdr_blosum_count[j] += 1

                blosum_total += score
                blosum_count += 1

        token_correct += np.sum(label_row[valid_positions] == pred_row[valid_positions])
        token_total += len(valid_positions)

    # Token-level overall accuracy
    token_accuracy = token_correct / token_total if token_total > 0 else 0.0

    # Per-CDR token-level accuracy
    cdr1_acc = cdr_correct[0] / cdr_total[0] if cdr_total[0] > 0 else 0.0
    cdr2_acc = cdr_correct[1] / cdr_total[1] if cdr_total[1] > 0 else 0.0
    cdr3_acc = cdr_correct[2] / cdr_total[2] if cdr_total[2] > 0 else 0.0

    # CDR-level reconstruction rate
    cdr1_recon = cdr_reconstruct_success[0] / num_samples
    cdr2_recon = cdr_reconstruct_success[1] / num_samples
    cdr3_recon = cdr_reconstruct_success[2] / num_samples
    overall_recon = sum(cdr_reconstruct_success) / (3 * num_samples)

    return {
        "token_accuracy": token_accuracy,
        "cdr1_accuracy": cdr1_acc,
        "cdr2_accuracy": cdr2_acc,
        "cdr3_accuracy": cdr3_acc,
        "cdr1_reconstruction_rate": cdr1_recon,
        "cdr2_reconstruction_rate": cdr2_recon,
        "cdr3_reconstruction_rate": cdr3_recon,
        "overall_cdr_reconstruction_rate": overall_recon,
        "blosum_total_mean_score": blosum_total / blosum_count if blosum_count else 0.0,
        "cdr1_blosum_mean_score": cdr_blosum_score[0] / cdr_blosum_count[0] if cdr_blosum_count[0] else 0.0,
        "cdr2_blosum_mean_score": cdr_blosum_score[1] / cdr_blosum_count[1] if cdr_blosum_count[1] else 0.0,
        "cdr3_blosum_mean_score": cdr_blosum_score[2] / cdr_blosum_count[2] if cdr_blosum_count[2] else 0.0,
    }

"""
Compute metrics used for huggingface trainer.
"""
def compute_metrics_closure(tokenizer):
    id_to_token = {v: k for k, v in tokenizer.get_vocab().items()}
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return calculate_metric_with_sklearn(logits, labels, id_to_token)
    return compute_metrics

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args)
    # load tokenizer
    if training_args.model_type in ['nanobert', 'vhhbert', 'antiberty', 'iglm']:
        tokenizer = RobertaTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )
    elif "esm-2" in training_args.model_type:
        tokenizer = EsmTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )
    else:
        tokenizer = RobertaTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )


    # define datasets and data collator
    train_dataset = SupervisedDataset(tokenizer=tokenizer, args=training_args,
                                     data_path=os.path.join(data_args.data_path, data_args.data_train_path))
    val_dataset = SupervisedDataset(tokenizer=tokenizer, args=training_args,
                                     data_path=os.path.join(data_args.data_path, data_args.data_val_path))
    test_dataset = SupervisedDataset(tokenizer=tokenizer, args=training_args,
                                     data_path=os.path.join(data_args.data_path, data_args.data_test_path))
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer,args=training_args)
    print(f'# train: {len(train_dataset)},val:{len(val_dataset)},test:{len(test_dataset)}')

    # load model
    if training_args.model_type == 'nanobert':
        print(training_args.model_type)
        print('Loading nanobert model')
        print(train_dataset.num_labels)
        model =  NanoBertForAminoAcidLevel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
            )
    elif training_args.model_type == 'vhhbert':  
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        print(f"model_args: {model_args}")
        model = VHHBertForAminoAcidLevel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )
    elif training_args.model_type == 'antiberty':
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = AntiBERTyForAminoAcidLevel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )        
    elif training_args.model_type == 'iglm':
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = IgLMForAminoAcidLevel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )        
    elif training_args.model_type == 'igbert':
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = IgBertForAminoAcidLevel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )
    elif training_args.model_type == 'antiberta2':
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = Antiberta2ForAminoAcidLevel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )
    elif training_args.model_type == 'antiberta2_cssp':
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = Antiberta2CSSPForAminoAcidLevel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )
    elif training_args.model_type == 'ablang_h':
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = AbLangHForAminoAcidLevel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )       
    elif training_args.model_type == 'ablang_l':
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = AbLangLForAminoAcidLevel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )
    elif training_args.model_type == 'protbert':
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = ProtBertForAminoAcidLevel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )
    elif "esm-2" in training_args.model_type:
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        print(f"model_args: {model_args}")
        print(f"model_args type:{type(model_args)}")
        model = ESMForAminoAcidLevel(
            model_args,
            num_labels=train_dataset.num_labels,
        )


    # define trainer
    trainer = transformers.Trainer(model=model,
                                   tokenizer=tokenizer,
                                   args=training_args,
                                   compute_metrics=compute_metrics_closure(tokenizer),
                                   train_dataset=train_dataset,
                                   eval_dataset=val_dataset,
                                   data_collator=data_collator,
                                   callbacks=[early_stopping],
                                   )
    trainer.train()

    if training_args.save_model:
        trainer.save_state()
        #safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # get the evaluation results from trainer
    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        results = trainer.evaluate(eval_dataset=test_dataset)
        print("on the test set:", results, "\n", results_path)
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "test_results.json"), "w") as f:
            json.dump(results, f, indent=4)
         




if __name__ == "__main__":
    
    train()