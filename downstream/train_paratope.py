import os
import csv
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Tuple, List

import torch
import random
import sklearn
import scipy
import transformers

import numpy as np
from torch.utils.data import Dataset
import pdb
from scipy.special import expit


os.environ["WANDB_DISABLED"] = "true"


from transformers import Trainer, TrainingArguments, BertTokenizer, RobertaTokenizer,EsmTokenizer, EsmModel, AutoConfig, AutoModel, EarlyStoppingCallback, AutoTokenizer
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_path)
sys.path.append(parent_dir)

from model.nanobert.modeling_nanobert import NanoBertForParatope
from model.vhhbert.modeling_vhhbert import VHHBertForParatope
from model.antiberty.modeling_antiberty import AntiBERTyForParatope
from model.igbert.modeling_igbert import IgBertForParatope
from model.ablang_h.modeling_ablang_h import AbLangHForParatope
from model.ablang_l.modeling_ablang_l import AbLangLForParatope
from model.antiberta2.modeling_antiberta2 import Antiberta2ForParatope
from model.antiberta2.modeling_antiberta2_cssp import Antiberta2CSSPForParatope
from model.protbert.modeling_protbert import ProtBertForParatope
from model.esm2.modeling_esm import ESMForParatope


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
    metric_for_best_model: str = field(default="accuracy")
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
                 embedding_path: str,
                 ):

        super(SupervisedDataset, self).__init__()

        # load data from the disk        
        with open(data_path, "r") as f:
            reader = csv.DictReader(f)
            logging.warning("Perform single sequence classification...")
            data = [
                    (row["seq_nanobody"], row["seq_antigen"],list(map(int, row["paratope"])))
                    for row in reader
            ]
        
        antigen_embeddings = torch.load(embedding_path)
        nanobody, antigen, labels = zip(*data)
        
        # ensure tokenier
        print(type(nanobody[0]))
        print(nanobody[0])
        test_example = tokenizer.tokenize(nanobody[0])
        print(test_example)
        print(len(test_example))
        print(tokenizer(nanobody[0]))
        self.labels = labels
        self.num_labels = 2
        self.nanobody = nanobody
        self.antigen = antigen
        self.antigen_embeddings = antigen_embeddings

    def __len__(self):
       return len(self.nanobody)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        antigen_embedding = self.antigen_embeddings[self.antigen[i]]
        return dict(input_ids=self.nanobody[i], antigen_embedding=antigen_embedding, labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        nanobody, antigen_embedding, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "antigen_embedding", "labels"))
        

        output = self.tokenizer(nanobody, padding='longest', add_special_tokens=False,max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        
        input_ids = output["input_ids"]
        attention_mask = output["attention_mask"]

        max_len = max(map(len, nanobody))
        padded_labels = [
            label + [-100] * (max_len - len(label)) for label in labels
        ]

        labels = torch.Tensor(padded_labels).long()
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            antigen_embedding=torch.stack(antigen_embedding),
            labels=labels,
        )


"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
"""
def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    if logits.shape[-1] == 1:
        logits = logits.squeeze(-1)
    predictions = np.argmax(logits, axis=-1)
    mask = labels != -100
    valid_preds = predictions[mask]
    valid_labels = labels[mask]

    positive_class_probabilities = expit(logits)
    valid_positive_class_probabilities = positive_class_probabilities[mask][:,1]
    # logit > 0 → prediction: 1
    # predictions = (positive_class_probabilities >= 0.5).astype(int)
    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_preds),
        "f1": sklearn.metrics.f1_score(valid_labels, valid_preds, average="macro", zero_division=0),
        "precision": sklearn.metrics.precision_score(valid_labels, valid_preds, average="macro", zero_division=0),
        "recall": sklearn.metrics.recall_score(valid_labels, valid_preds, average="macro", zero_division=0),
        "auprc": sklearn.metrics.average_precision_score(valid_labels, valid_positive_class_probabilities),
        "auroc": sklearn.metrics.roc_auc_score(valid_labels, valid_positive_class_probabilities),
    }

"""
Compute metrics used for huggingface trainer.
"""
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return calculate_metric_with_sklearn(logits, labels)

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
                                     data_path=os.path.join(data_args.data_path, data_args.data_train_path), embedding_path=os.path.join(data_args.data_path, 'antigen_embeddings.pt'))
    val_dataset = SupervisedDataset(tokenizer=tokenizer, args=training_args,
                                     data_path=os.path.join(data_args.data_path, data_args.data_val_path), embedding_path=os.path.join(data_args.data_path, 'antigen_embeddings.pt'))
    test_dataset = SupervisedDataset(tokenizer=tokenizer, args=training_args,
                                     data_path=os.path.join(data_args.data_path, data_args.data_test_path), embedding_path=os.path.join(data_args.data_path, 'antigen_embeddings.pt'))
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer,args=training_args)
    print(f'# train: {len(train_dataset)},val:{len(val_dataset)},test:{len(test_dataset)}')

    # load model
    if training_args.model_type == 'nanobert':
        print(training_args.model_type)
        print('Loading nanobert model')
        model =  NanoBertForParatope.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
            )
    elif training_args.model_type == 'vhhbert':  
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = VHHBertForParatope.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )        
    elif training_args.model_type == 'antiberty':
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = AntiBERTyForParatope.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )           
    elif training_args.model_type == 'igbert':
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = IgBertForParatope.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )
    elif training_args.model_type == 'antiberta2':
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = Antiberta2ForParatope.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )
    elif training_args.model_type == 'antiberta2_cssp':
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = Antiberta2CSSPForParatope.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )
    elif training_args.model_type == 'ablang_h':
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = AbLangHForParatope.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )          
    elif training_args.model_type == 'ablang_l':
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = AbLangLForParatope.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )
    elif training_args.model_type == 'protbert':
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = ProtBertForParatope.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )
    elif "esm-2" in training_args.model_type:
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = ESMForParatope(
            model_args,
            num_labels=train_dataset.num_labels,
        )

    # define trainer
    trainer = transformers.Trainer(model=model,
                                   tokenizer=tokenizer,
                                   args=training_args,
                                   compute_metrics=compute_metrics,
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

