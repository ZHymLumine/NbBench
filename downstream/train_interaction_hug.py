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
import datasets

import numpy as np
from torch.utils.data import Dataset
from scipy.special import expit

os.environ["WANDB_DISABLED"] = "true"


from transformers import Trainer, TrainingArguments, BertTokenizer, RobertaTokenizer,EsmTokenizer, EsmModel, AutoConfig, AutoModel, EarlyStoppingCallback, AutoTokenizer, EsmConfig
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_path)
sys.path.append(parent_dir)

from model.nanobert.modeling_nanobert import NanoBertForBindingSequenceClassification
from model.vhhbert.modeling_vhhbert import VHHBertForBindingSequenceClassification
from model.antiberty.modeling_antiberty import AntiBERTyForBindingSequenceClassification
from model.igbert.modeling_igbert import IgBertForBindingSequenceClassification
from model.ablang_h.modeling_ablang_h import AbLangHForBindingSequenceClassification
from model.ablang_l.modeling_ablang_l import AbLangLForBindingSequenceClassification
from model.antiberta2.modeling_antiberta2 import Antiberta2ForBindingSequenceClassification
from model.antiberta2.modeling_antiberta2_cssp import Antiberta2CSSPForBindingSequenceClassification
from model.protbert.modeling_protbert import ProtBertForBindingSequenceClassification
from model.esm2.modeling_esm import ESMForBindingSequenceClassification


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
    data_train_path: str = field(default=None, metadata={"help": "Path to the training data."})
    data_val_path: str = field(default=None, metadata={"help": "Path to the validation data."})
    data_test_path: str = field(default=None, metadata={"help": "Path to the test data. is list"})
    use_hf_dataset: bool = field(default=False, metadata={"help": "Whether to use HuggingFace dataset."})
    dataset_name: str = field(default=None, metadata={"help": "Name of the HuggingFace dataset."})
    antigen_embeddings_path: str = field(default=None, metadata={"help": "No longer used. Fixed to use data_path/antigen_embeddings.pt as the antigen embedding file path."})
    antigen_embeddings_file: str = field(default="antigen_embeddings.pt", metadata={"help": "No longer used. Fixed to use data_path/antigen_embeddings.pt as the antigen embedding file."})
    download_to_local: bool = field(default=False, metadata={"help": "Whether to download the HuggingFace dataset to local data_path."})
    force_download: bool = field(default=False, metadata={"help": "Whether to force download even if the dataset already exists locally."})

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
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    evaluation_strategy: str = field(default="steps")
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

        # Initialize variables to None to avoid potential reference issues
        self.nanobody = None
        self.antigen = None 
        self.labels = None
        self.antigen_embeddings = None
        self.num_labels = 2

        # load data from the disk        
        with open(data_path, "r") as f:
            reader = csv.DictReader(f)
            logging.warning("Perform single sequence classification...")
            
            # Check if required columns exist
            if reader.fieldnames is None:
                raise ValueError(f"CSV file {data_path} is empty or has no headers")
                
            required_cols = ["VHH_sequence", "Ag_sequence", "label"]
            missing_cols = [col for col in required_cols if col not in reader.fieldnames]
            
            if missing_cols:
                raise ValueError(f"Missing required columns in CSV file: {', '.join(missing_cols)}")
                
            data = [(row["VHH_sequence"], row["Ag_sequence"], int(row["label"])) for row in reader]
            
        if not data:
            raise ValueError(f"No data found in {data_path}")

        # Load antigen embeddings
        if not os.path.exists(embedding_path):
            raise ValueError(f"Could not find antigen embedding file: {embedding_path}")
            
        try:
            self.antigen_embeddings = torch.load(embedding_path)
        except Exception as e:
            raise ValueError(f"Error loading antigen embeddings from {embedding_path}: {e}")
            
        nanobody, antigen, labels = zip(*data)
        
        # ensure tokenier
        print(type(nanobody[0]))
        print(nanobody[0])
        test_example = tokenizer.tokenize(nanobody[0])
        print(test_example)
        print(len(test_example))
        print(tokenizer(nanobody[0]))
        self.labels = labels
        self.nanobody = nanobody
        self.antigen = antigen
        
    def __len__(self):
       return len(self.nanobody)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # Extract the corresponding antigen embedding
        try:
            antigen_sequence = self.antigen[i]
            if antigen_sequence not in self.antigen_embeddings:
                raise KeyError(f"Antigen sequence '{antigen_sequence}' not found in embeddings")
            antigen_embedding = self.antigen_embeddings[antigen_sequence]
            return dict(input_ids=self.nanobody[i], antigen_embedding=antigen_embedding, labels=self.labels[i])
        except Exception as e:
            raise RuntimeError(f"Error retrieving item {i}: {e}")



class HFDataset(Dataset):
    """Dataset for loading data from Hugging Face datasets."""

    def __init__(self, 
                 split: str,
                 dataset_name: str,
                 args,
                 tokenizer: transformers.PreTrainedTokenizer,
                 embedding_path: str,
                 ):

        super(HFDataset, self).__init__()
        
        # Initialize variables to None to avoid potential reference issues
        self.nanobody = None
        self.antigen = None
        self.labels = None
        
        # Load Hugging Face dataset
        logging.warning(f"Loading Hugging Face dataset: {dataset_name}, split: {split}")
        self.dataset = datasets.load_dataset(dataset_name, split=split)
        
        # Extract VHH sequences and antigen sequences
        if "VHH_sequence" in self.dataset.column_names and "Ag_sequence" in self.dataset.column_names:
            self.nanobody = self.dataset["VHH_sequence"]
            self.antigen = self.dataset["Ag_sequence"]
        else:
            # Try other possible column names
            for col in self.dataset.column_names:
                if col.lower() in ["vhh", "nanobody", "antibody", "ab_sequence", "antibody_sequence"]:
                    logging.warning(f"Using column '{col}' as nanobody sequence")
                    self.nanobody = self.dataset[col]
                if col.lower() in ["ag", "antigen", "ag_sequence", "antigen_sequence", "target"]:
                    logging.warning(f"Using column '{col}' as antigen sequence")
                    self.antigen = self.dataset[col]
        
        if self.nanobody is None or self.antigen is None:
            raise ValueError(f"Could not find nanobody or antigen sequence columns in dataset {dataset_name}, please check dataset format")
        
        # Extract labels
        if "label" in self.dataset.column_names:
            self.labels = self.dataset["label"]
            if not isinstance(self.labels[0], int):
                self.labels = [int(label) for label in self.labels]
        else:
            # Try other possible label column names
            for col in self.dataset.column_names:
                if col.lower() in ["labels", "class", "classes", "target", "targets", "binding", "interaction"]:
                    logging.warning(f"Using column '{col}' as label")
                    self.labels = self.dataset[col]
                    if not isinstance(self.labels[0], int):
                        self.labels = [int(label) for label in self.labels]
                    break
            else:
                raise ValueError(f"Could not find label column in dataset {dataset_name}, please check dataset format")
        
        # Load embeddings from local file
        if not os.path.exists(embedding_path):
            raise ValueError(f"Could not find antigen embedding file: {embedding_path}, please make sure the file exists")
        
        logging.warning(f"Loading embedding file from local path: {embedding_path}")
        self.antigen_embeddings = torch.load(embedding_path)
        
        # Test tokenizer
        print(f"Nanobody sequence type: {type(self.nanobody[0])}")
        print(f"Nanobody sequence example: {self.nanobody[0]}")
        print(f"Antigen sequence example: {self.antigen[0]}")
        print(f"Label example: {self.labels[0]}")
        test_example = tokenizer.tokenize(self.nanobody[0])
        print(f"Tokenizer test: {test_example}")
        print(f"Tokenizer result length: {len(test_example)}")
        print(tokenizer(self.nanobody[0]))
        
        self.num_labels = 2  # Binary classification task

    def __len__(self):
       return len(self.nanobody)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        try:
            antigen_sequence = self.antigen[i]
            if antigen_sequence not in self.antigen_embeddings:
                raise KeyError(f"Antigen sequence '{antigen_sequence}' not found in embeddings")
            antigen_embedding = self.antigen_embeddings[antigen_sequence]
            return dict(input_ids=self.nanobody[i], antigen_embedding=antigen_embedding, labels=self.labels[i])
        except Exception as e:
            raise RuntimeError(f"Error retrieving item {i}: {e}")

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        nanobody, antigen_embedding, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "antigen_embedding", "labels"))
        
        output = self.tokenizer(nanobody, padding='longest', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

        input_ids = output["input_ids"]
        attention_mask = output["attention_mask"]
        labels = torch.Tensor(labels).long()
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            antigen_embedding=torch.stack(antigen_embedding),
            labels=labels,
        )

"""
Manually calculate the accuracy, f1, precision, recall, auprc, auroc with sklearn.
"""
def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    if logits.shape[-1] == 1:
        logits = logits.squeeze(-1)

    positive_class_probabilities = expit(logits)

    # logit > 0 → prediction: 1
    predictions = (positive_class_probabilities >= 0.5).astype(int)

    # predictions = np.argmax(logits, axis=-1)
    # if logits.shape[-1] == 1:
        # logits = logits.squeeze(-1)

    # probabilities = expit(logits)
    # positive_class_probabilities = probabilities[:, 1]

    return {
        "accuracy": sklearn.metrics.accuracy_score(labels, predictions),
        "f1": sklearn.metrics.f1_score(labels, predictions, zero_division=0),
        "precision": sklearn.metrics.precision_score(labels, predictions, zero_division=0),
        "recall": sklearn.metrics.recall_score(labels, predictions, zero_division=0),
        "auprc": sklearn.metrics.average_precision_score(labels, positive_class_probabilities),
        "auroc": sklearn.metrics.roc_auc_score(labels, positive_class_probabilities),
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

    print(f"training_args: {training_args}")
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


    if data_args.use_hf_dataset and data_args.dataset_name:
        # Loading from Hugging Face dataset
        print(f"Loading data from Hugging Face: {data_args.dataset_name}")
        
        # Directly specify embedding file path to local fixed path
        embedding_path = os.path.join(data_args.data_path, 'antigen_embeddings.pt')
        print(f"Using local embedding file: {embedding_path}")
        
        # Ensure embedding file exists
        if not os.path.exists(embedding_path):
            raise ValueError(f"Could not find antigen embedding file: {embedding_path}, please make sure the file exists")
        
        # Directly use Hugging Face dataset, but still use local embedding file
        train_dataset = HFDataset(
            split="train", 
            dataset_name=data_args.dataset_name, 
            args=training_args, 
            tokenizer=tokenizer,
            embedding_path=embedding_path
        )
        val_dataset = HFDataset(
            split="validation", 
            dataset_name=data_args.dataset_name, 
            args=training_args, 
            tokenizer=tokenizer,
            embedding_path=embedding_path
        )
        test_dataset = HFDataset(
            split="test", 
            dataset_name=data_args.dataset_name, 
            args=training_args, 
            tokenizer=tokenizer,
            embedding_path=embedding_path
        )
    else:
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
        model =  NanoBertForBindingSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
            )
    elif training_args.model_type == 'vhhbert':  
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = VHHBertForBindingSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )        
    elif training_args.model_type == 'antiberty':
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = AntiBERTyForBindingSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )              
    elif training_args.model_type == 'igbert':
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = IgBertForBindingSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )
    elif training_args.model_type == 'antiberta2':
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = Antiberta2ForBindingSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )
    elif training_args.model_type == 'antiberta2_cssp':
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = Antiberta2CSSPForBindingSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )
    elif training_args.model_type == 'ablang_h':
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = AbLangHForBindingSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )   
    elif training_args.model_type == 'ablang_l':
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = AbLangLForBindingSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )
    elif training_args.model_type == 'protbert':
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = ProtBertForBindingSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )
    elif "esm-2" in training_args.model_type:
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = ESMForBindingSequenceClassification(
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

    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        results = trainer.evaluate(eval_dataset=test_dataset)
        print("on the test set:", results, "\n", results_path)
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "test_results.json"), "w") as f:
            json.dump(results, f, indent=4)

    # if training_args.eval_and_save_results:
    #     results_path = os.path.join(training_args.output_dir, "results_manual", training_args.run_name)
    #     os.makedirs(results_path, exist_ok=True)
        
    #     test_metrics = manual_evaluate(
    #         model=model,
    #         test_dataset=test_dataset,
    #         data_collator=data_collator,
    #         batch_size=training_args.per_device_eval_batch_size
    #     )
        
    #     print("on the test set:", test_metrics)
    #     with open(os.path.join(results_path, "test_results.json"), "w") as f:
    #         json.dump(test_metrics, f, indent=4)
         
if __name__ == "__main__":
    # To load dataset from Hugging Face, use the following command:
    # python train_interaction_hug.py --use_hf_dataset=True --dataset_name="ZYMScott/nanobody-antigen-binding" --model_type=nanobert --data_path="./data/downstream/binding"
    
    # Note: You must provide antigen_embeddings.pt file in the data_path directory, e.g.: ./data/downstream/binding/antigen_embeddings.pt
    # This file must exist, otherwise the program will terminate
    
    # If you want to download the dataset to a local directory, use:
    # python train_interaction_hug.py --use_hf_dataset=True --dataset_name="ZYMScott/nanobody-antigen-binding" --download_to_local=True --data_path="./data/downstream/binding" --model_type=nanobert
    
    # Force re-download the dataset (even if it already exists locally):
    # python train_interaction_hug.py --use_hf_dataset=True --dataset_name="ZYMScott/nanobody-antigen-binding" --download_to_local=True --data_path="./data/downstream/binding" --force_download=True --model_type=nanobert
    
    train()

