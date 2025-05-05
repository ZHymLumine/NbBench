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
import pdb


os.environ["WANDB_DISABLED"] = "true"


from transformers import Trainer, TrainingArguments, BertTokenizer, RobertaTokenizer,EsmTokenizer, EsmModel, AutoConfig, AutoModel, EarlyStoppingCallback, AutoTokenizer
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_path)
sys.path.append(parent_dir)

from model.nanobert.modeling_nanobert import NanoBertForSequenceClassification
from model.vhhbert.modeling_vhhbert import VHHBertForSequenceClassification
from model.antiberty.modeling_antiberty import AntiBERTyForSequenceClassification
from model.igbert.modeling_igbert import IgBertForSequenceClassification
from model.ablang_h.modeling_ablang_h import AbLangHForSequenceClassification
from model.ablang_l.modeling_ablang_l import AbLangLForSequenceClassification
from model.antiberta2.modeling_antiberta2 import Antiberta2ForSequenceClassification
from model.antiberta2.modeling_antiberta2_cssp import Antiberta2CSSPForSequenceClassification
from model.protbert.modeling_protbert import ProtBertForSequenceClassification
from model.esm2.modeling_esm import ESMForSequenceClassification


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
    use_hf_dataset: bool = field(default=False, metadata={"help": "Whether to use HuggingFace dataset."})
    dataset_name: str = field(default=None, metadata={"help": "Name of the HuggingFace dataset."})

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
    metric_for_best_model: str = field(default="rmse")
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

        # Initialize variables
        self.texts = None
        self.labels = None
        self.num_labels = None

        # load data from the disk        
        with open(data_path, "r") as f:
            reader = csv.DictReader(f)
            logging.warning("Perform thermostability prediction...")

            # Check if required columns exist
            if reader.fieldnames is None:
                raise ValueError(f"CSV file {data_path} is empty or has no headers")
                
            required_cols = ["seq", "tm"]
            missing_cols = [col for col in required_cols if col not in reader.fieldnames]
            
            if missing_cols:
                raise ValueError(f"Missing required columns in CSV file: {', '.join(missing_cols)}")
                
            data = [(row["seq"], float(row["tm"])) for row in reader]
            
        if not data:
            raise ValueError(f"No data found in {data_path}")
        
        texts, labels = zip(*data)
        
        # ensure tokenizer works
        print(type(texts[0]))
        print(texts[0])
        test_example = tokenizer.tokenize(texts[0])
        print(test_example)
        print(len(test_example))
        print(tokenizer(texts[0]))
        self.labels = labels
        self.num_labels = 1  # Regression task
        self.texts = texts

    def __len__(self):
       return len(self.texts)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.texts[i], labels=self.labels[i])

class HFDataset(Dataset):
    """Dataset for loading data from Hugging Face datasets."""

    def __init__(self, 
                 split: str,
                 dataset_name: str,
                 args,
                 tokenizer: transformers.PreTrainedTokenizer,
                 ):

        super(HFDataset, self).__init__()
        
        # Initialize variables
        self.texts = None
        self.labels = None
        
        # Load Hugging Face dataset
        logging.warning(f"Loading Hugging Face dataset: {dataset_name}, split: {split}")
        self.dataset = datasets.load_dataset(dataset_name, split=split)
        
        # Extract sequence data
        if "seq" in self.dataset.column_names:
            self.texts = self.dataset["seq"]
        else:
            # Try to find a suitable sequence column
            for col in self.dataset.column_names:
                if col.lower() in ["sequence", "protein", "amino_acid", "input", "nanobody_seq", "antibody_seq"]:
                    logging.warning(f"Using column '{col}' as sequence input")
                    self.texts = self.dataset[col]
                    break
            else:
                raise ValueError(f"No sequence data column found in dataset {dataset_name}, please check the dataset format")
        
        # Extract label data (thermostability values)
        if "label" in self.dataset.column_names:
            self.labels = self.dataset["label"]
        else:
            # Try to find a suitable label column
            for col in self.dataset.column_names:
                if col.lower() in ["temperature", "thermostability", "melting_temp", "melting_temperature", "tm_value"]:
                    logging.warning(f"Using column '{col}' as label")
                    self.labels = self.dataset[col]
                    break
            else:
                raise ValueError(f"No thermostability column found in dataset {dataset_name}, please check the dataset format")
        
        # Display info about the data
        print(f"Seq type: {type(self.texts[0])}")
        print(f"Seq example: {self.texts[0]}")
        print(f"Label example: {self.labels[0]}")
        test_example = tokenizer.tokenize(self.texts[0])
        print(f"Test tokenizer result: {test_example}")
        print(f"Test tokenizer length: {len(test_example)}")
        
        self.num_labels = 1  # Regression task for thermostability prediction

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

        output = self.tokenizer(seqs, padding='longest', add_special_tokens=True, max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        input_ids = output["input_ids"]
        attention_mask = output["attention_mask"]

        labels = torch.Tensor(labels).float()
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )


"""
Manually calculate the regression metrics: MSE, RMSE, MAE, R2 using sklearn.
"""
def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray):
    return {
        "mse": sklearn.metrics.mean_squared_error(labels, predictions),
        "rmse": np.sqrt(sklearn.metrics.mean_squared_error(labels, predictions)),
        "mae": sklearn.metrics.mean_absolute_error(labels, predictions),
        "r2": sklearn.metrics.r2_score(labels, predictions),
    }

"""
Compute metrics used for huggingface trainer.
"""
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Regression task outputs are squeezed
    predictions = np.squeeze(predictions)
    labels = np.squeeze(labels)
    return calculate_metric_with_sklearn(predictions, labels)

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

    # Load dataset either from Hugging Face or local files
    if data_args.use_hf_dataset and data_args.dataset_name:
        # Loading from Hugging Face dataset
        print(f"Loading data from Hugging Face: {data_args.dataset_name}")
        
        train_dataset = HFDataset(
            split="train", 
            dataset_name=data_args.dataset_name, 
            args=training_args, 
            tokenizer=tokenizer
        )
        val_dataset = HFDataset(
            split="validation", 
            dataset_name=data_args.dataset_name, 
            args=training_args, 
            tokenizer=tokenizer
        )
        test_dataset = HFDataset(
            split="test", 
            dataset_name=data_args.dataset_name, 
            args=training_args, 
            tokenizer=tokenizer
        )
    else:
        # Loading from local files
        train_dataset = SupervisedDataset(tokenizer=tokenizer, args=training_args,
                                        data_path=os.path.join(data_args.data_path, data_args.data_train_path))
        val_dataset = SupervisedDataset(tokenizer=tokenizer, args=training_args,
                                        data_path=os.path.join(data_args.data_path, data_args.data_val_path))
        test_dataset = SupervisedDataset(tokenizer=tokenizer, args=training_args,
                                        data_path=os.path.join(data_args.data_path, data_args.data_test_path))
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, args=training_args)
    print(f'# train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}')

    # load model
    if training_args.model_type == 'nanobert':
        print(training_args.model_type)
        print('Loading nanobert model')
        model = NanoBertForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
            )
    elif training_args.model_type == 'vhhbert':  
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = VHHBertForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )        
    elif training_args.model_type == 'antiberty':
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = AntiBERTyForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )              
    elif training_args.model_type == 'igbert':
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = IgBertForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )
    elif training_args.model_type == 'antiberta2':
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = Antiberta2ForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )
    elif training_args.model_type == 'antiberta2_cssp':
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = Antiberta2CSSPForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )
    elif training_args.model_type == 'ablang_h':
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = AbLangHForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )       
    elif training_args.model_type == 'ablang_l':
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = AbLangLForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )
    elif training_args.model_type == 'protbert':
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = ProtBertForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )
    elif "esm-2" in training_args.model_type:
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = ESMForSequenceClassification(
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
        
if __name__ == "__main__":
    # To load dataset from Hugging Face, use the following command:
    # python train_thermo_hug.py --use_hf_dataset=True --dataset_name="<dataset_name>" --model_type=nanobert
    
    train() 