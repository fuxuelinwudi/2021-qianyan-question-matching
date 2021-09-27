import os
import sys
import pickle
import random
import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer
from collections import defaultdict

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


sys.path.append('../../../src')
from src.models.nezha import *


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def batch2cuda(args, batch):
    return {item: value.to(args.device) for item, value in list(batch.items())}


def build_model_and_tokenizer(args):
    tokenizer = BertTokenizer.from_pretrained(args.vocab_path)
    model = NeZhaSequenceClassification_F.from_pretrained(args.model_path)
    model.to(args.device)

    return tokenizer, model


class PGD:
    def __init__(self, args, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}
        self.epsilon = args.epsilon
        self.emb_name = args.emb_name
        self.alpha = args.alpha

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=0.0, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            min_lr, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def build_optimizer(args, model, train_steps):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=train_steps * args.warmup_ratio,
                                                num_training_steps=train_steps)

    return optimizer, scheduler


def save_model(args, model, tokenizer):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_save_path = os.path.join(args.output_path, f'last-checkpoint')
    model_to_save.save_pretrained(model_save_path)
    tokenizer.save_vocabulary(model_save_path)

    print(f'model saved in : {model_save_path} .')


def read_data(args, tokenizer):
    train_df = pd.read_csv(args.train_path, header=None, sep='\t')
    inputs = defaultdict(list)
    for i, row in tqdm(train_df.iterrows(), desc=f'Preprocessing train data', total=len(train_df)):
        sentence_a, sentence_b, label = row[0], row[1], row[2]
        build_bert_inputs(inputs, label, sentence_a, sentence_b, tokenizer)
    data_cache_path = args.data_cache_path
    if not os.path.exists(data_cache_path):
        os.makedirs(data_cache_path)

    cache_pkl_path = os.path.join(data_cache_path, 'train.pkl')
    with open(cache_pkl_path, 'wb') as f:
        pickle.dump(inputs, f)

    return cache_pkl_path


def build_bert_inputs(inputs, label, sentence_a, sentence_b, tokenizer):
    inputs_dict = tokenizer.encode_plus(sentence_a, sentence_b, add_special_tokens=True,
                                        return_token_type_ids=True, return_attention_mask=True)
    inputs['input_ids'].append(inputs_dict['input_ids'])
    inputs['token_type_ids'].append(inputs_dict['token_type_ids'])
    inputs['attention_mask'].append(inputs_dict['attention_mask'])
    inputs['labels'].append(label)


class DGDataset(Dataset):
    def __init__(self, data_dict: dict, tokenizer: BertTokenizer):
        super(DGDataset, self).__init__()
        self.data_dict = data_dict
        self.tokenizer = tokenizer

    def __getitem__(self, index: int) -> tuple:
        data = (
            self.data_dict['input_ids'][index],
            self.data_dict['token_type_ids'][index],
            self.data_dict['attention_mask'][index],
            self.data_dict['labels'][index]
        )

        return data

    def __len__(self) -> int:
        return len(self.data_dict['input_ids'])


class Collator:
    def __init__(self, max_seq_len: int, tokenizer: BertTokenizer):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def pad_and_truncate(self, input_ids_list, token_type_ids_list,
                         attention_mask_list, labels_list, max_seq_len):
        input_ids = torch.zeros((len(input_ids_list), max_seq_len), dtype=torch.long)
        token_type_ids = torch.zeros_like(input_ids)
        attention_mask = torch.zeros_like(input_ids)
        for i in range(len(input_ids_list)):
            seq_len = len(input_ids_list[i])
            if seq_len <= max_seq_len:
                input_ids[i, :seq_len] = torch.tensor(input_ids_list[i], dtype=torch.long)
                token_type_ids[i, :seq_len] = torch.tensor(token_type_ids_list[i], dtype=torch.long)
                attention_mask[i, :seq_len] = torch.tensor(attention_mask_list[i], dtype=torch.long)
            else:
                input_ids[i] = torch.tensor(input_ids_list[i][:max_seq_len - 1] + [self.tokenizer.sep_token_id],
                                            dtype=torch.long)
                token_type_ids[i] = torch.tensor(token_type_ids_list[i][:max_seq_len], dtype=torch.long)
                attention_mask[i] = torch.tensor(attention_mask_list[i][:max_seq_len], dtype=torch.long)

        labels_list = [int(i) for i in labels_list]

        labels = torch.tensor(labels_list, dtype=torch.long)

        return input_ids, token_type_ids, attention_mask, labels

    def __call__(self, examples: list) -> dict:
        input_ids_list, token_type_ids_list, attention_mask_list, labels_list = list(zip(*examples))
        cur_max_seq_len = max(len(input_id) for input_id in input_ids_list)
        max_seq_len = min(cur_max_seq_len, self.max_seq_len)

        input_ids, token_type_ids, attention_mask, labels = \
            self.pad_and_truncate(input_ids_list, token_type_ids_list, attention_mask_list,
                                  labels_list, max_seq_len)

        data_dict = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

        return data_dict


def load_data(args, tokenizer):
    cache_pkl_path = os.path.join(args.data_cache_path, 'train.pkl')

    with open(cache_pkl_path, 'rb') as f:
        train_data = pickle.load(f)

    collate_fn = Collator(args.max_seq_len, tokenizer)
    train_dataset = DGDataset(train_data, tokenizer)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=collate_fn)
    return train_dataloader
