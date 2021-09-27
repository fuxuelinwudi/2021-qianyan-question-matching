# coding:utf-8

import gc
import sys
import warnings

from sklearn.metrics import accuracy_score
from torch import multiprocessing
from argparse import ArgumentParser

sys.path.append('../../src')
from src.util.tools.finetune_tools import *

multiprocessing.set_sharing_strategy('file_system')


def train(args):
    tokenizer, model = build_model_and_tokenizer(args)

    if not os.path.exists(os.path.join(args.data_cache_path, 'train.pkl')):
        read_data(args, tokenizer)

    train_dataloader = load_data(args, tokenizer)

    total_steps = args.num_epochs * len(train_dataloader)

    optimizer, scheduler = build_optimizer(args, model, total_steps)

    total_loss, cur_avg_loss, global_steps = 0., 0., 0

    model_saved = False
    for epoch in range(1, args.num_epochs + 1):

        if model_saved:
            break

        train_iterator = tqdm(train_dataloader, desc='Training', total=len(train_dataloader))

        model.train()

        for batch in train_iterator:
            batch_cuda = batch2cuda(args, batch)
            loss, logits = model(**batch_cuda)[:2]

            total_loss += loss.item()
            cur_avg_loss += loss.item()

            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if (global_steps + 1) % args.logging_step == 0:
                epoch_avg_loss = cur_avg_loss / args.logging_step
                global_avg_loss = total_loss / (global_steps + 1)

                print(f"\n>> epoch - {epoch},  global steps - {global_steps + 1}, "
                      f"epoch avg loss - {epoch_avg_loss:.4f}, global avg loss - {global_avg_loss:.4f}.")

                cur_avg_loss = 0.0

            global_steps += 1

            if -1 < args.max_steps == global_steps:
                save_model(args, model, tokenizer)
                model_saved = True
                break

    if args.max_steps == -1:
        save_model(args, model, tokenizer)

    del model, tokenizer, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()


def main():
    parser = ArgumentParser()

    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--output_path', type=str,
                        default='../../user_data/output_model')
    parser.add_argument('--train_path', type=str,
                        default='../../user_data/process_data/train.txt')
    parser.add_argument('--data_cache_path', type=str,
                        default='../../user_data/process_data/pkl')
    parser.add_argument('--vocab_path', type=str,
                        default='../../user_data/pretrain_model/nezha-cn-base/vocab.txt')
    parser.add_argument('--model_path', type=str,
                        default='../../user_data/pretrain_model/nezha-cn-base')

    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_seq_len', type=int, default=128)

    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--eps', type=float, default=1e-8)

    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    parser.add_argument('--logging_step', type=int, default=1000)
    parser.add_argument('--max_steps', type=str, default=-1)

    parser.add_argument('--seed', type=int, default=1000)

    parser.add_argument('--device', type=str, default='cuda')

    warnings.filterwarnings('ignore')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    seed_everything(args.seed)
    train(args)


if __name__ == '__main__':
    main()
