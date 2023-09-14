#!/usr/bin/env python
# coding: utf-8

# In[41]:


import json
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from collections import defaultdict, OrderedDict
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, T5ForConditionalGeneration

# Utility functions
import util
import sys

# # Global Settings

# In[42]:


# Global variables
k_name = "unifiedQA"
k_banner = "Developed by Mingzhe Du"
k_tokenizer = "t5-large"
k_model = "t5-large"


k_seed = 42
k_max_src_len = 512
k_max_tgt_len = 5
k_batch_size = 4
k_num_train = -1
k_num_val = -1
k_num_workers = 4
k_epochs=3
k_lr = 1e-4
k_adam_eps = 1e-8
k_warmup_steps = 0
k_max_grad_norm =  1.0



# # Data Preprocessing

class T5DataSet(Dataset):
    def __init__(self, tokenizer, data_dir: str, type_path, max_examples=-1, max_src_len=256, max_tgt_len=256):
        """
        max_examples: if > 0 then will load only max_examples into the dataset; -1 means use all
        max_src_len and max_tgt_len refer to number of tokens in the input sequences. These are not randomized. If they were we might need to collate.
        """

        valid_type_paths = ["train", "dev"]
        assert type_path in valid_type_paths, f"Type path must be one of {valid_type_paths}"

        self.example_path = Path(data_dir) / type_path
        self.max_examples = max_examples
        self.tokenizer = tokenizer

        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        self.inputs = []            # List of dict
        self.targets = []           # List of dict
        self.input_text = []        # List of str
        self.target_text = []       # List of str

        self._build()               # Fill inputs, targets, max_lens
    
    def __len__(self):
        return len(self.inputs)
 

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask    = self.inputs[index]["attention_mask"].squeeze()  # Might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # Might need to squeeze

        src_text = self.input_text[index]
        tgt_text = self.target_text[index]

        # These will be cast to torch.long in forward
        return {"source_ids": source_ids, "source_mask": src_mask,
                "target_ids": target_ids, "target_mask": target_mask,
                "source_text": src_text, "target_text": tgt_text}
    
    def _build(self):
        source_path = self.example_path.with_suffix(".source")
        target_path = self.example_path.with_suffix(".target")

        with open(source_path, 'r') as f_source, open(target_path, 'r') as f_target:
            source, target = f_source.readlines(), f_target.readlines()
            source_ct, target_ct = len(source), len(target)
            assert source_ct == target_ct , f"Lengths don't match"


            inputs_out = []     # Accumulate the output of batch_encode
            targets_out = []    # Same
            inputs_text = []    # Save the original text for evaluations
            targets_text = []   # Aame

            if self.max_examples > 0 :
                source_ct = min(self.max_examples, source_ct)

            for idx in range(source_ct):
                src = source[idx].strip().lower()
                tgt = target[idx].strip()

                inputs_text.append(src)
                targets_text.append(tgt)

                # Tokenize
                # padding="max_length" pads to max_len, otherwise (e.g. for batch), we could use padding=longest with truncation.
                # self.tokenizer returns a dict of input_ids and attention_masks (where attn masks corresponds to padding)
                # NOTE: don't need add_special_tokens since EOS added automatically and others are PAD
                # NOTE: padding could also be done via collate in dataloader
                # TODO: we could actually batch encode these (i.e. multiple per)
                tokenized_inputs = self.tokenizer([src], max_length=self.max_src_len, padding="max_length", return_tensors="pt", truncation=True)
                tokenized_targets = self.tokenizer([tgt], max_length=self.max_tgt_len, padding="max_length", return_tensors="pt", truncation=True)

                inputs_out.append(tokenized_inputs)
                targets_out.append(tokenized_targets)
            
            self.inputs = inputs_out
            self.targets = targets_out
            self.input_text = inputs_text
            self.target_text = targets_text


# In[50]:


def get_dataloaders(tokenizer, batch_size, num_train, num_val, data_dir, num_workers, logger, shuffle_train=True, shuffle_dev=False):
    """
    Returns: Tuple[train_loader : DataLoader, dev_loader : DataLoader]
    # NOTE: we default to not shuffling the dev set
    """
    # todo: should pass max src and max tgt len in as arguments
    train_data_set = T5DataSet(tokenizer, type_path="train", data_dir=data_dir, max_examples=num_train, max_src_len=k_max_src_len, max_tgt_len=k_max_tgt_len)
    eval_data_set = T5DataSet(tokenizer, type_path="dev", data_dir=data_dir, max_examples=num_val, max_src_len=k_max_src_len, max_tgt_len=k_max_tgt_len)

    train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    eval_loader = DataLoader(eval_data_set, batch_size=2*batch_size, shuffle=shuffle_dev, num_workers=num_workers)

    logger.warning(f'Using max_src_len, max_tgt_len = ({k_max_src_len}, {k_max_tgt_len})')
    logger.info(f'Datasets loaded with sizes: train: {len(train_data_set)}, dev: {len(eval_data_set)}')

    return train_loader, eval_loader


# # Pipeline

def forward(model, device, batch):
    src_ids = batch["source_ids"].to(device, dtype=torch.long)
    src_mask = batch["source_mask"].to(device, dtype=torch.long)
    tgt_ids = batch["target_ids"].to(device, dtype=torch.long)

    # Pad ids (pad=0) are set to -100, which means ignore for loss calculation
    tgt_ids[tgt_ids[: ,:] == 0 ] = -100
    label_ids = tgt_ids.to(device)

    # NOTE: when we call model() with labels, they will be
    # - automatically right shifted by 1 (for teacher forcing)
    # - prepended by BOS=Beginning of sequence which is a PAD token
    # - any token that was -100 will be masked_fill_ to <pad> for teacher forcing return_dict means return as a dictionary
    
    out_dict = model(src_ids, attention_mask=src_mask, labels=label_ids, return_dict=True)
    loss, logits = out_dict['loss'], out_dict['logits']
    return loss, logits


def pipeline(tokenizer, model, k_data_dir, k_save_dir, logger):
    # Set random seeds
    util.set_seed(k_seed)


    # Load training dataset and validation dataset
    train_loader, dev_loader = get_dataloaders(tokenizer=tokenizer, batch_size=k_batch_size, num_train=k_num_train, num_val=k_num_val, data_dir=k_data_dir, num_workers=k_num_workers, logger=logger)

    # Reset in case we used the -1 flag for all
    num_train   = len(train_loader.dataset)
    num_val     = len(dev_loader.dataset)
    total_steps = (num_train // k_batch_size) * k_epochs
    total_train = num_train * k_epochs

    model.to(device)

    optimizer = AdamW(model.parameters(), lr=k_lr, eps=k_adam_eps)
    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=k_warmup_steps, num_training_steps=total_steps)

    logger.info(f'device: {device}\n'
                f'total_steps: {total_steps}\n'
                f'total_train (num_t * epoch): {total_train}\n')

    epoch = 0       # number of times we have passed through entire set of training examples
    step = 0        # number of total examples we have done (will be epoch * len(data_set) at end of each epoch)

    while epoch < k_epochs:
        epoch += 1

        ### Training
        model.train()
        logger.info(f'Training at step {step}...')

        with torch.enable_grad(), tqdm(total=num_train) as progress_bar:
            for batch_num, batch in enumerate(train_loader):
                real_batch_size = len(batch["source_ids"])

                loss, logits = forward(model, device, batch)
                loss_val = loss.mean().item()      # get the item since loss is a tensor

                # Backward
                optimizer.zero_grad()
                loss.mean().backward()
                nn.utils.clip_grad_norm_(model.parameters(), k_max_grad_norm)
                optimizer.step()
                scheduler.step()

                # Log info
                step += real_batch_size
                progress_bar.update(real_batch_size)
                progress_bar.set_postfix(epoch=epoch, loss=loss_val)


        ### Evaluation
        logger.info(f'Evaluating at step {step}...')

        # For parallel model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        
        model.eval()

        # See how the model is doing with exact match on tokens
        pred_list_all = []                  # Accumulate for saving; list; one list per epoch
        pred_list_correct = []
        loss_meter = util.AverageMeter()    # NLL (default metric for model) (reset each time)

        # Set up two count variables
        total_matches_no_eos_ct = 0
        total_matches_with_eos_ct = 0

        with torch.no_grad(), tqdm(total=num_val) as progress_bar:
            for batch_num, batch in enumerate(dev_loader):
                real_batch_size = len(batch["source_ids"])

                # Evaluation for loss fcn
                loss, logits = forward(model, device, batch)
                loss_meter.update(loss.mean().item(), real_batch_size)

                # Predict/Generate for token matches
                src_ids = batch["source_ids"].to(device, dtype=torch.long)
                src_mask = batch["source_mask"].to(device, dtype=torch.long)
                tgt_ids = batch["target_ids"].to(device, dtype=torch.long)

                # Tweak the generation params. See huggingface details for generate
                # Batch generate
                generated_ids = model.generate(src_ids, attention_mask=src_mask)       

                # Collect some stats
                total_matches_no_eos, total_matches_with_eos, correct_indices = util.masked_token_match(tgt_ids, generated_ids, return_indices=True)
                total_matches_no_eos_ct += total_matches_no_eos
                total_matches_with_eos_ct += total_matches_with_eos

                # Save for qualitative analysis
                orig_text_input, orig_text_output = batch["source_text"], batch["target_text"]
                # todo: this could break once skip_special_tokens is fixed
                outputs_decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
                preds = list(zip(orig_text_input, orig_text_output, outputs_decoded))
                pred_list_all.extend(preds)

                # We also store only the correct indices
                for idx in correct_indices.tolist():    # tensor to list; these are the valid indices
                    pred_list_correct.append(preds[idx[0]])     # each item was a list of one element

                # Log info
                progress_bar.update(real_batch_size)
                progress_bar.set_postfix(NLL=loss_meter.avg)

        # Save predictions for qualititative analysis
        fname="predictions."+str(epoch)+".csv"
        util.save_preds(pred_list_all, record_dir, file_name=fname)
        util.save_preds(pred_list_correct, record_dir, file_name="preds_correct.csv")
        results_list = [('NLL', loss_meter.avg),
                        ('exact_match_with_eos', total_matches_with_eos_ct),
                        ('exact_match_no_eos', total_matches_no_eos_ct)]
        results = OrderedDict(results_list)

        # Log to console
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
        logger.info(f'Dev {results_str}')


    return model


if __name__=="__main__":

    
    if len(sys.argv)!=3:
        print ("args1: data-dir (must have train&dev.source train&dev.target), args2: out-dir")
        sys.exit(1)

    k_data_dir = sys.argv[1]
    k_save_dir = sys.argv[2]
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Output directory 
    record_dir = util.get_save_dir(k_save_dir, k_name)
    # Logger
    logger = util.get_logger(record_dir, "root")

    # Load T5 model and T5 tokenizer (for quicker loading)
    tokenizer = AutoTokenizer.from_pretrained(k_tokenizer)
    #tokenizer.add_tokens(['[SEP]', '[sep]', 'prev_sent:', 'this_sent:', 'next_sent:' ])

    model = T5ForConditionalGeneration.from_pretrained(k_model)
    model.to(device)

    model = pipeline(tokenizer, model, k_data_dir, k_save_dir, logger)
    model.save_pretrained(record_dir+"/model")
    tokenizer.save_pretrained(record_dir+"/model")
    
    print ("Model/Tokenizer written to "+record_dir+"/model")

