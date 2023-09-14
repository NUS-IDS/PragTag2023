# https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/flan-t5-samsum-summarization.ipynb
#####
from datasets import load_dataset
from random import randrange
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
import sys
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
nltk.download("punkt")

model_id="google/flan-t5-large"
max_source_length=512
max_target_length=5
##########

def preprocess_function(sample,padding="max_length"):
    # add prefix to the input for t5
    inputs = [prompt+ " "+ item for item in sample["source"]]

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["target"], max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs








prompt="Consider the definitions of labels: Recap: summarizes the manuscript, For e.g. \"The paper proposes a new method for...\"; \
        Strength: points out the merits of the work, For e.g. \"It is very well written and the contribution is significant.\"; \
        Weakness: points out a limitation, For e.g. \"However, the data is not publicly available, making the work hard to reproduce\"; \
        Todo: suggests the ways a manuscript can be improved, For e.g. \"Could the authors devise a standard procedure to obtain the data?\"; \
        Other: contains additional information such as reviewer's thoughts, background knowledge and performative statements, For e.g. \"Few examples from prior work: [1], [2], [3]\", \"Once this is clarified, the paper can be accepted.\"; \
        Structure: is used to organize the reviewing report, For e.g. \"Typos:\" \
        \
        Question: Which of the above labels most applies to the following sentence? Sentence: "



if __name__=="__main__":

    if len(sys.argv)!=3:
        print ("args1: data-dir (containing train.csv/dev.csv), args2: out-dir")
        sys.exit(1)

    data_path = sys.argv[1]
    out_path = sys.argv[2]

    print ("Reading files from "+data_path)
    ############## Load tokenizer and model 
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.add_tokens(['[SEP]'])
    dataset = load_dataset('csv', data_files={'train': data_path+'/train.csv', 'test': data_path+'/dev.csv'})
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    print(f"Train dataset size: {len(dataset['train'])}")
    print(f"Test dataset size: {len(dataset['test'])}")

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["source", "target"])


    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
    )



    # Define training args
    training_args = Seq2SeqTrainingArguments(
        output_dir="./scratch",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        fp16=False, # Overflows with fp16
        learning_rate=5e-5,
        num_train_epochs=3,
        # logging & evaluation strategies
        logging_dir=f"./scratch/logs",
        logging_strategy="steps",
        logging_steps=500,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=None, #compute_metrics,
    )

    # Start training
    trainer.train()

    model.save_pretrained(out_path+"/flan_model")
    tokenizer.save_pretrained(out_path+"/flan_model")
    print ("Model/Tokenizer written to "+out_path+"/flan_model")

