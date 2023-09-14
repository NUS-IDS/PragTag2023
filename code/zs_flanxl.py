#!/usr/bin/env python
# coding: utf-8

# In[41]:

import sys
import torch
import json
import codecs
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration
from nltk.tokenize import sent_tokenize

# # Global Settings

# In[42]:

json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)

k_model = "google/flan-t5-xl"
k_tokenizer = "google/flan-t5-xl"
k_max_tgt_len = 16
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

tokenizer = AutoTokenizer.from_pretrained(k_tokenizer)
model = T5ForConditionalGeneration.from_pretrained(k_model) #, device_map="auto")
model.to(device)

def run_model(input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt").to(device) #"cuda")
    res = model.generate(input_ids, max_length=k_max_tgt_len, **generator_args)
    return tokenizer.batch_decode(res, skip_special_tokens=True)[0]


prompt="Consider the definitions of labels: Recap: summarizes the manuscript, For e.g. \"The paper proposes a new method for...\"; \
        Strength: points out the merits of the work, For e.g. \"It is very well written and the contribution is significant.\"; \
        Weakness: points out a limitation, For e.g. \"However, the data is not publicly available, making the work hard to reproduce\"; \
        Todo: suggests the ways a manuscript can be improved, For e.g. \"Could the authors devise a standard procedure to obtain the data?\"; \
        Other: contains additional information such as reviewer's thoughts, background knowledge and performative statements, For e.g. \"Few examples from prior work: [1], [2], [3]\", \"Once this is clarified, the paper can be accepted.\"; \
        Structure: is used to organize the reviewing report, For e.g. \"Typos:\" \
        \
        Question: Which of the above labels most applies to the following sentence? Sentence: "

        
    
    
labelmap={"Other":0, "Recap":1, "Strength":2, "Structure":3, "Todo":4, "Weakness":5}

if __name__=="__main__":

    if len(sys.argv)!=3:
        print ("args1: data.json, args2: out-preds.json")
        sys.exit(1)

    inpf=sys.argv[1]
    outfile=sys.argv[2]
    data = json_load(inpf)
    print ("loaded data instances sz="+str(len(data)))
    print ("Using "+str(device))
    fout = open (outfile, "w")
    newdata=[]
    for ex, ele in enumerate(data):
        sents=ele["sentences"]
        labels=[]
        for sx, sent in enumerate(sents):

            input_str = prompt+" "+sent
            input_ids = tokenizer.encode(input_str, return_tensors="pt").to(device)
            res = model.generate(input_ids) #, max_length=k_max_tgt_len, **generator_args)
            op = tokenizer.batch_decode(res, skip_special_tokens=True)[0]
            if op not in labelmap:
                print ("Ignoring pair, "+op+")")
                op="Other"
            
            labels.append(op)

        ele["labels"]=labels
        newdata.append(ele)
        if len(newdata)%25==0:
            print ("Processed "+str(len(newdata)))


    json_dump(newdata, outfile)
    print ("predictions written to "+outfile)

    
    


