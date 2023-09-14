#!/usr/bin/env python
# coding: utf-8

# In[41]:


import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import sys
import json
import codecs

json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)



prompt="Consider the definitions of labels: Recap: summarizes the manuscript, For e.g. \"The paper proposes a new method for...\"; \
        Strength: points out the merits of the work, For e.g. \"It is very well written and the contribution is significant.\"; \
        Weakness: points out a limitation, For e.g. \"However, the data is not publicly available, making the work hard to reproduce\"; \
        Todo: suggests the ways a manuscript can be improved, For e.g. \"Could the authors devise a standard procedure to obtain the data?\"; \
        Other: contains additional information such as reviewer's thoughts, background knowledge and performative statements, For e.g. \"Few examples from prior work: [1], [2], [3]\", \"Once this is clarified, the paper can be accepted.\"; \
        Structure: is used to organize the reviewing report, For e.g. \"Typos:\" \
        \
        Question: Which of the above labels most applies to the following sentence? Sentence: "


max_source_length=256
padding="max_length"

# # Global Settings

# In[42]:


labelmap={"Other":0, "Recap":1, "Strength":2, "Structure":3, "Todo":4, "Weakness":5}

if __name__=="__main__":

    if len(sys.argv)!=4:
        print ("args1: model/tokenizer-dir, args2: inp-data.json, args3: out-preds.json")
        sys.exit(1)


    modelpath=sys.argv[1]
    inpdataf=sys.argv[2]
    outfile=sys.argv[3]
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print ("model-path: "+modelpath)
    print ("reading inp data from: "+inpdataf)
    print ("specified out file: "+outfile)
    print ("using device: "+str(device))
    print ()

    model = AutoModelForSeq2SeqLM.from_pretrained(modelpath).to(device)
    tokenizer = AutoTokenizer.from_pretrained(modelpath)

    data = json_load(inpdataf)
    print ("loaded data instances sz="+str(len(data)))
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
    print ("Predictions written to "+outfile)
