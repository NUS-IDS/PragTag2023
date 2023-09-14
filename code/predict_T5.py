#!/usr/bin/env python
# coding: utf-8

# In[41]:


import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, T5ForConditionalGeneration
from nltk.tokenize import sent_tokenize

# # Global Settings

# In[42]:
import sys
import json
import codecs

json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)





def run_model(tokenizer, model, device, input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt").to(device)
    res = model.generate(input_ids, max_length=k_max_tgt_len, **generator_args)
    return tokenizer.batch_decode(res, skip_special_tokens=True)[0]



labelmap={"Other":0, "Recap":1, "Strength":2, "Structure":3, "Todo":4, "Weakness":5}

if __name__=="__main__":
 
    if len(sys.argv)!=4:
        print ("args1: model/tok path, args2: inpdata.json, args3: outpreds.json")
        sys.exit(1)

    modelpath = sys.argv[1]

    inpf=sys.argv[2] #"/home/idssdg/expts_pragtag_cc/processed_cc/test_labels.json"
    outf=sys.argv[3] #"/home/idssdg/expts_pragtag_cc/fs/10pct/preds/test_combdata_ss_t5.json"

    k_max_tgt_len = 10
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    tokenizer = AutoTokenizer.from_pretrained(modelpath)
    model = T5ForConditionalGeneration.from_pretrained(modelpath)
    model.to(device)

    data = json_load(inpf)
    print (len(data))
    newdata=[]
    for ex, ele in enumerate(data):
        sents = ele["sentences"]
        labels=[]
        for sent in sents:
            op = run_model(tokenizer, model, device, sent).strip()
            if op not in labelmap:
                print ("Ignoring op="+op)
                op="Other"
            labels.append(op)

        ele["labels"]=labels
        if ex%25==0:
            print (ele["labels"])
        newdata.append(ele)

    json_dump(newdata, outf)
    print ("output written to "+outf)



        
    
    
