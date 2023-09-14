#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 12:48:51 2023

@author: sdas
"""

from nltk.tokenize import sent_tokenize
import sys
import os
import json
import codecs
json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), 
                                   indent=2, ensure_ascii=False)


if __name__=="__main__":

    if len(sys.argv)!=3:
        print ("args1: AuxARR-data-dir, args2: out.json")
        sys.exit(1)


    sectypes={}

    inpd=sys.argv[1]
    outf=sys.argv[2]
    sdlist = os.listdir(inpd)

    newdata=[]
    for sdir in sdlist:
    
        path = inpd+"/"+sdir+"/v1/reviews.json"

        if os.path.exists(path):
            reviews = json_load(path)
        
            for review in reviews:
                rid = review["rid"]
                for key in review:
                    if key not in sectypes:
                        sectypes[key]=""
            
                if "report" in review:
                    for key in review["report"]:
                        if key not in sectypes:
                            sectypes[key]=""
                else:
                    continue
            
                sentences=[]
                labels=[]
                for key in review["report"]:
                    main_text = review["report"][key]
                    if "paper_summary" in key:
                        key="Recap"
                    elif "strengths" in key:
                        key="Strength"
                    elif "weaknesses" in key:
                        key="Weakness"
                    elif "typos" in key:
                        key="Structure_Todo_Other"
#                elif "ethical" in key:
#                    key="Other"
                    else:
                        key="Other"
                        
                    tsentences = sent_tokenize(main_text)
                
                    if len(tsentences)>0:
                    
                        for sent in tsentences:
                            sentences.append(sent)
                            labels.append(key)
                        
                temp={}
                temp["id"]=rid
                temp["pid"]=sdir
                temp["domain"]="arr_cs"
                temp["sentences"]=sentences
            
                temp["labels"]=labels
                newdata.append(temp)
                    
            if len(newdata)%25==0:
                print ("processed "+str(len(newdata)))
                  
    print (len(newdata))
    json_dump(newdata, outf)
    print (sectypes)
