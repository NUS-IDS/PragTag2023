#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 15:12:40 2023

@author: sdas
"""
import sys
import csv
import json
import codecs

json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)


if __name__=="__main__":


    if len(sys.argv)!=3:
        print ("\nA simple function to convert pragtag/json to T5 files that we need\n")
        print ("args1: data.json, args2: outfile-pfx (for creating .source and .target files)")
        sys.exit(1)

    inpfile=sys.argv[1]
    outpfx=sys.argv[2]

    trdata=json_load(inpfile)
    print (len(trdata))


    fout1 = open (outpfx+".source", "w")
    fout2 = open (outpfx+".target", "w")

    for temp in trdata:    
    
        for sx, sentence in enumerate(temp["sentences"]):
            top = sentence.replace("\t"," ")
            fout1.write(top.strip()+"\n")
            fout2.write(temp["labels"][sx]+"\n")
    
        fout1.flush()
        fout2.flush()


    fout1.close()
    fout2.close()
    print ("T5 files written: "+outpfx+".source and "+outpfx+".target")
