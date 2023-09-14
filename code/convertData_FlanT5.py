#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 15:12:40 2023

"""
import sys
import csv
import json
import codecs

json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)


if __name__=="__main__":
    if len(sys.argv)!=3:
        print ("\nA simple function to convert pragtag/json to FlanT5 files that we need")
        print ("args1: data.json, args2: out.csv\n")
        sys.exit(1)

    inpfile=sys.argv[1]
    outfile=sys.argv[2]

    trdata=json_load(inpfile)
    print (len(trdata))


    fout1 = open (outfile, "w")



    headerrow=["id","pid","domain","sentid", "source","target"]

    csvwriter = csv.writer(fout1, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)

    csvwriter.writerow(headerrow)
    for temp in trdata:

        for sx, sentence in enumerate(temp["sentences"]):
            row=[temp["id"], temp["pid"], temp["domain"], sx, sentence, \
             temp["labels"][sx]]
            csvwriter.writerow(row)

    fout1.close()
    print ("FlanT5 files written: "+outfile)
