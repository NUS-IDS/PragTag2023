
from sklearn.metrics import classification_report
import json
import codecs
import sys

json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)
labelmap={"Other":0, "Recap":1, "Strength":2, "Structure":3, "Todo":4, "Weakness":5}

if __name__=="__main__":
    
    if len(sys.argv)!=3:
        print ("args1: gold.json, args2: preds.json")
        sys.exit(1)

    goldf=sys.argv[1]
    predf=sys.argv[2]
    gold = json_load(goldf)
    preds = json_load(predf)

    y_gold=[]
    y_pred=[]

    for ex, ele in enumerate(gold):

        g = ele["labels"]
        p = preds[ex]["labels"]


        for lx, label in enumerate(g):
            y_gold.append(labelmap[label])
            y_pred.append(labelmap[p[lx]])


    print (len(y_gold))
    print (len(y_pred))
    print(classification_report(y_gold, y_pred, target_names=["Other", "Recap", "Strength", "Structure", "Todo", "Weakness"]))
    
    



        
    
    
