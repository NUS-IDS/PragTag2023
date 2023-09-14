import sys
import json
import codecs

json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)



if __name__=="__main__":

    if len(sys.argv)!=3:

        print ("args1: data-dir containing data.json, data.json, flanT5_preds.json, T5_preds.json, RoBERTa_preds.json\nargs2: outfile.json")
        print ("This program takes in a directory containing the following input files in JSON format")
        print ("unlabeled-json-data=data.json, flanT5_preds.json, T5_preds.json, RoBERTa_preds.json")
        print ("Use the previously trained classifiersand predict code to obtain these prediction files")
        print ()
        sys.exit(1)

    inpdir=sys.argv[1]
    outfile=sys.argv[2]

    datafile=inpdir+"/data.json"
    inpf1=inpdir+"/FlanT5_preds.json"
    inpf2=inpdir+"/T5_preds.json"
    inpf3=inpdir+"/RoBERTa_preds.json"

    print ("Entering data from "+datafile+"\n"+inpf1+"\n"+inpf2+"\n"+inpf3)
    
    inpdata = json_load(datafile)

    fpreds = json_load(inpf1)
    t5preds = json_load(inpf2)
    robpreds = json_load(inpf3)

    print ("Size of data read:")
    print (len(inpdata))
    print (len(fpreds))
    print (len(robpreds))
    print (len(t5preds))
    

    newdata=[]

    count = 0
    mc = 0
    for dx, ele in enumerate(inpdata):

        
        rid = ele["id"]
        sents = ele["sentences"]
        pid = ele["pid"]
        domain = ele["domain"]

        newsents=[]
        newlabels=[]

        t5_ele = t5preds[dx]
        f_ele = fpreds[dx]
        r_ele = robpreds[dx]

        if r_ele["id"]==rid and rid==t5_ele["id"] and rid==f_ele["id"]:

            rlabels = r_ele["labels"]
            t5labels = t5_ele["labels"]
            flabels = f_ele["labels"]

            count += len(rlabels)

            for lx in range(len(sents)):

         #       if rlabels[lx].strip()==flabels[lx].strip() and rlabels[lx].strip()==t5labels[lx].strip():
         #           mc += 1    
         #           newsents.append(sents[lx])
         #           newlabels.append(rlabels[lx].strip())

                if (rlabels[lx].strip()==flabels[lx].strip()) and (rlabels[lx].strip()!=t5labels[lx].strip()):
                    mc += 1    
                    newsents.append(sents[lx])
                    newlabels.append(rlabels[lx].strip())


        else:
            print ("Review ID mismatch\t"+t5_ele["id"]+"\t"+rid)

        if len(newsents)>0:
            temp={}
            temp["id"]=rid
            temp["pid"]=pid
            temp["domain"]=domain
            temp["sentences"]=newsents
            temp["labels"]=newlabels
            newdata.append(temp)
            if len(newdata)%25==0:
                print ("Processed "+str(len(newdata)))




    print ("Count (overall)= "+str(count))
    print ("Match count majority= "+str(mc))
    print (len(fpreds))
    json_dump(newdata, outfile)
    print (len(inpdata))
    print (len(newdata))
