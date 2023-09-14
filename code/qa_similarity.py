from sentence_transformers import SentenceTransformer, util

from nltk.tokenize import sent_tokenize
import torch
import json
import codecs
import sys

json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#Load the model
model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1', device=device)

def scoreDocs(query, docs):
    
    #Encode query and documents
    query_emb = model.encode(query)
    doc_emb = model.encode(docs)

    #Compute dot score between query and all document embeddings
    #scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
    scores = util.cos_sim(query_emb, doc_emb)[0].cpu().tolist()

    #Combine docs & scores
    doc_score_pairs = list(zip(docs, scores))

    #Sort by decreasing score
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

    return doc_score_pairs

    #Output passages & scores
    #for doc, score in doc_score_pairs:
    #print(score, doc)


bathe_questions={
        "Recap": ["What is the main contribution?",
        "What is the main finding?",
        "What is described in the paper?",
        "what new method is proposed?",
        "what did the authors show in this paper?",
         "what problem was addressed in this paper?",
        "what is a brief summary of the paper?",
        "what work is described in this paper?",
        "what contributions does the paper make?",
        "what solution is presented in this paper?",],

        "Strength": [ "what is the positive aspects in this paper?",
        "what is the valuable contribution made in this paper?",
        "what is the strength of this contribution?",
        "how original are the results described in the paper?",
        "what is the good qualities of this paper?",
        "what is exciting about this paper?",
        "what is inspiring about this paper?",],
        
        "Weakness":["what is the main drawback of this paper?",
        "what is the main risk of this paper?",
        "what is the key weakness of this paper?",
        "what are the key concerns stated?",
        "what is missing in this paper?",
        "what is lacking in the paper?",],
        
        "Todo":["How can this paper be improved?",
        "what experiments can be added in the paper?",
        "what else needs to be done in the paper?",
        "How can these results be made stronger?",],
        
        "Structure": ["What describes the organization of the review?",
        "What typo or grammar is mentioned?",
        "what stylistic issues can be addressed?",
        "what is the heading of this review section?",],
        
        "Other":["what thoughts are mentioned?",
        "What background knowledge is mentioned?",
        "What statement describes performance?",
        "What other references or citations are provided?"]

}

threshold=0.4

if __name__=="__main__":

    if len(sys.argv)!=3:
        print ("args1: data.json, args2: out-preds.json")
        sys.exit(1)

    inpf=sys.argv[1]
    outfile=sys.argv[2]
    data = json_load(inpf)

    newdata=[]
    data = json_load(inpf)

    for ele in (data):
        temp={}
        temp["id"] = ele["id"]
  #      if ele["id"]!="report22299":
  #          continue

        sents=ele["sentences"]
       # temp["sentences"]=sents
        doc2maxscore={}
        doc2maxtype={}
#        print ("DEBUG "+str(len(sents)))
        question2qtype={}
        for qtype in bathe_questions:
            questions = bathe_questions[qtype]
            
            for question in questions:
                doc_score_pairs = scoreDocs(question, sents)
 #           print (question)
 #           print (len(doc_score_pairs))
                for doc, score in doc_score_pairs:
                    if doc not in doc2maxscore:
                        doc2maxscore[doc]=score
                        doc2maxtype[doc]=(qtype, question)
                    else:
                        if doc2maxscore[doc]<score:
                            doc2maxscore[doc]=score
                            doc2maxtype[doc]=(qtype, question)


        for doc in doc2maxscore:
            (qtype, question) = doc2maxtype[doc]
            if doc2maxscore[doc]<threshold:
                doc2maxtype[doc]=("Other", "OtherQ")
        

        labels=[]
        for doc in sents:
            (qtype, _) = doc2maxtype[doc]
            labels.append(qtype)

        temp["labels"]=labels
        temp["pid"] = ele["pid"]
        temp["domain"] = ele["domain"]

        if len(ele["sentences"])!=len(labels):
            print ()
            print ("number of labels and sentences mismatch for "+ele["id"])
            print (len(ele["sentences"]))
            print (len(labels))

        newdata.append(temp)
        if len(newdata)%25==0:
            print (temp["labels"])
    
    print ("#preds="+str(len(newdata))+" "+str(len(data)))
    json_dump(newdata, outfile)
