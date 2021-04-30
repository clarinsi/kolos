from joblib import dump, load
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,f1_score,roc_auc_score
import sys
import pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--eval', action='store_true')
parser.add_argument('--output', action='store_true')
args = parser.parse_args()

def combine(emb1,emb2):
    return np.concatenate((emb1,emb2,emb1+emb2))

kas_gramrels={}
kas_gramrels_bidirectional={}
kas_gramrels_dup={}
kas_gramrels_bidirectional_dup={}
lexemes={}
counter=0
labels=[]
kas_headwords=set()
kas_headwords_bidirectional=set()
for line in open('kas.csv'):
    counter+=1
    line=line.strip().split('|')
    gramrel_tokens=line[3].split()
    gramrel_first=gramrel_tokens[0].split('-')[-1]
    gramrel_last=gramrel_tokens[-1].split('-')[-1]
    if gramrel_first.islower() and not gramrel_last.islower():
        head=(line[4].split('_')[0],gramrel_first[0].lower())
        dep=(line[2].split('_')[0],gramrel_last[0].lower())
    elif gramrel_last.islower() and not gramrel_first.islower():
        head=(line[2].split('_')[0],gramrel_first[0].lower())
        dep=(line[4].split('_')[0],gramrel_last[0].lower())
    else:
        #print gramrel_tokens
        continue
    kas_headwords.add(head)
    kas_headwords_bidirectional.add(head)
    kas_headwords_bidirectional.add(dep)
    lexemes[head]=None
    lexemes[dep]=None
    gramrel=line[3].lower()
    if gramrel not in kas_gramrels:
        kas_gramrels[gramrel]={}
        kas_gramrels_dup[gramrel]={}
        kas_gramrels_bidirectional[gramrel]={}
        kas_gramrels_bidirectional[gramrel.upper()]={}
        kas_gramrels_bidirectional_dup[gramrel.upper()]={}
    if head not in kas_gramrels[gramrel]:
        kas_gramrels[gramrel][head]=[]
        kas_gramrels_dup[gramrel][head]=set()
        kas_gramrels_bidirectional[gramrel][head]=[]
    if dep not in kas_gramrels_bidirectional[gramrel.upper()]:
        kas_gramrels_bidirectional[gramrel.upper()][dep]=[]
        kas_gramrels_bidirectional_dup[gramrel.upper()][dep]=set()
    if dep not in kas_gramrels_dup[gramrel][head]:
        kas_gramrels[gramrel][head].append((dep,float(line[0]),float(line[1]),{'DA':1,'NE':0}[line[5]]))
        kas_gramrels_bidirectional[gramrel][head].append((dep,float(line[0]),float(line[1]),{'DA':1,'NE':0}[line[5]]))
        labels.append(line[5])
        kas_gramrels_dup[gramrel][head].add(dep)
    if head not in kas_gramrels_bidirectional_dup[gramrel.upper()][dep]:
        kas_gramrels_bidirectional[gramrel.upper()][dep].append((head,float(line[0]),float(line[1]),{'DA':1,'NE':0}[line[5]]))
        kas_gramrels_bidirectional_dup[gramrel.upper()][dep].add(head)

print('Loaded KAS data...')

kas_embeddings=pickle.load(open('kas_embeddings.pkl','rb'),encoding='bytes')
print('Loaded KAS embeddings')

for gramrel in ('pbz0 sbz0','PBZ0 SBZ0','sbz0 sbz2','SBZ0 SBZ2','gbz sbz4','GBZ SBZ4','rbz pbz0','RBZ PBZ0'):
    if gramrel[0].isupper():
        gramrel_out=gramrel.lower().replace(' ','_')+'_rev.joblib'
    else:
        gramrel_out=gramrel.lower().replace(' ','_')+'.joblib'
    model=load('models/'+gramrel_out)
    results=[]
    output=[]
    for head in kas_gramrels_bidirectional[gramrel]:
        X_train=[]
        X_candidate=[]
        y_train=[]
        if len(kas_gramrels_bidirectional[gramrel][head])>=10 and len(set([e[-1] for e in kas_gramrels_bidirectional[gramrel][head]]))>1:
            for dep in kas_gramrels_bidirectional[gramrel][head]:
                X_train.append(combine(kas_embeddings[head],kas_embeddings[dep[0]]))
                y_train.append(dep[3])
                X_candidate.append((head,dep[0]))
        if len(X_train)>0:
            y_pred=[e[1] for e in model.predict_proba(X_train)]
            #print(y_pred)
            results.extend([roc_auc_score(y_train,y_pred)]*len(y_train))
            if args.output:
                print('###',gramrel,head,'###')
                for c,p,t in sorted(zip(X_candidate,y_pred,y_train),key=lambda x:-x[1]):
                    print(c,p,t)
    if args.eval:
        print(gramrel,np.mean(results))
