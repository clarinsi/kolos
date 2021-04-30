from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
import random
import numpy as np
from joblib import dump, load

def combine(emb1,emb2):
    return np.concatenate((emb1,emb2,emb1+emb2))

def head_out(head):
    return head[0]+'#'+{'p':'A','s':'N','r':'R','g':'V'}[head[1]]

embeddings = pickle.load(open('kolos_embeddings.pkl', 'rb'), encoding='bytes')
print('Loaded embeddings...',len(embeddings))

gramrels={}
gramrels_bidirectional={}
labels=[]
headwords=set()
headwords_bidirectional=set()

for line in open('kolos.csv'):
    line=line.split('|')
    gramrel_tokens=line[5].split()
    gramrel_first=gramrel_tokens[0].split('-')[-1]
    gramrel_last=gramrel_tokens[-1].split('-')[-1]
    if gramrel_first.islower() and not gramrel_last.islower():
        head=(line[8].split('_')[0],gramrel_first[0].lower())
        dep=(line[3].split('_')[0],gramrel_last[0].lower())
    elif gramrel_last.islower() and not gramrel_first.islower():
        head=(line[3].split('_')[0],gramrel_first[0].lower())
        dep=(line[8].split('_')[0],gramrel_last[0].lower())
    gramrel=line[5].lower()
    if gramrel not in gramrels:
        gramrels[gramrel]={}
        gramrels_bidirectional[gramrel]={}
        gramrels_bidirectional[gramrel.upper()]={}
    if head not in gramrels[gramrel]:
        gramrels[gramrel][head]=[]
        gramrels_bidirectional[gramrel][head]=[]
    if dep not in gramrels_bidirectional[gramrel.upper()]:
        gramrels_bidirectional[gramrel.upper()][dep]=[]
    gramrels[gramrel][head].append((dep,float(line[0]),float(line[1]),{'DA':1,'NE':0}[line[11]]))
    gramrels_bidirectional[gramrel][head].append((dep,float(line[0]),float(line[1]),{'DA':1,'NE':0}[line[11]]))
    gramrels_bidirectional[gramrel.upper()][dep].append((head,float(line[0]),float(line[1]),{'DA':1,'NE':0}[line[11]]))
    labels.append(line[11])
    headwords.add(head)
    headwords_bidirectional.add(head)
    headwords_bidirectional.add(dep)
print('Loaded gramrels...',len(gramrels))

models={}
X_train_all=[]
y_train_all=[]
train_instances=set()
for gramrel in ['PBZ0 SBZ0','RBZ PBZ0','rbz pbz0','sbz0 sbz2','GBZ SBZ4','gbz sbz4','SBZ0 SBZ2','pbz0 sbz0']:
    X_train=[]
    y_train=[]
    if gramrel[0].isupper():
            gramrel_out=gramrel.lower().replace(' ','_')+'_rev'
    else:
        gramrel_out=gramrel.lower().replace(' ','_')
    for head in gramrels_bidirectional[gramrel]:
        if len(gramrels_bidirectional[gramrel][head])>10 and len(set([e[-1] for e in gramrels_bidirectional[gramrel][head]]))>1:
            train_instances.add((head_out(head),gramrel_out))
            for dep in gramrels_bidirectional[gramrel][head]:
                X_train.append(combine(embeddings[head],embeddings[dep[0]]))
                y_train.append(dep[3])
                X_train_all.append(combine(embeddings[head],embeddings[dep[0]]))
                y_train_all.append(dep[3])
    if len(X_train)>0:
        print(gramrel)
        clf=Pipeline([('scl',StandardScaler()),('clf',SVC(probability=True))])
        clf.fit(X_train,y_train)
        models[gramrel]=clf
        dump(clf,'models/'+gramrel_out+'.joblib')
print('Trained models...',len(models))

