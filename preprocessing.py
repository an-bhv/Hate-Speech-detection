import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import json
import copy
import re
import nltk
import pandas as pd
import io
import re
import string
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split

cols = ['id','text','sentiment']

import io
#train = pd.read_csv(io.BytesIO(uploaded['agr_en_train.csv']),header=None, names=cols)

train = pd.read_csv("agr_hi_train.csv",header=None, names=cols)
# train = train.append(pd.read_csv("agr_hi_dev.csv",header=None, names=cols))
# train = train.append(pd.read_csv("agr_hi_fb_gold.csv",header=None, names=cols))
# train = train.append(pd.read_csv("agr_hi_tw_gold.csv",header=None, names=cols))
train.drop(['id'],axis=1,inplace=True)
train['sentiment'] = train['sentiment'].map({'OAG': int(1), 'CAG': int(0),'NAG': int(0)})

train = train.dropna()
train

from unidecode import unidecode

cnt =0
l = []
for i in range(len(train['text'])):
    try:
        train['text'][i] = unidecode(train['text'][i])
    except:
        cnt += 1
        l.append(i)
print(cnt)
train.drop(train.index[l],inplace=True)

train

new_text = []
corpus = train['text']
print(type(corpus))
for i in corpus:
    new_text.append(i.split(" "))
new_text
for i in range(len(new_text)):
    for j in range(len(new_text[i])):
        new_text[i][j] = new_text[i][j].replace("\n","")

for i in range(len(new_text)):
    #print(i,len(i),i[0][0])
    while "" in new_text[i]:
        new_text[i].remove("")
    while '' in new_text[i]:
        new_text[i].remove('')
    while '""' in new_text[i]:
        new_text[i].remove('""')
    while "''" in new_text[i]:
        new_text[i].remove("''")
    
    # for j in range(len(new_text[i])):
    #     if new_text[i][j][-1] == ".":
    #         new_text[i][j] = new_text[i][j][:-1]
    
    for j in range(len(new_text[i])):
        new_text[i][j] = re.sub(r"([.]){2,}", r"",new_text[i][j])
        
    for j in range(0,len(new_text[i])):
        if "#" in new_text[i][j]:
            new_text[i][j] = new_text[i][j][1:]
            
    for j in range(0,len(new_text[i])):
        #print(i[j],sep = "")
        #print(i[j])
        for k in range(len(new_text[i][j])-1):
            
            if k<len(new_text[i][j]):
                if (new_text[i][j][k]=='@'):
                    #print(i)
                    #print(i[j][k:])
                    el = new_text[i][j][k:].find(" ")
                    if el==-1:
                        new_text[i][j] = new_text[i][j][:k]
                        #print(i[j])
                    else:
                        new_text[i][j] = new_text[i][j][:k] + new_text[i][j][k+el:]
        #print(i[j])
    
    for j in range(0,len(new_text[i])):
        if j<len(new_text[i]):
            if "http://"in new_text[i][j] or "https://"in new_text[i][j] or ".com/" in new_text[i][j]:
                #print(i[j])
                new_text[i].pop(j)

def re_sub(pattern, repl):
        return re.sub(pattern, repl, text)
    
for i in range(len(new_text)):
    for j in range(len(new_text[i])):
        text = new_text[i][j]
        text = re_sub(r"([pls?s]){2,}", r"\1")
        text = re_sub(r"([plz?z]){2,}", r"\1")
        text = re_sub(r'\\n', r' ')
        text = re_sub(r" sx "," sex ")
        text = re_sub(r" u "," you ")
        text = re_sub(r" u r "," you are ")
        text = re_sub(r" r "," are ")
        text = re_sub(r" y "," why ")
        text = re_sub(r" Y "," why ")
        text = re_sub(r"Y "," why ")
        text = re_sub(r"RT ","")
        text = re_sub(r" hv "," have ")
        text = re_sub(r" c "," see ")
        text = re_sub(r" bcz "," because ")
        text = re_sub(r" coz "," because ")
        text = re_sub(r" v "," we ")
        text = re_sub(r" ppl "," people ") 
        text = re_sub(r" pepl "," people ")
        text = re_sub(r" r b i "," rbi ")
        text = re_sub(r" R B I "," rbi ")
        text = re_sub(r" R b i "," rbi ")
        text = re_sub(r" R "," are ")
        text = re_sub(r" hav "," have ")
        text = re_sub(r"R "," are ")
        text = re_sub(r" U "," you ")
        text = re_sub(r" pls "," please ")
        text = re_sub(r"Pls ","Please ")
        text = re_sub(r"plz ","please ")
        text = re_sub(r"Plz ","Please ")        
        text = re_sub(r"PLZ ","Please ")
        text = re_sub(r"Pls","Please ")
        text = re_sub(r"plz","please ")
        text = re_sub(r"Plz","Please ")
        text = re_sub(r"PLZ","Please ")
        text = re_sub(r"&","and")
        text = re_sub(r"\?"," question ")
        text = re_sub(r"\!"," exclamation ")
        text = re_sub(r"/"," ")
        #text = re_sub()
        new_text[i][j] = text

for i in range(len(new_text)):
    for j in range(len(new_text[i])):
        new_text[i][j] = new_text[i][j].replace('.',' ')
        new_text[i][j] = new_text[i][j].replace('"',' ')
        new_text[i][j] = new_text[i][j].replace("'"," ")
        new_text[i][j] = new_text[i][j].replace('#',' ')
        new_text[i][j]=new_text[i][j].replace("-"," ")
        new_text[i][j]=new_text[i][j].replace(":"," ")
        new_text[i][j]=new_text[i][j].replace(";"," ")
        new_text[i][j]=new_text[i][j].replace("(","")
        new_text[i][j]=new_text[i][j].replace(")","")
        new_text[i][j]=new_text[i][j].replace("[","")
        new_text[i][j]=new_text[i][j].replace("]","")
        if (new_text[i][j][0:].isalnum()==False):
            new_text[i][j]=new_text[i][j][0:]
        if (new_text[i][j][:-1].isalnum()==False):
            new_text[i][j]=new_text[i][j][:-1]
        new_text[i][j]=new_text[i][j].replace("_"," ")
        new_text[i][j]=new_text[i][j].replace("+"," ")
        new_text[i][j]=new_text[i][j].replace("-"," ")
        new_text[i][j]=new_text[i][j].replace(","," ")

        new_text[i][j] = " ".join(new_text[i][j].split())
        


print(new_text)


newest_text=[]
for i in new_text:
    stri = " ".join(i),"\n"
    newest_text.append(stri[0].split(" "))


for i in newest_text:
    print(" ".join(i),"\n")

print(len(newest_text))

for i in range(len(newest_text)):
    for j in range(len(newest_text[i])):
        newest_text[i][j]=newest_text[i][j].lower()

def RepetitionStemmer(word):
    i=0
    newWord = ''
    while(i <len(word)):
        c = word[i]
        newWord+=c
        while(i<len(word) and word[i] == c):
            i=i+1

    return newWord
    
def stem2dListOfWords(listOfWords2d):
    output = []
    for sentenceOfWords in listOfWords2d:
        output.append([RepetitionStemmer(word) for word in sentenceOfWords])
    return output

def RegexStemmer(word):
    re.sub(r'(.{2,}?)([aeiougyn]+$)',r'\1', word)
    return word

def stemListOfWords(listOfWords):
    return [RepetitionStemmer(word) for word in listOfWords]

stem1_text=[]
stem2_text=[]
wordj=[]

stem2_text = stem2dListOfWords(newest_text)

print(len(stem2_text))
for i in stem2_text:
    print(" ".join(i),"\n")
    continue

stop_list=[]
stopwords = open("stopwords.txt",'r')
stop_list.append([line.strip() for line in stopwords.readlines()])
stop_list=stop_list[0]
stop_list2=stemListOfWords(stop_list)
for i in range(97,123):
    stop_list.append(chr(i))
stop_list.append("ji")
stop_list.append("")
stop_list.append(" ")
print(stop_list)
stop_list=stop_list+stop_list2
cnt =0
newest_text = stem2_text

for i in range(len(newest_text)):
    for j in newest_text[i].copy():
        if j in stop_list:
            print(j)
            newest_text[i].remove(j)
            cnt+=1

result = train['sentiment']
print(result[7])
allres = []
for i in result:
    allres.append(i)
print(allres)
print(allres[6])
#file1 = open("nonaug/test.txt",'w')
file1 = open("preprocessed.txt",'w')
for i in range(len(newest_text)):
    if i<20:
        print(allres[i])
        print(newest_text[i])
    line = " ".join(newest_text[i])
    if i<20:
        print(i,line,allres[i])
    towr = line+","+str(int(allres[i]))+"\n"
    if i<20:
        print(i,'towr->',towr)
    file1.write(towr)
