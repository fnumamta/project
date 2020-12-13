import numpy as np
import pandas as pd
import os
import nltk
import re, string, unicodedata
import seaborn as sns
import sys
import os
import gensim
import sklearn
from bs4 import BeautifulSoup
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from gensim.models import word2vec, Word2Vec
from pandas import Series
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, BayesianRidge
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import ensemble


def predict(words):
    linearModel = joblib.load('linearReg.pkl')
    svmModel = joblib.load('svm.pkl')
    # model_random_forest_regressor = joblib.load('model_random_forest.pkl')
    # bayesModel = joblib.load('bayes.pkl')
    bayesModel = joblib.load('bayesRidge.pkl')
    data = {'comment': Series(words)}
    data = pd.DataFrame(data)
    print(data)
    data['comment'] = data['comment'].apply(removeBrackets)
    data['comment'] = data['comment'].apply(regEX)
    data['comment'] = data['comment'].apply(stemmer)
    data['comment'] = data['comment'].apply(removeStopwords)

    data['comment'] = data.comment.str.lower()
    data['sentences'] = data.comment.str.split('.')
    data['tokens'] = data['sentences']
    data['tokens'] = list(
        map(lambda sentences: list(map(nltk.word_tokenize, sentences)), data.sentences))
    data['tokens'] = list(
        map(lambda sentences: list(filter(lambda lst: lst, sentences)), data.tokens))
    print(data)
    # sentence = data['tokenized_sentences'][0]
    wordToVecmodel = Word2Vec.load("Word2Vec2")

    data['sentenceVectors'] = list(map(lambda sen_group:
                                        sentenceVectors(wordToVecmodel, sen_group),
                                        data.tokens))
    words = vectorToFeatures(data, 300)
    print(words)
    # lr_y_predict = model_lr.predict(words)
    predictlinearReg = linearModel.predict(words)
    # predictBayes = bayes.predict(words)
    predictBayesRidge = bayesModel.predict(words)
    predictSVMModel = svmModel.predict(words)

    return predictlinearReg, predictBayesRidge, predictSVMModel

def sentenceVectors(model, sentence):
    #Collecting all words in the text
#     print(sentence)
    vector = np.zeros(model.vector_size, dtype="float32")
    if sentence == [[]] or sentence == []  :
        return vector
    words=np.concatenate(sentence)
#     words = sentence
    #Collecting words that are known to the model
    modelVocKeys = set(model.wv.vocab.keys()) 
#     print(len(model_voc))

    # Use a counter variable for number of words in a text
    totalWords = 0
    # Sum up all words vectors that are know to the model
    for word in words:
        if word in modelVocKeys: 
            vector += model[word]
            totalWords += 1.

    # Now get the average
    if totalWords > 0:
        vector /= totalWords
    return vector


def vectorToFeatures(df, ndim):
    idx=[]
    for i in range(ndim):
        df[f'w2v_{i}'] = df['sentenceVectors'].apply(lambda x: x[i])
        idx.append(f'w2v_{i}')
    return df[idx]


#Removing the html strips
def removeHTML(text):
    text = BeautifulSoup(text, "html.parser")
    return text.get_text()

#Removing the square brackets
def removeBrackets(text):
    return re.sub('\[[^]]*\]', '', text)

#Removing the noisy text
def denoiseData(text):
    text = removeHTML(text)
    text = removeBrackets(text)
    return text
#Apply function on review column
#data['comment']=data['comment'].apply(removeBrackets)


# Apply function on review column
#Define function for removing special characters
def regEX(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text
#Apply function on review column
#data['comment']=data['comment'].apply(regEX)
#print(data.head())


# Apply function on review column
def stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text


# Apply function on review column


# Tokenization of text
toktokTokenizer = ToktokTokenizer()
# Setting English stopwords
#stopwordList = []
stopwordList = nltk.corpus.stopwords.words('english')
stop_words=set(stopwords.words('english'))
print(stop_words)


# removing the stopwords

def removeStopwords(text, is_lower_case=False):
    tokens = toktokTokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filteredTokens = [token for token in tokens if token not in stopwordList]
    else:
        filteredTokens = [token for token in tokens if token.lower() not in stopwordList]
    filteredTokens = ' '.join(filteredTokens).strip()
    return filteredTokens
# Apply function on review column



if __name__ == '__main__':
    print(predict(["This is a great game. I've even got a number of non game players enjoying it.  Fast to learn and always changing.","This is a great game. I've even got a number of non game players enjoying it.  Fast to learn and always changing."]))
    # text = "you are super good"
    # sentence = map(nltk.word_tokenize, text)
    # sentence = filter(lambda lst: lst, sentence)
    # print(sentence)
    # print(sklearn.__version__ )