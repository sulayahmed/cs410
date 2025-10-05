 # Place your imports here
import numpy as np
import nltk
import pandas as pd
from nltk.corpus import stopwords
import math
import gensim
from gensim.models import KeyedVectors
import numpy as np


class TextRetrieval():

  #For preprocessing
  punctuations = ""
  stop_words=set()

  #For VSM definition
  vocab = np.zeros(200)
  dataset = None
  K = 3 #

  def __init__(self):
    ##
    #TODO: obtain the file "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
    # and store it locally in a location accessible directly by this script (e.g. same directory don't use absolute paths)
    test = pd.read_csv("https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",header=None)
    ### TODO: Initialize punctuations (a string) and stop_words (a set)
    # Define self.punctuations to be any '"\,<>./?@#$%^&*_~/!()-[]{};:
    # Define self.stop_words from stopwords.words('english')
    self.punctuations = '\'"\\,<>./?@#$%^&*_~/!()-[]{};:'
    self.stop_words = stopwords.words('english')
    
  def read_and_preprocess_Data_File(self):
    dataset = pd.read_csv("https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",header=None)
    training = pd.read_csv("https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",header=None)
    punctuations = self.punctuations
    stop_words = self.stop_words

    dataset.head()
    for index, row in dataset.iterrows():
      line = row[2]
      #TODO: Implement removing stopwords and punctuation
      raw_words = line.split(" ")
      translator = str.maketrans("", "", punctuations)
      words = []
      for i in raw_words:
        i = i.translate(translator)
        i = i.lower()
        if i not in stop_words:
          words.append(i)
      dataset.loc[index, 2] = ' '.join(words)

    
    training.head()
    for index, row in training.iterrows():
      line = row[2]
      #TODO: Implement removing stopwords and punctuation
      raw_words = line.split(" ")
      translator = str.maketrans("", "", punctuations)
      words = []
      for i in raw_words:
        i = i.translate(translator)
        i = i.lower()
        if i not in stop_words:
          words.append(i)
      training.loc[index, 2] = ' '.join(words)
      
    self.training = training #Set dataset as object attribute
    self.dataset = dataset
    sentences = [text for text in self.training[2]]
    self.w2v = gensim.models.Word2Vec(
        sentences,
        vector_size=100,
        window=5,
        sg=1,            # 1 = skip-gram, 0 = CBOW
        min_count=2,
        workers=4
    )
    


  def w2v_score(self, query, doc):
    q_words = [w for w in query.lower().split() if w in self.w2v.wv]
    d_words = [w for w in doc.lower().split() if w in self.w2v.wv]

    if not q_words or not d_words:
        return 0.0
    score = 0.0
    count = 0
    for dw in d_words:
        log_probs = self.w2v.wv.log_probabilities(q_words)
        if dw in log_probs:
            score += log_probs[dw]
            count += 1
    return score / count if count > 0 else 0.0

  def execute_search_w2v(self, query):
    relevances = np.zeros(self.dataset.shape[0])
    for idx, row in self.dataset.iterrows():
        relevances[idx] = self.w2v_score(query, row[2])
    return relevances
        



if __name__ == '__main__':
    tr = TextRetrieval()
    tr.read_and_preprocess_Data_File() #builds the collection

    # tr.build_vocabulary()#builds an initial vocabulary based on common words
    queries = ["olympic gold athens", "reuters stocks friday", "investment market prices"]
    print("#########\n")
    print("Results for Word2Vec")
    for query in queries:
      print("QUERY:",query)
      relevance_docs = tr.execute_search_w2v(query)
      #TODO: Once the relevances are computed, print the top 5 most relevant documents and the bottom 5 least relevant (for your reference) 
      print("Top 5: ", relevance_docs[:5])
      print("Bottom 5: ", relevance_docs[-5:])






