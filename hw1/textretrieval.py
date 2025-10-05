 # Place your imports here
import numpy as np
import nltk
import pandas as pd
from nltk.corpus import stopwords
import math

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
    ### Reads the test.csv file and iterates over every document content (entry in the column 2)
    ### removes leading and trailing spaces, transforms to lower case, remove punctuation, and removes stopwords
    ### Stores the formated information in the same "dataset" object

    dataset = pd.read_csv("https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",header=None)
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
      
    self.dataset = dataset #Set dataset as object attribute


  #### Bit Vector with Dot Product

  def build_vocabulary(self): #,collection):
    ### Return an array of 200 most frequent works in the collection
    ### dataset has to be read before calling the vocabulary construction

    #TODO: Create a vocabulary. Assume self.dataset has been preprocessed. Count the ocurrance of the words in the dataset. Select the 200 most common words as your vocabulary vocab. 
    freqMap = {}
    for line in self.dataset[2]:
      words = line.split()
      for i in words:
        freqMap[i] = freqMap.get(i, 0) + 1

    items_sorted = sorted(freqMap.items(), key=lambda item: item[1], reverse=True)
    vocab = []
    for i in range(200):
      vocab.append(items_sorted[i][0])

    self.vocab = np.array(vocab)

  def text2BitVector(self,text):
    ### return the bit vector representation of the text

    #TODO: Use self.vocab (assume self.vocab is created already) to transform the content of text into a bitVector
    #Use the order in the vocabulary to match the order in the bitVector
    bitVector = []
    words = text.split()
    for i in self.vocab:
      if i in words:
        bitVector.append(1)
      else:
        bitVector.append(0)
    bitVector = np.array(bitVector)
    return bitVector

  def bit_vector_score(self, query,doc):
    ### query and doc are the space-sparated list of words in the query and document
    q = self.text2BitVector(query)
    d = self.text2BitVector(doc)
    relevance = 0
    for i in range(len(q)):
      relevance += q[i] * d[i]
  
    #TODO: compute the relevance using q and d
    return relevance

  def adapt_vocab_query(self,query):
    ### Updates the vocabulary to add the words in the query

    #TODO: Use self.vocab and check whether the words in query are included in the vocabulary
    #If a word is not present, add it to the vocabulary (new size of vocabulary = original + #of words not in the vocabulary)
    #you can use a local variable vocab to work your changes and then update self.vocab
    vocab = self.vocab.tolist()

    query = query.split(" ")
    translator = str.maketrans("", "", self.punctuations)
    for i in range(len(query)):
      word = query[i].lower()
      word = word.translate(translator)
      if word not in vocab:
        vocab.append(word)
      
    self.vocab = np.array(vocab)

  def execute_search_BitVec(self,query):
    ### executes the computation of the relevance score for each document
    ### but first it verifies the query words are in the vocabulary
    #e.g.: query = "olympic gold athens"


    self.adapt_vocab_query(query) #Ensure query is part of the "common language" of documents and query

    relevances = np.zeros(self.dataset.shape[0]) #Initialize relevances of all documents to 0

    #TODO: Use self.vocab to compute the relevance/ranking score of each document in the dataset using bit_vector_score

    for idx, row in self.dataset.iterrows():
      relevances[idx] = self.bit_vector_score(query,row[2])


    
    return relevances # in the same order of the documents in the dataset

  #### TF-IDF with Dot Product

  def compute_IDF(self,M,collection):
    ### M number of documents in the collection; collection: documents (i.e., column 3 (index 2) in the dataset)

    #To solve this question you should use self.vocab

    self.IDF  = np.zeros(self.vocab.size)
    vocab = self.vocab.tolist()
    #TODO: for word in vocab: Compute the IDF frequency of each word in the vocabulary


    for i in range(len(vocab)):
      df = 0
      for text in collection:
        words = text.split()
        if vocab[i] in words:
          df += 1
      if df != 0:
        self.IDF[i] = math.log((M+1)/df)
      else:
        self.IDF[i] = 0






  def text2TFIDF(self,text, applyBM25_and_IDF=False):
    ### returns the bit vector representation of the text

    #TODO: Use self.vocab, self.K and self.IDF to compute the TF-IDF representation of the text
    tfidfVector = np.zeros(self.vocab.size)
    vocab = self.vocab.tolist()

    for word in vocab:
      if word in text.split():
        #TODO: Set the value of TF-IDF to be (temporarily) equal to the word count of word in the text
        ct = 0
        idx = vocab.index(word)
        for i in text.split():
          if i == word:
            ct += 1
        tfidfVector[idx] += ct


        if applyBM25_and_IDF:
            #TODO: update the value of the tfidfVector entry to be equal to BM-25 (of the word in the document) multiplied times the IDF of the word
            tfidfVector[idx] = ((self.IDF[idx] + 1) * tfidfVector[idx]) / (tfidfVector[idx] + self.IDF[idx]) * self.IDF[idx]
    return tfidfVector

  #grade (enter your code in this cell - DO NOT DELETE THIS LINE)
  def tfidf_score(self,query,doc, applyBM25_and_IDF=False):
    q = self.text2TFIDF(query)
    d = self.text2TFIDF(doc,applyBM25_and_IDF)
    relevance = 0
    for i in range(len(self.vocab)):
      relevance += q[i] * d[i] * self.IDF[i]
    #TODO: compute the relevance using q and d

    return relevance

  def execute_search_TF_IDF(self,query):
    #DIFF: Compute IDF
    self.adapt_vocab_query(query) #Ensure query is part of the "common language" of documents and query
    # global IDF
    self.compute_IDF(self.dataset.shape[0],self.dataset[2]) #IDF is needed for TF-IDF and can be precomputed for all words in the vocabulary and a given fixed collection (this excercise)

    #For this function, you can use self.IDF and self.dataset
    relevances = np.zeros(self.dataset.shape[0]) #Initialize relevances of all documents to 0

    #TODO: Use self.vocab to compute the relevance/ranking score of each document in the dataset using tfidf_score
    for idx, row in self.dataset.iterrows():
      doc = row[2]
      relevances[idx] = self.tfidf_score(query,doc,True)

    return relevances # in the same order of the documents in the dataset




if __name__ == '__main__':
    tr = TextRetrieval()
    tr.read_and_preprocess_Data_File() #builds the collection
    tr.build_vocabulary()#builds an initial vocabulary based on common words
    queries = ["olympic gold athens", "reuters stocks friday", "investment market prices"]
    print("#########\n")
    print("Results for BitVector")
    print(tr.vocab)
    for query in queries:
      print("QUERY:",query)
      relevance_docs = tr.execute_search_BitVec(query)
      #TODO: Once the relevances are computed, print the top 5 most relevant documents and the bottom 5 least relevant (for your reference) 
      print("Top 5: ", relevance_docs[:5])
      print("Bottom 5: ", relevance_docs[-5:])



      

    print("#########\n")
    print("Results for TF-IDF")
    for query in queries:
      print("QUERY:",query)
      relevance_docs = tr.execute_search_TF_IDF(query)
      #TODO: Once the relevances are computed, print the top 5 most relevant documents and the bottom 5 least relevant (for your reference) 
      print("Top 5: ", relevance_docs[:5])
      print("Bottom 5: ", relevance_docs[-5:])