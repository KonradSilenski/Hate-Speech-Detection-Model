import pandas as pd
import numpy as np
import gensim
from gsdmm import MovieGroupProcess
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
import nltk
import pickle
pd.options.mode.chained_assignment = None

class modelTrain:
    def __init__(self, data):
        self.tweets = None
        print(data)
        if data.endswith('.csv'):
            self.tweets = pd.read_csv(data)
        elif data.endswith('.parquet'):
            self.tweets = pd.read_parquet(data)

        self.stop_words = stopwords.words('english')
        self.stop_words.extend(
            ['url', 'need', 'want', 'the', 'u', 'etc', 'sorry', 'help', 'cute', 'one', 'please', 'like', 'many', 'love',
             'get', 'say', 'think', 'good', 'think',
             'love', 'would', 'saying', 'much', 'see', 'great', 'could', 'us', 'time', 'makes', 'world', 'let', 'nice',
             'full', 'really', 'well', 'wish', 'full',
             'al', 'igbo', 'abeg', 'miraj', 'lailat', 'make', 'new', 'nawaz', 'nugs', 'mubarak', 'notwithstanding',
             'sucking', 'yt', 'ya', 'ur', 'show', 'add',
             'sh', 'tr', 'losangeles', 'children', 'hope', 'even', 'work', 'know', 'eat', 'look', 'wa', 'ha', 'thanks', 'thank'])

        self.lemmatizer = WordNetLemmatizer()

    def sent_to_words(self, sentences):
        for sentence in sentences:
            yield ([self.lemmatizer.lemmatize(word) for word in gensim.utils.simple_preprocess(str(sentence), deacc=True)])

    def remove_stopwords(self, texts):
        return [[word for word in simple_preprocess(str(doc))
                 if word not in self.stop_words] for doc in texts]

    def preProcessData(self, tweets):
        # Remove the columns
        self.tweets = tweets[['text']]

        # Remove punctuation
        self.tweets['text_processed'] = \
            self.tweets['text'].map(lambda x: re.sub('[,\.!?]', '', x))
        # Convert the titles to lowercase
        self.tweets['text_processed'] = \
            self.tweets['text_processed'].map(lambda x: x.lower())

        self.data = self.tweets.text_processed.values.tolist()
        self.data_words = list(self.sent_to_words(self.data))
        # remove stop words
        self.data_words = self.remove_stopwords(self.data_words)

        return self.data_words

    def saveModel(self, model, texts, corpus, dict):
        with open('./models/gsdmm_models/model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('./models/gsdmm_models/model_texts.pkl', 'wb') as f:
            pickle.dump(texts, f)
        with open('./models/gsdmm_models/model_corpus.pkl', 'wb') as f:
            pickle.dump(corpus, f)
        with open('./models/gsdmm_models/model_dict.pkl', 'wb') as f:
            pickle.dump(dict, f)

    def train(self, data_words):
        # Create Dictionary
        self.id2word = corpora.Dictionary(data_words)

        # Create Corpus
        self.texts = data_words
        # Term Document Frequency
        self.corpus = [self.id2word.doc2bow(text) for text in self.texts]

        self.vocab_length = len(self.id2word)

        # initialize GSDMM
        self.gsdmm = MovieGroupProcess(K=20, alpha=0.1, beta=0.3, n_iters=22)

        # fit GSDMM model
        self.gsdmm.fit(self.texts, self.vocab_length, None)

        self.saveModel(self.gsdmm, data_words, self.corpus, self.id2word)

if __name__ == "__main__":
    mt = modelTrain('./data/measuring-hate-speech.parquet')
    data = mt.preProcessData(mt.tweets)
    mt.train(data)