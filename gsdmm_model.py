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

# Read data into papers
papers = pd.read_parquet('C:/Users/Minec/PycharmProjects/dissTopicModel/venv/data/measuring-hate-speech.parquet')
#papers2 = pd.read_csv('C:/Users/Minec/PycharmProjects/dissTopicModel/venv/data/hatespeechdset.csv')
# Remove the columns
papers = papers[['text']]
#papers2 = papers2[['text']]

# Remove punctuation
papers['text_processed'] = \
papers['text'].map(lambda x: re.sub('[,\.!?]', '', x))
#papers2['text2_processed'] = \
#papers2['text'].map(lambda x: re.sub('[,\.!?]', '', x))
# Convert the titles to lowercase
papers['text_processed'] = \
papers['text_processed'].map(lambda x: x.lower())
#papers2['text2_processed'] = \
#papers2['text2_processed'].map(lambda x: x.lower())

stop_words = stopwords.words('english')
stop_words.extend(['url', 'need', 'want', 'the', 'u', 'etc', 'sorry', 'help', 'cute', 'one', 'please', 'like', 'many', 'love', 'get', 'say', 'think', 'good', 'think',
                   'love', 'would', 'saying', 'much', 'see', 'great', 'could', 'us', 'time', 'makes', 'world', 'let', 'nice', 'full', 'really', 'well', 'wish', 'full',
                   'al', 'igbo', 'abeg', 'miraj', 'lailat', 'make', 'new', 'nawaz', 'nugs', 'mubarak', 'notwithstanding', 'sucking', 'yt', 'ya', 'ur', 'show', 'add',
                   'sh', 'tr', 'losangeles', 'children', 'hope', 'even', 'work', 'know', 'eat', 'look', 'wa', 'ha'])

lemmatizer = WordNetLemmatizer()

def sent_to_words(sentences):
    for sentence in sentences:
        yield([lemmatizer.lemmatize(word) for word in gensim.utils.simple_preprocess(str(sentence), deacc=True)])
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]

data = papers.text_processed.values.tolist()
#papers2_list = papers2.text2_processed.tolist()
#data.extend(papers2_list)
print(len(data))
data_words = list(sent_to_words(data))
# remove stop words
data_words = remove_stopwords(data_words)

# Create Dictionary
id2word = corpora.Dictionary(data_words)

# Create Corpus
texts = data_words
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

vocab_length = len(id2word)

# initialize GSDMM
gsdmm = MovieGroupProcess(K=20, alpha=0.1, beta=1, n_iters=14)

# fit GSDMM model
gsdmm.fit(texts, vocab_length)

with open('model.pkl', 'wb') as f:
    pickle.dump(gsdmm, f)