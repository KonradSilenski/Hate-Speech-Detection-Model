import IPython
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pprint import pprint

import pandas as pd
import os
import matplotlib.pyplot as plt
import re

import pyLDAvis.gensim
import pickle
import pyLDAvis



os.chdir('..')
# Read data into papers
papers = pd.read_parquet('C:/Users/Minec/PycharmProjects/dissTopicModel/venv/data/measuring-hate-speech.parquet')
papers2 = pd.read_csv('C:/Users/Minec/PycharmProjects/dissTopicModel/venv/data/hatespeechdset.csv')
# Remove the columns
papers = papers[['text']]
papers2 = papers2[['text']]

# Remove punctuation
papers['text_processed'] = \
papers['text'].map(lambda x: re.sub('[,\.!?]', '', x))
papers2['text2_processed'] = \
papers2['text'].map(lambda x: re.sub('[,\.!?]', '', x))
# Convert the titles to lowercase
papers['text_processed'] = \
papers['text_processed'].map(lambda x: x.lower())
papers2['text2_processed'] = \
papers2['text2_processed'].map(lambda x: x.lower())

stop_words = stopwords.words('english')
stop_words.extend(['url', 'need', 'want', 'the', 'u', 'etc', 'sorry', 'help', 'cute', 'one', 'please', 'like', 'many', 'love', 'get', 'say', 'think', 'good', 'think',
                   'love', 'would', 'saying', 'much', 'see', 'great', 'could', 'us', 'time', 'makes', 'world', 'let', 'nice', 'full', 'really', 'well', 'wish', 'full',
                   'al', 'igbo', 'abeg', 'miraj', 'lailat', 'make', 'new', 'nawaz', 'nugs', 'mubarak', 'notwithstanding', 'sucking', 'yt', 'ya', 'ur', 'show', 'add',
                   'sh', 'tr'])

lemmatizer = WordNetLemmatizer()

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(lemmatizer.lemmatize(str(doc)))
             if word not in stop_words] for doc in texts]

data = papers.text_processed.values.tolist()
papers2_list = papers2.text2_processed.tolist()
data.extend(papers2_list)
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
# View
print(corpus[:1][0][:30])

if __name__ ==  '__main__':
    num_topics = 10
    random_st = 200
    update = 1
    chunks = 2000
    passes = 10
    iterations = 5
    min_probab= 0.01

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=num_topics,
                                                random_state=random_st,
                                                update_every=update,
                                                chunksize=chunks,
                                                passes=passes,
                                                alpha='auto',
                                                per_word_topics=True,
                                                iterations=iterations,
                                                minimum_probability=min_probab)
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]

    print('\nPerplexity: ', lda_model.log_perplexity(corpus))

    lda_model.save('C:/Users/Minec/PycharmProjects/dissTopicModel/venv/models/model.lda')

    LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(LDAvis_prepared, 'C:/Users/Minec/PycharmProjects/dissTopicModel/venv/results/ldavis_prepared_' + str(num_topics) + '.html')
    LDAvis_prepared