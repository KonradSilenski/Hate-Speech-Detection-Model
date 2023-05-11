import csv
import re
import random
import gensim
from gensim import corpora
import pandas as pd
import pickle
import nltk
from nltk.stem import SnowballStemmer, WordNetLemmatizer
nltk.download('punkt')
import numpy as np

class analyseData:
    def __init__(self, modeldir, dataset, isDataset):
        self.model = None
        with open(modeldir, 'rb') as f:
            self.model = pickle.load(f)

        self.doc_count = np.array(self.model.cluster_doc_count)
        self.top_index = self.doc_count.argsort()[-20:][::-1]

        self.topic_dict = {}
        self.topic_names = ['Topic #1',
                       'Topic #2',
                       'Topic #3',
                       'Topic #4',
                       'Topic #5',
                       'Topic #6',
                       'Topic #7',
                       'Topic #8',
                       'Topic #9',
                       'Topic #10',
                       'Topic #11',
                       'Topic #12',
                       'Topic #13',
                       'Topic #14',
                       'Topic #15',
                       'Topic #16',
                       'Topic #17',
                       'Topic #18',
                       'Topic #19',
                       'Topic #20'
                       ]
        for i, topic_num in enumerate(self.top_index):
            self.topic_dict[topic_num] = self.topic_names[i]

        text = None
        if isDataset == True:
            text = self.randomText(dataset)
        else:
            text = dataset
            text = text.lower()
        self.textout = text
        text_stems = self.preProcessData(text, self.getTopWords(self.model, 10))
        self.result = self.createTopicsDataframe(data_text=text_stems,  model=self.model, threshold=0.3, topic_dict=self.topic_dict)

    def sendResult(self):
        output = []
        output.append(self.result)
        output.append(self.textout)
        return output

    def randomText(self, data):
        if data.endswith('.csv'):
            self.df = pd.read_csv(data)
        elif data.endswith('.parquet'):
            self.df = pd.read_parquet(data)

        # Select a random row from the DataFrame
        self.random_row = self.df.sample(n=1)

        # Extract the text from the selected row
        self.text = self.random_row['text'].iloc[0]

        self.text = re.sub(r'[^a-zA-Z0-9\s]', '', self.text)

        self.text = self.text.lower()

        return self.text

    def preProcessData(self, text, top_words):
        # Preprocess the input text
        self.preprocessed_text = gensim.utils.simple_preprocess(text)
        self.stemmer = SnowballStemmer('english')
        self.lemmatizer = WordNetLemmatizer()
        self.words = nltk.word_tokenize(text)
        self.lemmatized_words = [self.lemmatizer.lemmatize(word) for word in self.words]
        self.filtered_words = [word for word in self.lemmatized_words if word in top_words]
        self.text_stems = [self.stemmer.stem(word) for word in self.filtered_words]
        return self.text_stems

    def getTopWords(self, model, n):
        # Get the top 10 words for each topic
        top_words = []
        for topic in range(model.K):
            word_count = model.cluster_word_distribution[topic]
            sorted_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
            top_words.append([word[0] for word in sorted_word_count[:n]])

        # Create a set that contains all the top 10 words for all the topics
        self.all_top_words = set([word for sublist in top_words for word in sublist])

        return self.all_top_words

    def createTopicsDataframe(self, data_text, model, threshold, topic_dict):
        self.result = pd.DataFrame(columns=['text', 'topic'])
        self.result.at[0, 'text'] = ' '.join(data_text)
        self.prob = model.choose_best_label(data_text)
        if self.prob[1] >= threshold:
            self.result.at[0, 'topic'] = topic_dict[self.prob[0]]
        else:
            self.result.at[0, 'topic'] = 'Other'
        return self.result

if __name__ == "__main__":
    ad = analyseData('./models/gsdmm_models/model.pkl', './data/measuring-hate-speech.parquet', True)
    print(ad.sendResult())
