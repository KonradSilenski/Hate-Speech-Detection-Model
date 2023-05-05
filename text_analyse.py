import csv
import random
import gensim
from gensim import corpora
import pandas as pd
import pickle
import nltk
from nltk.stem import SnowballStemmer
nltk.download('punkt')
import numpy as np


# with open('C:/Users/Minec/PycharmProjects/dissTopicModel/venv/data/hatespeechdset.csv', 'r', encoding="utf8") as csvfile:
#     csvreader = csv.reader(csvfile)
#     rows = list(csvreader)
#
# random_row = random.choice(rows)
# text = random_row[3]

df = pd.read_parquet('C:/Users/Minec/PycharmProjects/dissTopicModel/venv/data/measuring-hate-speech.parquet')

# Select a random row from the DataFrame
random_row = df.sample(n=1)

# Extract the text from the selected row
text = random_row['text'].iloc[0]

text = text.lower()
print(text)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Get the top 10 words for each topic
top_words = []
for topic in range(model.K):
        word_count = model.cluster_word_distribution[topic]
        sorted_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        top_words.append([word[0] for word in sorted_word_count[:50]])

# Create a set that contains all the top 10 words for all the topics
all_top_words = set([word for sublist in top_words for word in sublist])

# Preprocess the input text
preprocessed_text = gensim.utils.simple_preprocess(text)

stemmer = SnowballStemmer('english')
words = nltk.word_tokenize(text)
filtered_words = [word for word in words if word in all_top_words]
text_stems = [stemmer.stem(word) for word in filtered_words]
print(text_stems)

doc_count = np.array(model.cluster_doc_count)
top_index = doc_count.argsort()[-20:][::-1]

topic_dict = {}
topic_names = ['Topic #1',
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
for i, topic_num in enumerate(top_index):
    topic_dict[topic_num]=topic_names[i]

def create_topics_dataframe(data_text=text_stems,  model=model, threshold=0.3, topic_dict=topic_dict):
    result = pd.DataFrame(columns=['text', 'topic'])
    result.at[0, 'text'] = ' '.join(data_text)
    prob = model.choose_best_label(data_text)
    print(prob)
    print(len(data_text))
    if prob[1] >= threshold:
        result.at[0, 'topic'] = topic_dict[prob[0]]
    else:
        result.at[0, 'topic'] = 'Other'
    return result


dfx = create_topics_dataframe(data_text=text_stems,  model=model, threshold=0.3, topic_dict=topic_dict)

print(dfx)