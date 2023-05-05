import pandas as pd
import os
import matplotlib.pyplot as plt
import re
from wordcloud import WordCloud

os.chdir('..')
# Read data into papers
papers = pd.read_parquet('C:/Users/Minec/PycharmProjects/dissTopicModel/venv/data/measuring-hate-speech.parquet')
# Remove the columns
papers = papers[['text']]

# Remove punctuation
papers['text_processed'] = \
papers['text'].map(lambda x: re.sub('[,\.!?]', '', x))
# Convert the titles to lowercase
papers['text_processed'] = \
papers['text_processed'].map(lambda x: x.lower())
# Print out the first rows of papers
print(papers['text_processed'].head())
# Import the wordcloud library
# Join the different processed titles together.
long_string = ','.join(list(papers['text_processed'].values))
# Import the wordcloud library
# Join the different processed titles together.
long_string = ','.join(list(papers['text_processed'].values))
# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud.generate(long_string)
# Visualize the word cloud
image = wordcloud.to_image()
image.show()