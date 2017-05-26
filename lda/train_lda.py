# Trains and saves an LDA model with the given text files.

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import glob
import string
import json

import numpy as np


whitelist = string.letters  + ' ' # + string.digits
text_data_path = '../../../datasets/WebVision/'
model_path = '../../../datasets/WebVision/models/LDA/lda_model_500_30000chunck.model'
words2filter = ['wikipedia','google', 'flickr', 'figure', 'photo', 'image', 'homepage', 'url', 'youtube']

num_topics = 500
threads = 8
passes = 1 #Passes over the whole corpus
chunksize = 80000 #Update the model every 10000 documents
# See https://radimrehurek.com/gensim/wiki.html
update_every = 1

repetition_threshold = 150

#Initialize Tokenizer
tokenizer = RegexpTokenizer(r'\w+')
# create English stop words list
en_stop = get_stop_words('en')
# add own stop words
for w in words2filter:
    en_stop.append(w)
# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

posts_text = []
texts = [] #List of lists of tokens

# -- LOAD DATA FROM INSTAGRAM --
former_filename = ' '
print "Loading data"
file = open(text_data_path + 'info/train_meta_list_all.txt', "r")

for line in file:

    filename = line.split(' ')[0]
    idx = int(line.split(' ')[1])

    if filename != former_filename:
        print filename
        json_data = open(text_data_path + filename)
        d = json.load(json_data)
        former_filename = filename

    caption = ''
    filtered_caption = ''

    if d[idx-1].has_key('description'): caption = caption + d[idx-1]['description'] + ' '
    if d[idx-1].has_key('title'): caption = caption + d[idx-1]['title'] + ' '
    if d[idx-1].has_key('tags'):
        for tag in d[idx-1]['tags']:
            caption = caption + tag + ' '


    # Replace hashtags with spaces
    caption = caption.replace('#', ' ')
    # Keep only letters and numbers
    for char in caption:
        if char in whitelist:
            filtered_caption += char

    posts_text.append(filtered_caption.decode('utf-8').lower())

    # print filtered_caption.decode('utf-8').lower()

print "Number of posts: " + str(len(posts_text))

print "Creating tokens"
c= 0

for t in posts_text:

    c += 1
    if c % 10000 == 0:
        print c

    try:
        t = t.lower()
        tokens = tokenizer.tokenize(t)
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        # stem token
        text = [p_stemmer.stem(i) for i in stopped_tokens]
        # add proceced text to list of lists
        texts.append(text)
    except:
        continue
    #Remove element from list if memory limitation TODO
    #del tweets_text[0]
posts_text = []

# Remove words that appear less than N times
print "Removing words appearing less than: " + str(repetition_threshold)
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
texts = [[token for token in text if frequency[token] > repetition_threshold] for text in texts]

# Construct a document-term matrix to understand how frewuently each term occurs within each document
# The Dictionary() function traverses texts, assigning a unique integer id to each unique token while also collecting word counts and relevant statistics.
# To see each token unique integer id, try print(dictionary.token2id)
dictionary = corpora.Dictionary(texts)
print(dictionary)


# TODO check this
# dictionary.compactify()
# Filter out tokens that appear in less than no_below documents (absolute number) or more than no_above documents (fraction of total corpus size, not absolute number).
# after (1) and (2), keep only the first keep_n most frequent tokens (or keep all if None).
# dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=None)
# dictionary.compactify()  # remove gaps in id sequence after words that were removed

# Convert dictionary to a BoW
# The result is a list of vectors equal to the number of documents. Each document containts tumples (term ID, term frequency)
corpus = [dictionary.doc2bow(text) for text in texts]

texts = []

#Randomize training elements
corpus = np.random.permutation(corpus)


# Generate an LDA model
print "Creating LDA model"
 # the minimum_probability=0 argument is necessary in order for
# gensim to return the full document-topic-distribution matrix.  If
# this argument is omitted and left to the gensim default of 0.01,
# then all document-topic weights below that threshold will be
# returned as NaN, violating the subsequent LDAvis assumption that
# all rows (documents) in the document-topic-distribution matrix sum
# to 1.

#ldamodel = models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word = dictionary, passes=passes, minimum_probability=0)
ldamodel = models.LdaMulticore(corpus, num_topics=num_topics, id2word = dictionary, chunksize=chunksize, passes=passes, workers=threads, minimum_probability=0)
ldamodel.save(model_path)
# Our LDA model is now stored as ldamodel

print(ldamodel.print_topics(num_topics=8, num_words=10))

print "DONE"







