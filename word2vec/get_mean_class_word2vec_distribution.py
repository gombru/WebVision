# Load trained LDA model and infer topics for unseen text.
# Make the train/val/test splits for CNN regression training
# It also creates the splits train/val/test randomly

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import glob
import string
from joblib import Parallel, delayed
import numpy as np
from random import randint
import json

# Load data and model
text_data_path = '../../../datasets/WebVision/'
model_path = '../../../datasets/WebVision/models/word2vec/word2vec_model_webvision.model'
tfidf_weighted = True
tfidf_model_path = '../../../datasets/WebVision/models/tfidf/tfidf_model_webvision.model'
tfidf_dictionary_path = '../../../datasets/WebVision/models/tfidf/docs.dict'

# Create output files
class_means_file = '../../../datasets/WebVision/word2vec_gt/class_means.txt'
class_means_file = open(class_means_file, "w")
class_num_file = '../../../datasets/WebVision/word2vec_gt/class_num.txt'
class_num_file = open(class_num_file, "w")

num_topics = 400
threads = 12

words2filter = ['wikipedia','google', 'flickr', 'figure', 'photo', 'image', 'homepage', 'url', 'youtube', 'images', 'blog', 'pinterest']

# create English stop words list
en_stop = get_stop_words('en')

# add own stop words
for w in words2filter:
    en_stop.append(w)

whitelist = string.letters + ' ' # + string.digits

model = gensim.models.Word2Vec.load(model_path)
tfidf_model = gensim.models.TfidfModel.load(tfidf_model_path)
tfidf_dictionary = gensim.corpora.Dictionary.load(tfidf_dictionary_path)
tokenizer = RegexpTokenizer(r'\w+')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()


def infer_word2vec(d):

        caption = d[1]
        filtered_caption = ""

        # Replace hashtags with spaces
        caption = caption.replace('#',' ')

        # Keep only letters and numbers
        for char in caption:
            if char in whitelist:
                filtered_caption += char

        filtered_caption = filtered_caption.lower()
        #Gensim simple_preproces instead tokenizer
        tokens = gensim.utils.simple_preprocess(filtered_caption)
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        tokens_filtered = [token for token in stopped_tokens if token in model.wv.vocab]

        embedding = np.zeros(num_topics)

        if not tfidf_weighted:
            c = 0
            for tok in tokens_filtered:
                try:
                    embedding += model[tok]
                    c += 1
                except:
                    #print "Word not in model: " + tok
                    continue
            if c > 0:
                embedding /= c

        if tfidf_weighted:
            vec = tfidf_dictionary.doc2bow(tokens_filtered)
            vec_tfidf = tfidf_model[vec]
            for tok in vec_tfidf:
                word_embedding = model[tfidf_dictionary[tok[0]]]
                embedding += word_embedding * tok[1]

        if len(vec_tfidf) > 0:
            embedding /= len(vec_tfidf)
        embedding = embedding - min(embedding)
        if max(embedding) > 0:
            embedding = embedding / max(embedding)

        if np.isnan(embedding).any():
            return

        if sum(embedding) < 0.00000001:
            return

        out = ""
        for e in embedding:
            out = out + ',' + str(e)

        return str(d[0]) + '_' + out



sources=['google','flickr']
former_filename = ' '

class_topics = np.zeros([1000,400])
class_instances = np.zeros([1000,1])

for s in sources:
    data = []
    print 'Loading data from ' + s
    data_file = open(text_data_path + 'info/train_meta_list_' + s + '.txt', "r")
    img_list_file = open(text_data_path + 'info/train_filelist_' + s + '.txt', "r")

    img_classes = []
    for line in img_list_file:
        img_classes.append(int(line.split(' ')[1]))

    for i,line in enumerate(data_file):

        # if i == 20000: break
        filename = line.split(' ')[0].replace(s,s+'_json')
        idx = int(line.split(' ')[1])

        if filename != former_filename:
            # print filename
            json_data = open(text_data_path + filename)
            d = json.load(json_data)
            former_filename = filename

        caption = ''

        if d[idx - 1].has_key('description'): caption = caption + d[idx - 1]['description'] + ' '
        if d[idx - 1].has_key('title'): caption = caption + d[idx - 1]['title'] + ' '
        if d[idx - 1].has_key('tags'):
            for tag in d[idx-1]['tags']:
                caption = caption + tag + ' '

        data.append([img_classes[i],caption])


    print "Number of elements for " + s + ": " + str(len(data))
    parallelizer = Parallel(n_jobs=threads)
    print "Infering word2vec scores"
    tasks_iterator = (delayed(infer_word2vec)(d) for d in data)
    r = parallelizer(tasks_iterator)
    # merging the output of the jobs
    strings = np.vstack(r)


    print "Resulting number of elements for " + s + ": " + str(len(strings))

    print "Saving results"
    for s in strings:

        try:
            img_d = s[0].split('_')
            class_id = int(img_d[0])
            distribution = np.fromstring(img_d[1][1:], dtype=float, sep=",")

            # Accumulate class stats
            class_topics[class_id,:] = class_topics[class_id,:] + distribution
            class_instances[class_id] = class_instances[class_id] + 1
        except:
            print "Error saving sample info"
            continue

    data_file.close()
    img_list_file.close()

for i in range(0, 1000):
    print "Num per class " + str(i) + ' : ' + str(class_instances[i])
    class_topics[i, :] = class_topics[i, :] / class_instances[i]
np.savetxt(class_means_file, class_topics, fmt='%10.5f', newline="\n")
np.savetxt(class_num_file, class_instances, fmt='%i', newline="\n")

class_num_file.close()
class_means_file.close()

print "Done"
