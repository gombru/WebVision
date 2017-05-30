# Load trained LDA model and infer topics for unseen text.
# Make the train/val/test splits for CNN regression training
# It also creates the splits train/val/test randomly

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import glob
import string
from joblib import Parallel, delayed
import numpy as np
from random import randint
import json

# Load data and model
text_data_path = '../../../datasets/WebVision/'
model_path = '../../../datasets/WebVision/models/LDA/lda_model_500_80000chunck.model'

# Create output files
class_means_file = '../../../datasets/WebVision/lda_gt/class_means_500_80000chunck.txt'
class_means_file = open(class_means_file, "w")
class_num_file = '../../../datasets/WebVision/lda_gt/class_num.txt'
class_num_file = open(class_num_file, "w")



num_topics = 500
threads = 12

words2filter = ['wikipedia','google', 'flickr', 'figure', 'photo', 'image', 'homepage', 'url', 'youtube']

# create English stop words list
en_stop = get_stop_words('en')

# add own stop words
for w in words2filter:
    en_stop.append(w)

whitelist = string.letters + ' ' # + string.digits

ldamodel = models.ldamodel.LdaModel.load(model_path)
tokenizer = RegexpTokenizer(r'\w+')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

topics = ldamodel.print_topics(num_topics=num_topics, num_words=50)

# print topics

# Save a txt with the topics and the weights
file = open('topics.txt', 'w')
i = 0
for item in topics:
    file.write(str(i) + " - ")
    file.write("%s\n" % item[1])
    i+=1
file.close()


def infer_LDA(d):

        caption = d[1]
        filtered_caption = ""

        # Replace hashtags with spaces
        caption = caption.replace('#',' ')

        # Keep only letters and numbers
        for char in caption:
            if char in whitelist:
                filtered_caption += char

        filtered_caption = filtered_caption.lower()

        tokens = tokenizer.tokenize(filtered_caption)
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        # stem token

        # Handle stemmer error
        while "aed" in stopped_tokens:
            stopped_tokens.remove("aed")
            print "aed error"

        try:
            text = [p_stemmer.stem(i) for i in stopped_tokens]
            bow = ldamodel.id2word.doc2bow(text)
            #r = ldamodel[bow] # Warning, this uses a threshold of 0.01 on tropic probs, and usually returns only 1 max 2...
            r = ldamodel.get_document_topics(bow,  minimum_probability=0) #This 0 is changed to 1e-8 inside
            #print len(r)
        except:
            print "Tokenizer error"
            print stopped_tokens
            return


        # GT for regression
        # Add zeros to topics without score
        topic_probs = ''
        for t in range(0,num_topics):
            assigned = False
            for topic in r:
                    if topic[0] == t:
                        topic_probs = topic_probs + ',' + str(topic[1])
                        assigned = True
                        continue
            if not assigned:
                topic_probs = topic_probs + ',' + '0'

        # print id + topic_probs
        return str(d[0]) + '_' + topic_probs



sources=['google','flickr']
former_filename = ' '

class_topics = np.zeros([1000,500])
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

        if i == 10: break
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
    print "Infering LDA scores"
    tasks_iterator = (delayed(infer_LDA)(d) for d in data)
    r = parallelizer(tasks_iterator)
    # merging the output of the jobs
    strings = np.vstack(r)



    print "Resulting number of elements for " + s + ": " + str(len(strings))

    print "Saving results"
    for s in strings:

        img_d = s[0].split('_')
        class_id = int(img_d[0])
        distribution = np.fromstring(img_d[1][1:], dtype=float, sep=",")

        # Accumulate class stats
        class_topics[class_id,:] = class_topics[class_id,:] + distribution
        class_instances[class_id] = class_instances[class_id] + 1

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
