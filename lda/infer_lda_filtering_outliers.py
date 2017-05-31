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
train_gt_path = '../../../datasets/WebVision/lda_gt/train_filteredbyLDA_500_chunck80000.txt'
train_file = open(train_gt_path, "w")

filtered_file_path = '../../../datasets/WebVision/lda_gt/numfilteredbyLDA_500_chunck80000.txt'
filtered_file = open(filtered_file_path, "w")

# Read mean class topic distributions
mean_class_distributions_path = '../../../datasets/WebVision/lda_gt/class_means_500_80000chunck.txt'
mean_class_distributions = np.loadtxt(mean_class_distributions_path)
filtered = np.zeros([1000,1])

score_th = 0.005

num_topics = 500
threads = 12

words2filter = ['wikipedia','google', 'flickr', 'figure', 'photo', 'image', 'homepage', 'url', 'youtube', 'images', 'blog', 'pinterest']

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

        caption = d[2]
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

        # Compute distance between mean topic distribution of the class and this sample
        distribution = np.fromstring(topic_probs[1:], dtype=float, sep=",")
        score = np.dot(distribution,mean_class_distributions[int(d[1]),:])

        if score < score_th :
            #print score
            return "_" + ' ' + str(d[0]) + ' ' + str(d[1])

        # print id + topic_probs
        return str(d[0]) + ' ' + str(d[1])



sources=['google','flickr']
former_filename = ' '
for s in sources:
    data = []
    print 'Loading data from ' + s
    data_file = open(text_data_path + 'info/train_meta_list_' + s + '.txt', "r")
    img_list_file = open(text_data_path + 'info/train_filelist_' + s + '.txt', "r")

    img_names = []
    img_classes = []
    for line in img_list_file:
        img_names.append(line.split(' ')[0])
        img_classes.append(int(line.split(' ')[1]))

    for i,line in enumerate(data_file):

        #if i == 4000: break

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

        data.append([img_names[i],img_classes[i],caption])


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

        if s[0].split(' ')[0] == '_':
            filtered[int(s[0].split(' ')[2])] = filtered[int(s[0].split(' ')[2])] + 1
            continue
        # Create splits random
        try:
            train_file.write(s[0] + '\n')
        except:
            print "Error writing to file: "
            print s[0]
            continue

    data_file.close()
    img_list_file.close()

for i, f in enumerate(filtered):
    print "Filtered in Class " + str(i) + ": " + str(f)

np.savetxt(filtered_file, filtered, fmt='%i', newline="\n")
train_file.close()
filtered_file.close()

print "Done"