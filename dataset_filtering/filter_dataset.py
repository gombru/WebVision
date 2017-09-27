# Inputs:
# - Noise per class
# - CNN top5 prediction per training sample
# - Mean word2vec embedding of classes
#
# Computes the distance between training sample word2vec embedding and the mean word2vec embedding of the class. Ranks the
# samples according to the similarity with the mean text embedding. Discard samples starting from the ones with more distance.
# Discard those samples where the CNN also fails with the TOP5. Stop discarding when the maximum noise per class
# has been reached.


from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import string
from joblib import Parallel, delayed
import numpy as np
import json
import gensim
import os
from shutil import copyfile


maximum_noise_per_class = 30 # %
CNN_error = 15 # CNN mean class error on clean data

results = {}
for i in xrange(1000):
    results[i] = []

# Create output files
output_file_path = '../../../datasets/WebVision/info/train_filtered.txt'
output_file = open(output_file_path, "w")
num_filtered_file_path = '../../../datasets/WebVision/info/num_filtered.txt'
num_fitlered_file = open(num_filtered_file_path, "w")

# Word2vec files
text_data_path = '../../../datasets/WebVision/'
model_path = '../../../datasets/WebVision/models/word2vec/word2vec_model_webvision.model'
tfidf_weighted = True
tfidf_model_path = '../../../datasets/WebVision/models/tfidf/tfidf_model_webvision.model'
tfidf_dictionary_path = '../../../datasets/WebVision/models/tfidf/docs.dict'

# Read class means
class_means_file = '../../../datasets/WebVision/word2vec_gt/class_means.txt'
class_means = np.loadtxt(class_means_file)

filtered = np.zeros([1000,1])

# Read noise per class
class_noise_file = '../../../datasets/WebVision/info/CNN_noise_classes.txt'
class_noise = np.loadtxt(class_noise_file, dtype=str)

# Read top5 per training samples
top5s_file_path = '../../../datasets/WebVision/results/classification_crop/WebVision_Inception_iter_120000/train.txt'
top5s_file = np.loadtxt(top5s_file_path, dtype=str)
top5s = {}

for el in top5s_file:
    top5s[el[0]] = el[1:]
top5s_file = []

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

        caption = d[2]
        filtered_caption = ""

        # Replace hashtags with spaces
        caption = caption.replace('#',' ')
        caption = caption.replace('_', ' ')

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
            embedding =  np.zeros(num_topics)

        if sum(embedding) < 0.00000001:
            embedding =  np.zeros(num_topics)

        out = ""
        for e in embedding:
            out = out + ',' + str(e)


        return str(d[0]) + '#' + str(d[1]) + '#' + out



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
    img_names = []
    for line in img_list_file:
        img_names.append(line.split(' ')[0])
        img_classes.append(int(line.split(' ')[1]))

    for i,line in enumerate(data_file):

        #if i == 5: break
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

        data.append([img_names[i], img_classes[i], caption])


    print "Number of elements for " + s + ": " + str(len(data))
    parallelizer = Parallel(n_jobs=threads)
    print "Infering word2vec scores"
    tasks_iterator = (delayed(infer_word2vec)(d) for d in data)
    r = parallelizer(tasks_iterator)
    # merging the output of the jobs
    strings = np.vstack(r)


    print "Resulting number of elements for " + s + ": " + str(len(strings))

    for s in strings:
        try:
            img_d = s[0].split('#')
            img_name = img_d[0]
            class_id = int(img_d[1])
            distribution = np.fromstring(img_d[2][1:], dtype=float, sep=",")
            score = np.dot(distribution,class_means[class_id,:])
            results[class_id].append([img_name, score])
        except:
            print "Error with sample " + str(s[0])

num_x_class = np.zeros([1000,1])
for i in xrange(1000):
    num_x_class[i] = len(results[i])

# Classes to save discarded images
classes=[5,50,100,200,250,300,400,520,600,800]

# Sort lists by score in ascendent order, so by ditance to mean class and discard outliers
for i in xrange(1000):

    print "Class: " + str(i) + " Elements: " + str(num_x_class[i])
    percent_samples_2_discard = float(class_noise[i]) - CNN_error
    if percent_samples_2_discard > maximum_noise_per_class:
        percent_samples_2_discard = maximum_noise_per_class
        print "Samples to discard changed from " + str(percent_samples_2_discard) + " to " + str(maximum_noise_per_class)
    if percent_samples_2_discard <= 0:
        print "Not discarding samples for class " + str(i)
        continue

    samples_to_discard = num_x_class[i] * float(percent_samples_2_discard)/100

    if i in classes:
        if not os.path.exists('../../../datasets/WebVision/far_from_mean/' + str(i)):
            os.makedirs('../../../datasets/WebVision/far_from_mean/' + str(i))

    results[i] = sorted(results[i], key=lambda x:x[1])
    for el in results[i]:
        if i in top5s[el[0]]:
            continue
        else:
            results[i].remove(el)
            filtered[i] += 1
            if i in classes:
                copyfile('../../../datasets/WebVision/' + el[0], '../../../datasets/WebVision/far_from_mean/' + str(i) + '/' + el[0].split('/')[-1])

        if filtered[i] > samples_to_discard:
            break

    print "Class: " + str(i) + " -- " + str(filtered[i]) + " from " + str(num_x_class[i]) + " discarded (" + str(percent_samples_2_discard) + "%)"

for i in xrange(1000):
    for el in results[i]:
        output_file.write(el[0] + ' ' + str(i) + "\n")
    num_fitlered_file.write(str(int(num_x_class[i][0])) + ' ' + str(int(filtered[i][0])) + "\n")

print "Done"
