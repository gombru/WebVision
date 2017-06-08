from load_regressions_from_txt import load_regressions_from_txt
from gensim import corpora, models
from get_classes_from_topics import get_top5_classes


database_path = '../../../datasets//WebVision/results/regression/WebVision_Inception_LDAfiltering_500_80000chunck_iter_320000/val.txt'
lda_model_path = '../../../datasets/WebVision/models/LDA/lda_model_500_80000chunck.model'
num_topics = 500
max_per_topic = 100
max_per_sample = 5
topic_threshold = 0.005
word_threshold = 0.0015

output_file_path = '../../../datasets/WebVision/results/regression/WebVision_Inception_LDAfiltering_500_80000chunck_iter_320000/regression2results_500_raw_topth_' + str(topic_threshold) + 'ingth_' + str(word_threshold) + '.txt'


# Get topics and associated ingredients
ldamodel = models.ldamodel.LdaModel.load(lda_model_path)
# topics = ldamodel.print_topics(num_topics=num_topics, num_words=20)
#Get dictionary

out_file = open(output_file_path,'w')

database = load_regressions_from_txt(database_path, num_topics)

for id in database: #For each image

    inferred_classes = []
    aux_list = []
    for t,topic_value in enumerate(database[id]): #For each topic
        if topic_value > topic_threshold: #Topic may contribute with a word
            for word in ldamodel.show_topic(t,max_per_topic): #For each word associated to the topic
                word_score = topic_value * word[1] #Topic prob * word prob = word confidence
                if word_score > word_threshold:
                    try:
                        if word[0] not in aux_list:
                            aux_list.append(word[0])
                            inferred_classes.append([word[0], word_score])
                        else:
                            index = aux_list.index(word[0])
                            #score = max(img_ing[1][1], word_score)
                            score = inferred_classes[index][1] + word_score
                            inferred_classes[index] = [word[0], score]
                    except:
                        print "Failed saving class: "
                        print word

    # Sort list by ingredient confidence
    inferred_classes = sorted(inferred_classes, key=lambda x: -x[1])
    out_file.write(id.strip('\n') + ',')

    print id + ' --  Num ing: ' + str(len(inferred_classes))

    for i in range(0,max_per_sample):
        if i >= len(inferred_classes):
            print "WARNING!!!! Less than 5 classes infered!"
            break
        out_file.write(inferred_classes[i][0] + ',')
    out_file.write('\n')

out_file.close()
print 'DONE'
print output_file_path

print "Computing P-R:"
get_top5_classes(output_file_path)

