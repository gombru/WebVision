import os
import sys
from nltk.stem.porter import PorterStemmer




def get_top5_classes(results_path):

    out_file = open(results_path.replace('WebVision','TOP5_WebVision'), 'w')

    results = {}

    p_stemmer = PorterStemmer()

    classes_names_path = '../../../datasets/WebVision/class_names.txt'
    classes_names_file = open(classes_names_path, 'r')
    class_names=[]

    for line in classes_names_file:
        class_names.append(line.strip('\n'))

    file = open(results_path, "r")

    print "Loading data ..."
    print results_path

    for line in file:   #For each sample
        img_classes = [] #Will store results for this class
        print line

        d = line.split(',')

        for t in range(1,len(d) - 1):  #For each inferred word try top match it with one GT class

            for i,c in enumerate(class_names): #For each class name
                class_name_words = c.split(' ')

                for cw in class_name_words: #For each word in a class name

                    cw = cw.decode('utf-8').lower()
                    cw = p_stemmer.stem(cw)

                    if d[t] == cw and i not in img_classes: #If word of a gt class has matched with an inferred word and that class hasn't been assigned yet.
                        img_classes.append(i)    #Assign class to image
                        break

            if len(img_classes) == 5: break #Output only top socred 5 matches

        results[d[0]] = img_classes #Save img id and its inferred classes in a dict to the save them


    for img_name, classes in results.iteritems():
        out_file.write(img_name)
        for c in classes :
            out_file.write(',' + str(c))
        out_file.write('\n')


