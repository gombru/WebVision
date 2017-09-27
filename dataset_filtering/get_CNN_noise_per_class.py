import numpy as np


model = 'WebVision_Inception_iter_120000'

print "Loading data ..."
data = np.loadtxt('../../../datasets/WebVision/results/classification_crop/'+model+'/train.txt', dtype=str)
test = np.loadtxt('../../../datasets/WebVision/info/train_filelist_all.txt', dtype=str)

noise_file_path = '../../../datasets/WebVision/info/CNN_noise_classes.txt'
noise_file = open(noise_file_path,'w')

correct_class = np.zeros([1000,1])
wrong_class = np.zeros([1000,1])

top5 = 0

print "Computing ..."

for i,r in enumerate(data):

    gt_label = test[i][1]
    # r = r.split(',')

    if gt_label in r[1:6]:
        top5 += 1
        correct_class[int(gt_label)] += 1

    else:
        wrong_class[int(gt_label)] += 1


total_top5 = float(top5) / len(data)

print "TOP5: " + str(total_top5)


for i in range(0,1000):
    noisy = wrong_class[int(i)] / (correct_class[i][0] + wrong_class[int(i)])
    noise_file.write(str(noisy[0] * 100) + '\n')

noise_file.close()


