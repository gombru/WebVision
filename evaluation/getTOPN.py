import numpy as np

data = np.loadtxt('../../../datasets//WebVision/results/classification/WebVision_Inception_LDAfiltering_500_80000chunck_iter_320000/val.txt', dtype=str)
test = np.loadtxt('../../../datasets/WebVision/info/val_filelist.txt', dtype=str)

correct_file_dir = '../../../datasets/WebVision/results/classification/WebVision_Inception_LDAfiltering_500_80000chunck_iter_320000/correctPerClass.txt'
correct_file = open(correct_file_dir,'w')

wrong_file_dir = '../../../datasets/WebVision/results/classification/WebVision_Inception_LDAfiltering_500_80000chunck_iter_320000/WrongPerClass.txt'
wrong_file = open(wrong_file_dir,'w')

correct_class = np.zeros([1000,1])
wrong_class = np.zeros([1000,1])

top1 = 0
top5 = 0

for i,r in enumerate(data):
    r = r.split(',')
    gt_label = test[i][1]

    if gt_label == r[1]:
        top1 += 1
        top5 += 1
        correct_class[int(gt_label)] += 1
    else:
        wrong_class[int(gt_label)] += 1

    if gt_label in r[2:6]:
        top5 += 1

total_top1 = float(top1) / len(data)
total_top5 = float(top5) / len(data)

print "TOP1: " + str(total_top1)
print "TOP5: " + str(total_top5)


for i in range(0,1000):
    correct_file.write(str(int(correct_class[i][0])) + '\n')
    wrong_file.write(str(int(wrong_class[i][0])) + '\n')


correct_file.close()
wrong_file.close()



