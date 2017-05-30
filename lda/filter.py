

filter = False
start = 10000000
lines = []

with open('../../../datasets/WebVision/lda_gt/bad/train_500_chunck80000.txt', 'r') as annsfile:
    for c,i in enumerate(annsfile):
        print i[0]
        if i[0] == 'f':
            start = c
            break

print "GOOGLE IMGS = " + str(start/2)

with open('../../../datasets/WebVision/lda_gt/bad/train_500_chunck80000.txt', 'r') as annsfile:
    for c,i in enumerate(annsfile):
        #print i[0]
        if c < start/2: lines.append(i)
        if c > start: lines.append(i)


print "TOTAL IMGS = " +str(len(lines))

with open('../../../datasets/WebVision/lda_gt/bad/train_500_chunck80000_fil.txt', 'w') as annsfile:
    for l in lines:
        annsfile.write(l)



