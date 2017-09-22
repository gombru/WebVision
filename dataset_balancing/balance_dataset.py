import numpy as np

path = "../../../datasets/WebVision/info/train_filelist_all.txt"
dest_path =  "../../../datasets/WebVision/info/train_balanced_filelist.txt"
file = open(path, "r")

print("Loading data ...")
print(path)

listofclasses = {}
for c in range(0,1000):
    listofclasses[c] = []

# Load data
for line in file:
    d = line.split()
    listofclasses[int(d[1])].append(d[0])
file.close()

# Count number per class
numxclass = np.zeros((1000,1))
for c in range(0,1000):
    numxclass[c] = len(listofclasses[c])
maxxclass = max(numxclass)
print "Max per class: " + str(maxxclass)

minxclass = int(maxxclass - maxxclass * 0.5)
print "Min per class: " + str(minxclass)


print "Writing data"
# Write data balancing
file = open(dest_path, "w")
for c in range(0,1000):
    elements_writed = 0
    while elements_writed <= minxclass:
        for el in listofclasses[c]:
            file.write(el + " " + str(c) + "\n")
            elements_writed += 1
            if elements_writed > minxclass and elements_writed > numxclass[c]: break
    print "Class " + str(c) + " : " + str(elements_writed)

file.close()
print "DONE"

