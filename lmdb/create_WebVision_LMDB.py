import numpy as np
import lmdb
import caffe
import cv2
import glob

outout_path = 'testlmdb'

data_path = '../../../datasets/WebVision/'

img_names = []
img_classes = []

print "Reading indices"
sources=['google']#,'flickr']
for s in sources:
    data = []
    print 'Loading data from ' + s
    data_file = open(data_path + 'info/train_meta_list_' + s + '.txt', "r")
    img_list_file = open(data_path + 'info/train_filelist_' + s + '.txt', "r")

    c=0
    for line in img_list_file:
        if c == 100: break
        c+=1
        img_names.append(line.split(' ')[0])
        img_classes.append(int(line.split(' ')[1]))

N = len(img_names)
X = np.zeros((N, 3, 256, 256), dtype=np.uint8)
y = np.zeros(N, dtype=np.int64)

print "Number of images: " + str(N)

print "Reading images"
for i,img_path in enumerate(img_names):
    img = cv2.imread(data_path + img_path, cv2.IMREAD_COLOR)
    img = img[:, :, ::-1]
    img = img.transpose((2, 0, 1))
    X[i]=img
    y[i]=img_classes[i]

img_names = []
img_classes = []

# We need to prepare the database for the size. We'll set it 10 times
# greater than what we theoretically need. There is little drawback to
# setting this too big. If you still run into problem after raising
# this, you might want to try saving fewer entries in a single
# transaction.
map_size = X.nbytes * 10

env = lmdb.open(outout_path, map_size=map_size)

with env.begin(write=True) as txn:
    # txn is a Transaction object
    for i in range(N):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = X.shape[1]
        datum.height = X.shape[2]
        datum.width = X.shape[3]
        datum.data = X[i].tobytes()  # or .tostring() if numpy < 1.9
        datum.label = int(y[i])
        str_id = '{:08}'.format(i)

        # The encode is only essential in Python 3
        txn.put(str_id.encode('ascii'), datum.SerializeToString())

