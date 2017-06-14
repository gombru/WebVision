import sys
import caffe
import numpy as np
from PIL import Image
import os

# Run in GPU
caffe.set_device(0)
caffe.set_mode_gpu()

test = np.loadtxt('../../../datasets/WebVision/info/val_filelist.txt', dtype=str)

#Model name
model = 'WebVision_Inception_LDAfiltering_500_80000chunck_iter_1440000'

#Output file
output_file_dir = '../../../datasets/WebVision/results/classification_crop/' + model
if not os.path.exists(output_file_dir):
    os.makedirs(output_file_dir)
output_file_path = output_file_dir + '/val.txt'
output_file = open(output_file_path, "w")

# load net
net = caffe.Net('../googlenet/prototxt/deploy.prototxt', '../../../datasets/WebVision/models/saved/'+ model + '.caffemodel', caffe.TEST)

resize = True
size = 224

# Images are 256*>256. The idea is to crop the 256x256 center and resize the square image to 224 to avoid distortion
# That makes sense because we have trained the net with crops, avoiding image distortion.
crop = True
crop_size = 256

# Reshape net
batch_size = 200
net.blobs['data'].reshape(batch_size, 3, size, size)

print 'Computing  ...'

count = 0
i = 0
while i < len(test):
    indices = []
    if i % 100 == 0:
        print i

    # Fill batch
    for x in range(0, batch_size):

        if i > len(test) - 1: break

        # load image
        filename = '../../../datasets/WebVision/val_images_256/' + test[i][0]
        im = Image.open(filename)
        im_o = im


        #Crop center
        if crop:
            # Crops the central sizexsize part of an image
            width, height = im.size
            if width != crop_size:
                left = (width - crop_size) / 2
                right = width - left
                im = im.crop((left, 0, right, height))
            if height != crop_size:
                top = (height - crop_size) / 2
                bot = height - top
                im = im.crop((0, top, width, bot))

        #Resize
        if resize:
            im = im.resize((size, size), Image.ANTIALIAS)


        indices.append(test[i][0])

        # Turn grayscale images to 3 channels
        if (im.size.__len__() == 2):
            im_gray = im
            im = Image.new("RGB", im_gray.size)
            im.paste(im_gray)

        #switch to BGR and substract mean
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= np.array((104, 117, 123))
        in_ = in_.transpose((2,0,1))

        net.blobs['data'].data[x,] = in_

        i += 1

    # run net and take scores
    net.forward()

    # Save results for each batch element
    for x in range(0,len(indices)):
        probs = net.blobs['probs'].data[x]
        top5 = probs.argsort()[::-1][0:5]
        top5str = ''

        for t in top5:
            top5str = top5str + ',' + str(t)

        output_file.write(indices[x] + top5str + '\n')

output_file.close()

print "DONE"


