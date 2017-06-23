import sys
import caffe
import numpy as np
from PIL import Image
import os


#switch to BGR and substract mean
def preprocess(im):
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104, 117, 123))
    in_ = in_.transpose((2,0,1))
    return in_

# Run in GPU
caffe.set_device(0)
caffe.set_mode_gpu()

split = 'val'

test = np.loadtxt('../../../datasets/WebVision/info/'+split+'_filelist.txt', dtype=str)

#Model name
model = 'WebVision_Inception_LDAscored_500_80000chunck_iter_300000'

#Ensemble 2 classifiers
ensembleClassifiers = True
model2 = 'WebVision_Inception_finetune_withregressionhead025_iter_460k+40000'

num_crops = 4

#Output file
output_file_dir = '../../../datasets/WebVision/results/classification_crops' + '/' + model
if ensembleClassifiers: output_file_dir = '../../../datasets/WebVision/results/classification_ensemble_crops_' + str(num_crops) + '/' + model + '_' + model2
if not os.path.exists(output_file_dir):
    os.makedirs(output_file_dir)
output_file_path = output_file_dir + '/'+split+'.txt'
output_file = open(output_file_path, "w")

# load net
net = caffe.Net('../googlenet/prototxt/deploy.prototxt', '../../../datasets/WebVision/models/saved/'+ model + '.caffemodel', caffe.TEST)
if ensembleClassifiers: net2 = caffe.Net('../googlenet/prototxt/deploy.prototxt', '../../../datasets/WebVision/models/saved/'+ model2 + '.caffemodel', caffe.TEST)

# Reshape net
batch_size = 140
size = 224
net.blobs['data'].reshape(batch_size, 3, size, size)
if ensembleClassifiers: net2.blobs['data'].reshape(batch_size, 3, size, size)


print 'Computing  ...'

count = 0
i = 0
while i < len(test):
    indices = []

    x = 0
    # Fill batch
    while (x < batch_size - num_crops):

        if i > len(test) - 1: break

        if i % 100 == 0:
            print i

        # load image
        if split == 'test':
            filename = '../../../datasets/WebVision/'+split+'_images_256/' + test[i]
        else:
            filename = '../../../datasets/WebVision/'+split+'_images_256/' + test[i][0]
        im = Image.open(filename)
        # im_o = im

        # Turn grayscale images to 3 channels
        if (im.size.__len__() == 2):
            im_gray = im
            im = Image.new("RGB", im_gray.size)
            im.paste(im_gray)

        width, height = im.size

        #Do crops
        crops = []

        #1 Crop 256x256 center and resize to 224x224
        crop_size = 256
        patch = im.copy()

        if width != crop_size:
            left = (width - crop_size) / 2
            right = width - left
            patch = patch.crop((left, 0, right, height))
        if height != crop_size:
            top = (height - crop_size) / 2
            bot = height - top
            patch = patch.crop((0, top, width, bot))
        crops.append(patch)

        # 2 Crop 224x224 center
        crop_size = 224
        patch = im.copy()
        left = (width - crop_size) / 2
        right = width - left
        top = (height - crop_size) / 2
        bot = height - top
        patch = patch.crop((left, top, right, bot))
        crops.append(patch)

        # 3 If wide image: Crop left and right 256x256 and resize to 224x224
        if width > height:
            patch = im.copy()
            left = 0
            right = 255
            patch = patch.crop((left, 0, right, height))
            crops.append(patch)

            patch = im.copy()
            left = width - 256
            right = width
            patch = patch.crop((left, 0, right, height))
            crops.append(patch)

        # 3 If vertical image: Crop top and bot 256x256 and resize to 224x224
        else: #height > width:
            patch = im.copy()
            top = 0
            bot = 255
            crop = patch.crop((0, top, width, bot))
            crops.append(patch)

            patch = im.copy()
            top = height - 256
            bot = height
            patch = patch.crop((0, top, width, bot))
            crops.append(patch)


        if len(crops) != num_crops: print "Warning, not " +str(num_crops)+ " crops for this image"

        for crop in crops:
            crop = crop.resize((224, 224), Image.ANTIALIAS)
            crop = preprocess(crop)
            net.blobs['data'].data[x,] = crop
            if ensembleClassifiers: net2.blobs['data'].data[x,] = crop
            if split == 'test':
                indices.append(test[i])
            else:
                indices.append(test[i][0]) # Each image will have 2 indices repeated
            x += 1

        i += 1

    # run net and take scores
    net.forward()
    if ensembleClassifiers: net2.forward()


    # Save results for each batch element
    x=0
    while x < len(indices):
        c=0
        probs = np.zeros(net.blobs['probs'].data[0].size)
        probs2 = np.zeros(net.blobs['probs'].data[0].size)

        while c < num_crops: # for each crop
            probs += net.blobs['probs'].data[x+c]
            if ensembleClassifiers: probs2 += net2.blobs['probs'].data[x+c]
            c+=1

            # print probs.argsort()[::-1][0:5]
            # print probs2.argsort()[::-1][0:5]

        probs = probs / num_crops
        if ensembleClassifiers: probs2 = probs2 / num_crops
        if ensembleClassifiers: probs = (probs + probs2) / 2

        top5 = probs.argsort()[::-1][0:5]
        top5str = ''

        for t in top5:
            top5str = top5str + ' ' + str(t)

        output_file.write(indices[x] + top5str + '\n')

        x+=num_crops

output_file.close()

print "DONE"
print output_file_dir

