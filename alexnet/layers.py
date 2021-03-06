import caffe

import numpy as np
from PIL import Image
from PIL import ImageOps
import time

import random



class SoftmaxSoftLabel(caffe.Layer):
    """
    Compute the Softmax Loss in the same manner but consider soft labels
    as the ground truth
    """
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 3:
            raise Exception("Need three inputs to compute distance (infered,labels and reliability).")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].num != bottom[1].num:
            raise Exception("Infered scores and labels must have the same dimension.")
        if bottom[0].num != bottom[2].num:
            raise Exception("Reliability scores wrong dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)


    # TODO PROBLEM HERE; getting exp_scores of 0 which crash in the probs. Problem is because of code or because of net?
    def forward(self, bottom, top):
        labels_scores = bottom[2].data
        labels = bottom[1].data
        scores = bottom[0].data

        #normalizing to avoid instability
        scores -= np.max(scores) # Care, should I normalize this for every img or for the whole batch?
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # correct_logprobs = -np.log(probs[range(bottom[0].num), np.array(bottom[1].data, dtype=np.uint16)])
        correct_logprobs = np.zeros([bottom[0].num,1])
        for r in range(bottom[0].num):
            correct_logprobs[r] = probs[r,int(labels[r])] * labels_scores[r]

        correct_logprobs = -np.log(correct_logprobs)
        data_loss = np.sum(correct_logprobs) / bottom[0].num

        self.diff[...] = probs
        top[0].data[...] = data_loss


    def backward(self, top, propagate_down, bottom):
        delta = self.diff
        labels = bottom[1].data
        labels_scores = bottom[2].data

        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                for r in range(bottom[0].num):
                    delta[r,int(labels[r])]-= 1 * labels_scores[r]

                #delta[range(bottom[0].num), np.array(labels, dtype=np.uint16)] -= 1

            bottom[i].diff[...] = delta / bottom[0].num




class customDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from the SBDD extended labeling
    of PASCAL VOC for semantic segmentation
    one-at-a-time while reshaping the net to preserve dimensions.

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - sbdd_dir: path to SBDD `dataset` dir
        - split: train / seg11valid
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for SBDD semantic segmentation.

        N.B.segv11alid is the set of segval11 that does not intersect with SBDD.
        Find it here: https://gist.github.com/shelhamer/edb330760338892d511e.

        example

        params = dict(sbdd_dir="/path/to/SBDD/dataset",
            mean=(104.00698793, 116.66876762, 122.67891434),
            split="valid")
        """
        # config
        params = eval(self.param_str)
        self.dir = params['dir']
        self.train = params['train']
        self.split = params['split']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.batch_size = params['batch_size']
        self.resize = params['resize']
        self.resize_w = params['resize_w']
        self.resize_h = params['resize_h']
        self.crop_w = params['crop_w']
        self.crop_h = params['crop_h']
        self.crop_margin = params['crop_margin']
        self.mirror = params['mirror']
        self.rotate = params['rotate']
        self.HSV_prob = params['HSV_prob']
        self.HSV_jitter = params['HSV_jitter']

        self.num_classes= params['num_classes']

        print "Initialiting data layer"

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and classification label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/{}.txt'.format(self.dir,self.split)

        # self.anns = open(split_f, 'r').read().splitlines()

        # get number of images
        #num_lines = 10001
        num_lines = sum(1 for line in open(split_f))
        print "Number of images: " + str(num_lines)


        # Load labels for multiclass
        self.indices = np.empty([num_lines], dtype="S50")
        self.labels = np.zeros((num_lines, 1))

        print "Reading labels file: " + '{}/{}.txt'.format(self.dir,self.split)
        with open(split_f, 'r') as annsfile:
            for c,i in enumerate(annsfile):
                data = i.split(' ')
                #Load index
                self.indices[c] = data[0]
                #Load classification labels
                self.labels[c] = int(data[1])

                if c % 10000 == 0: print "Read " + str(c) + " / " + str(num_lines)
                #if c == 10000: break

        
        print "Labels read."

        # make eval deterministic
        # if 'train' not in self.split and 'trainTrump' not in self.split:
        #     self.random = False

        self.idx = np.arange(self.batch_size)
        # randomization: seed and pick
        if self.random:
            print "Randomizing image order"
            random.seed(self.seed)
            for x in range(0,self.batch_size):
                self.idx[x] = random.randint(0, len(self.indices) - 1)
        else:
            for x in range(0, self.batch_size):
                self.idx[x] = x


        # reshape tops to fit
        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        top[0].reshape(self.batch_size, 3, params['crop_w'], params['crop_h'])
        top[1].reshape(self.batch_size, 1)


    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = np.zeros((self.batch_size, 3, self.crop_w, self.crop_h))
        self.label = np.zeros((self.batch_size, 1))

        #start = time.time()
        for x in range(0,self.batch_size):
            self.data[x,] = self.load_image(self.indices[self.idx[x]])
            self.label[x,] = self.labels[self.idx[x],]
        #end = time.time()
        #print "Time Read IMG, LABEL and dat augmentation: " + str((end-start))

    def forward(self, bottom, top):
        # assign output
        #start = time.time()

        top[0].data[...] = self.data
        top[1].data[...] = self.label

        self.idx = np.arange(self.batch_size)

        # pick next input
        if self.random:
            for x in range(0,self.batch_size):
                self.idx[x] = random.randint(0, len(self.indices) - 1)

        else:
            for x in range(0,self.batch_size):
                self.idx[x] = self.idx[x] + self.batch_size

            if self.idx[self.batch_size-1] == len(self.indices):
                for x in range(0, self.batch_size):
                    self.idx[x] = x

        #end = time.time()
        #print "Time fordward: " + str((end-start))


    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        # print '{}/img/trump/{}.jpg'.format(self.dir, idx)
        #start = time.time()
        if self.split == '/info/val_filelist':
            im = Image.open('{}/{}/{}'.format(self.dir,'val_images_256', idx))
        else:
            im = Image.open('{}/{}'.format(self.dir, idx))
        # To resize try im = scipy.misc.imresize(im, self.im_shape)
        #.resize((self.resize_w, self.resize_h), Image.ANTIALIAS) # --> No longer suing this resizing, no if below
        #end = time.time()
        #print "Time load and resize image: " + str((end - start))

        if self.resize:
            if im.size[0] != self.resize_w or im.size[1] != self.resize_h:
                im = im.resize((self.resize_w, self.resize_h), Image.ANTIALIAS)

        if( im.size.__len__() == 2):
            im_gray = im
            im = Image.new("RGB", im_gray.size)
            im.paste(im_gray)

        #start = time.time()
        if self.train: #Data Aumentation
            if(self.rotate is not 0):
                im = self.rotate_image(im)

            width, height = im.size
            if self.crop_h is not height or self.crop_h is not width:
                im = self.random_crop(im)

            if(self.mirror and random.randint(0, 1) == 1):
                im = self.mirror_image(im)

            if(self.HSV_prob is not 0):
                im = self.saturation_value_jitter_image(im)

        #end = time.time()
        #print "Time data aumentation: " + str((end - start))

        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        return in_



    #DATA AUMENTATION

    def random_crop(self,im):
        # Crops a random region of the image that will be used for training. Margin won't be included in crop.
        width, height = im.size
        margin = self.crop_margin
        left = random.randint(margin, width - self.crop_w - 1 - margin)
        top = random.randint(margin, height - self.crop_h - 1 - margin)
        im = im.crop((left, top, left + self.crop_w, top + self.crop_h))
        return im

    def mirror_image(self, im):
        return ImageOps.mirror(im)

    def rotate_image(self, im):
        return im.rotate(random.randint(-self.rotate, self.rotate))

    def saturation_value_jitter_image(self,im):
        if(random.randint(0, int(1/self.HSV_prob)) == 0):
            return im
        im = im.convert('HSV')
        data = np.array(im)  # "data" is a height x width x 3 numpy array
        data[:, :, 1] = data[:, :, 1] * random.uniform(1 - self.HSV_jitter, 1 + self.HSV_jitter)
        data[:, :, 2] = data[:, :, 2] * random.uniform(1 - self.HSV_jitter, 1 + self.HSV_jitter)
        im = Image.fromarray(data, 'HSV')
        im = im.convert('RGB')
        return im


class customDataLayerWithLabelScore(caffe.Layer):

    def setup(self, bottom, top):

        params = eval(self.param_str)
        self.dir = params['dir']
        self.train = params['train']
        self.split = params['split']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.batch_size = params['batch_size']
        self.resize = params['resize']
        self.resize_w = params['resize_w']
        self.resize_h = params['resize_h']
        self.crop_w = params['crop_w']
        self.crop_h = params['crop_h']
        self.crop_margin = params['crop_margin']
        self.mirror = params['mirror']
        self.rotate = params['rotate']
        self.HSV_prob = params['HSV_prob']
        self.HSV_jitter = params['HSV_jitter']

        self.num_classes = params['num_classes']

        print "Initialiting data layer"

        # two tops: data and label
        if len(top) != 3:
            raise Exception("Need to define three tops: data, classification label and label score.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f = '{}/{}.txt'.format(self.dir, self.split)

        # self.anns = open(split_f, 'r').read().splitlines()

        # get number of images
        # num_lines = 10001
        num_lines = sum(1 for line in open(split_f))
        print "Number of images: " + str(num_lines)

        # Load labels for multiclass
        self.indices = np.empty([num_lines], dtype="S50")
        self.labels = np.zeros((num_lines, 1), dtype="int")
        self.labels_scores = np.ones((num_lines, 1), dtype="float32") #Default labels_scores are ones

        is_training = False
        # if not self.split.__contains__('val'):
        #     is_training = True
        #     print "Is training: Will read labels scores."

        print "Reading labels file: " + '{}/{}.txt'.format(self.dir, self.split)
        with open(split_f, 'r') as annsfile:
            for c, i in enumerate(annsfile):
                data = i.split(' ')
                # Load index
                self.indices[c] = data[0]
                # Load classification labels
                self.labels[c] = int(data[1])
                # Load label scores
                if is_training:
                    self.labels_scores[c] = float(data[2])


                if c % 10000 == 0: print "Read " + str(c) + " / " + str(num_lines)
                # if c == 10000: break

        print "Labels read."

        # make eval deterministic
        # if 'train' not in self.split and 'trainTrump' not in self.split:
        #     self.random = False

        self.idx = np.arange(self.batch_size)
        # randomization: seed and pick
        if self.random:
            print "Randomizing image order"
            random.seed(self.seed)
            for x in range(0, self.batch_size):
                self.idx[x] = random.randint(0, len(self.indices) - 1)
        else:
            for x in range(0, self.batch_size):
                self.idx[x] = x

        # reshape tops to fit
        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        top[0].reshape(self.batch_size, 3, params['crop_w'], params['crop_h'])
        top[1].reshape(self.batch_size, 1)
        top[2].reshape(self.batch_size, 1)


    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = np.zeros((self.batch_size, 3, self.crop_w, self.crop_h))
        self.label = np.zeros((self.batch_size, 1))
        self.label_score = np.zeros((self.batch_size, 1))


        for x in range(0, self.batch_size):
            self.data[x,] = self.load_image(self.indices[self.idx[x]])
            self.label[x,] = self.labels[self.idx[x],]
            self.label_score[x,] = self.labels_scores[self.idx[x],]


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label
        top[2].data[...] = self.label_score


        self.idx = np.arange(self.batch_size)

        # pick next input
        if self.random:
            for x in range(0, self.batch_size):
                self.idx[x] = random.randint(0, len(self.indices) - 1)

        else:
            for x in range(0, self.batch_size):
                self.idx[x] = self.idx[x] + self.batch_size

            if self.idx[self.batch_size - 1] == len(self.indices):
                for x in range(0, self.batch_size):
                    self.idx[x] = x

    def backward(self, top, propagate_down, bottom):
        pass

    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        # print '{}/img/trump/{}.jpg'.format(self.dir, idx)
        # start = time.time()
        if self.split == '/info/val_filelist':
            im = Image.open('{}/{}/{}'.format(self.dir, 'val_images_256', idx))
        else:
            im = Image.open('{}/{}'.format(self.dir, idx))
        # To resize try im = scipy.misc.imresize(im, self.im_shape)
        # .resize((self.resize_w, self.resize_h), Image.ANTIALIAS) # --> No longer suing this resizing, no if below
        # end = time.time()
        # print "Time load and resize image: " + str((end - start))

        if self.resize:
            if im.size[0] != self.resize_w or im.size[1] != self.resize_h:
                im = im.resize((self.resize_w, self.resize_h), Image.ANTIALIAS)

        if (im.size.__len__() == 2):
            im_gray = im
            im = Image.new("RGB", im_gray.size)
            im.paste(im_gray)

        # start = time.time()
        if self.train:  # Data Aumentation
            if (self.rotate is not 0):
                im = self.rotate_image(im)

            width, height = im.size
            if self.crop_h is not height or self.crop_h is not width:
                im = self.random_crop(im)

            if (self.mirror and random.randint(0, 1) == 1):
                im = self.mirror_image(im)

            if (self.HSV_prob is not 0):
                im = self.saturation_value_jitter_image(im)

        # end = time.time()
        # print "Time data aumentation: " + str((end - start))

        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:, :, ::-1]
        in_ -= self.mean
        in_ = in_.transpose((2, 0, 1))
        return in_

    # DATA AUMENTATION

    def random_crop(self, im):
        # Crops a random region of the image that will be used for training. Margin won't be included in crop.
        width, height = im.size
        margin = self.crop_margin
        left = random.randint(margin, width - self.crop_w - 1 - margin)
        top = random.randint(margin, height - self.crop_h - 1 - margin)
        im = im.crop((left, top, left + self.crop_w, top + self.crop_h))
        return im

    def mirror_image(self, im):
        return ImageOps.mirror(im)

    def rotate_image(self, im):
        return im.rotate(random.randint(-self.rotate, self.rotate))

    def saturation_value_jitter_image(self, im):
        if (random.randint(0, int(1 / self.HSV_prob)) == 0):
            return im
        im = im.convert('HSV')
        data = np.array(im)  # "data" is a height x width x 3 numpy array
        data[:, :, 1] = data[:, :, 1] * random.uniform(1 - self.HSV_jitter, 1 + self.HSV_jitter)
        data[:, :, 2] = data[:, :, 2] * random.uniform(1 - self.HSV_jitter, 1 + self.HSV_jitter)
        im = Image.fromarray(data, 'HSV')
        im = im.convert('RGB')
        return im
