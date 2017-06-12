
import sys
import caffe
from create_net_AlexNet import build_AlexNet

from create_solver import create_solver
from do_solve import do_solve
import os


caffe.set_device(0)
caffe.set_mode_gpu()

weights = '../../../datasets/SocialMedia/models/pretrained/bvlc_reference_caffenet.caffemodel'
assert os.path.exists(weights)


split_train = '/info/train_filelist_mini'
split_val = '/info/val_filelist'
num_labels = 1000
batch_size = 40 #100
resize_w = 300
resize_h = 300
crop_w = 227 #227 AlexNet, 224 VGG16
crop_h = 227
crop_margin = 10 #The crop won't include the margin in pixels
mirror = True #Mirror images with 50% prob
rotate = 0 #15,8 #Always rotate with angle between -a and a
HSV_prob = 0 #0.3,0.15 #Jitter saturation and vale of the image with this prob
HSV_jitter = 0 #0.1,0.05 #Saturation and value will be multiplied by 2 different random values between 1 +/- jitter


# #Create the net architecture
# net_train = build_AlexNet(split_train, num_labels, batch_size, resize_w, resize_h, crop_w, crop_h, crop_margin, mirror, rotate, HSV_prob, HSV_jitter, train=True)
# #Prepare validation net
# net_val = build_AlexNet(split_val, num_labels, batch_size, crop_w, crop_h, crop_h, crop_h, 0, 0, 0, 0, 0, train=False)

niter = 1000111
base_lr = 0.001 #VGG  #AlexNet 0.0001
display_interval = 1

#number of validating images  is  test_iters * batchSize
test_interval = 400 #200
test_iters = 20

# #Set solver configuration
# solver_filename = create_solver(net_train, net_val, base_lr=base_lr)
# # Load solver
# solver = caffe.get_solver(solver_filename)
solver = caffe.get_solver('solver.prototxt')

#Copy init weights
solver.net.copy_from(weights)

#Restore solverstate
#solver.restore('models/IIT5K/cifar10/IIT5K_iter_15000.caffemodel')


print 'Running solvers for %d iterations...' % niter
solvers = [('my_solver', solver)]
_, _, _ = do_solve(niter, solvers, display_interval, test_interval, test_iters)
print 'Done.'