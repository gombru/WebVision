import caffe
import tempfile
import numpy as np
import os
from pylab import zeros, arange, subplots, plt, savefig


caffe.set_device(0)
caffe.set_mode_gpu()

training_id = 'WebVision_Inception_LDAfiltered_500_80000chunck' # name to save the training plots

weights = '../../../datasets/WebVision/models/saved/WebVision_2head_Inception_500_80000chunck_iter_60000.caffemodel'
assert os.path.exists(weights)

display_interval = 500
niter = 100011100

#number of validating images  is  test_iters * batchSize
test_interval = 5000 #200
test_iters = 100 #20
solver_filename = 'prototxt/solver.prototxt'
solver = caffe.get_solver(solver_filename)

#Copy init weights
solver.net.copy_from(weights)

#Restore solverstate
#solver.restore('models/IIT5K/cifar10/IIT5K_iter_15000.caffemodel')



def do_solve(maxIter, solver, display, test_interval, test_iters):

    # SET PLOTS DATA
    train_loss_C = zeros(maxIter/display)
    train_top1 = zeros(maxIter/display)
    train_top5 = zeros(maxIter/display)

    val_loss_C = zeros(maxIter/test_interval)
    val_top1 = zeros(maxIter/test_interval)

    it_axes = (arange(maxIter) * display) + display
    it_val_axes = (arange(maxIter) * test_interval) + test_interval

    _, ax1 = subplots()
    ax2 = ax1.twinx()
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('train loss C (r), val loss C (y)')
    ax2.set_ylabel('train TOP1 (b), val TOP1 (g), train TOP-5 (2) (c)')
    ax2.set_autoscaley_on(False)
    ax2.set_ylim([0, 1])

    lossC = np.zeros(maxIter)
    acc1 = np.zeros(maxIter)
    acc5 = np.zeros(maxIter)


    #RUN TRAINING
    for it in range(niter):
        solver.step(1)  # run a single SGD step in Caffepy()

        #PLOT
        if it % display == 0 or it + 1 == niter:
            lossC[it] = solver.net.blobs['loss3/loss3'].data.copy()
            acc1[it] = solver.net.blobs['loss3/top-1'].data.copy()
            acc5[it] = solver.net.blobs['loss2/top-5'].data.copy()

            loss_disp = 'loss3C= ' + str(lossC[it]) +  '  top-1= ' + str(acc1[it])

            print '%3d) %s' % (it, loss_disp)

            train_loss_C[it / display] = lossC[it]
            train_top1[it / display] = acc1[it]
            train_top5[it / display] = acc5[it]

            ax1.plot(it_axes[0:it / display], train_loss_C[0:it / display], 'r')
            ax2.plot(it_axes[0:it / display], train_top1[0:it / display], 'b')
            ax2.plot(it_axes[0:it / display], train_top5[0:it / display], 'c')

            ax1.set_ylim([0, 10])
            plt.title(training_id)
            plt.ion()
            plt.grid(True)
            plt.show()
            plt.pause(0.001)

        #VALIDATE
        if it % test_interval == 0 and it > 0:
            loss_val_C = 0
            top1_val = 0
            for i in range(test_iters):
                solver.test_nets[0].forward()
                loss_val_C += solver.test_nets[0].blobs['loss3/loss3'].data
                top1_val += solver.test_nets[0].blobs['loss3/top-1'].data

            loss_val_C /= test_iters
            top1_val /= test_iters

            print("Val loss C: {:.3f}".format(loss_val_C))

            val_loss_C[it / test_interval - 1] = loss_val_C
            val_top1[it / test_interval - 1] = top1_val

            ax1.plot(it_val_axes[0:it / test_interval], val_loss_C[0:it/ test_interval], 'y')
            ax2.plot(it_val_axes[0:it / test_interval], val_top1[0:it / test_interval], 'g')

            ax1.set_ylim([0, 10])
            plt.title(training_id)
            plt.ion()
            plt.grid(True)
            plt.show()
            plt.pause(0.001)
            title = '../../../datasets/WebVision/models/training/' + training_id + str(it) + '.png'  # Save graph to disk
            savefig(title, bbox_inches='tight')

    return



print 'Running solvers for %d iterations...' % niter
_, _, _ = do_solve(niter, solver, display_interval, test_interval, test_iters)
print 'Done.'