import caffe
import numpy as np

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
        scores = bottom[0].data # .astype(np.float128)
        print "Scores MAX: " + str(scores.max())
        print "Scores MIN: " + str(scores.max())

        #normalizing to avoid instability
        scores -= np.max(scores) # Care, should I normalize this for every img or for the whole batch?
        # for s in range(0,len(scores)):
        #     scores[s,:] -= np.max(scores[s,:])
        exp_scores = np.exp(scores)
        if exp_scores.min() == 0:
            print "WARNING, Exp Score is 0"
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        logprob = -np.log(probs)
        data_loss = np.sum(np.sum(labels_scores*logprob,axis=1))/bottom[0].num

        self.diff[...] = probs
        top[0].data[...] = data_loss

    def backward(self, top, propagate_down, bottom):
        delta = self.diff
        labels_scores = bottom[2].data
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i==0:
                delta[range(bottom[0].num), np.array(bottom[1].data,dtype=np.uint16)] -= 1
                delta = delta*labels_scores

            bottom[i].diff[...] = delta/bottom[0].num