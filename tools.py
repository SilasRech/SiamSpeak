import random
import numpy as np
import numpy.linalg as nl
from scipy import interpolate
from scipy.spatial.distance import pdist, cdist, squareform
import tensorflow as tf
import math


def get_input_shape(dataset, phase='train'):
    example = dataset.take(1)
    if phase == 'train':
        for spec1, spec2, y in example.as_numpy_iterator():
            input_shape = spec1.shape
    else:
        for spec1 in example.as_numpy_iterator():
            input_shape = spec1.shape

    return input_shape


def EER(clients, impostors):
    threshold = np.arange(-1, 1, 0.1) #[-0.5,-0.4,-0.3,-0.2,-0.1,0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    FAR = []
    FRR = []
    for th in threshold:
        far = 0.0
        for score in impostors:
            if score.item() > float(th):
                far += 1
        frr = 0.0
        for score in clients:
            if score.item() <= float(th):
                frr += 1
        FAR.append(far/impostors.size)
        FRR.append(frr/clients.size)
    ERR = 0.
    dist = 1.
    for far, frr in zip(FAR, FRR):
        if abs(far-frr) < dist:
            ERR = (far+frr)/2
            dist = abs(far-frr)
    return float("{0:.3f}".format(100*ERR))


def calculate_EER(scores):

    #lst = open(filename,'r')

    #scores = []
    #lines = lst.readlines()
    #for x in lines:
    #    scores.append(np.float(x.split()[0]))

    c = np.array(scores[0::2])
    i = np.array(scores[1::2])

    print('EER is : %.3f' % EER(c, i))


def read_trials(list):
    name1 = []
    name2 = []
    label = []
    lines = list.readlines()
    for x in lines:
        name1.append(x.split()[0])
        name2.append(x.split()[1])
        label.append(x.split()[2])
    return name1, name2, label


def evaluate_model(model, dataset):
    prediction = []
    label = []
    for x1, y in dataset.as_numpy_iterator():
        test1 = np.argmax(model.predict(x1), axis=1)

        prediction.append(test1)
        label.append((np.argmax(y, axis=1)))

    prediction = np.asarray(prediction).flatten()
    label = np.asarray(label).flatten()

    accuracy = np.mean(y_pred == label) * 100
    print('Classfication Accuracy is {}'.format(accuracy))

    return prediction, label

def mel_to_freq(m):

    return 700*(math.exp((m/1127))-1)

def makeT(cp):
    # cp: [K x 2] control points
    # T: [(K+3) x (K+3)]
    K = cp.shape[0]
    T = np.zeros((K + 3, K + 3))
    T[:K, 0] = 1
    T[:K, 1:3] = cp
    T[K, 3:] = 1
    T[K + 1:, 3:] = cp.T
    R = squareform(pdist(cp, metric='euclidean'))
    R = R * R
    R[R == 0] = 1  # a trick to make R ln(R) 0
    R = R * np.log(R)
    np.fill_diagonal(R, 0)
    T[:K, 3:] = R
    return T


def read_trials(list):
    name1 = []
    name2 = []
    label = []
    lines = list.readlines()
    for x in lines:
        name1.append(x.split()[0])
        name2.append(x.split()[1])
        label.append(x.split()[2])
    return name1, name2, label


def liftPts(p, cp):
    # p: [N x 2], input points
    # cp: [K x 2], control points
    # pLift: [N x (3+K)], lifted input points
    N, K = p.shape[0], cp.shape[0]
    pLift = np.zeros((N, K + 3))
    pLift[:, 0] = 1
    pLift[:, 1:3] = p
    R = cdist(p, cp, 'euclidean')
    R = R * R
    R[R == 0] = 1
    R = R * np.log(R)
    pLift[:, 3:] = R
    return pLift


def spec_augment(spec, param):

    W = param
    T = param
    F = 13
    mt = 2
    mf = 2

    # Nframe : number of spectrum frame
    Nframe = spec.shape[1]
    # Nbin : number of spectrum freq bin
    Nbin = spec.shape[0]
    # check input length
    if Nframe < W * 2 + 1:
        W = int(Nframe / 4)
    if Nframe < T * 2 + 1:
        T = int(Nframe / mt)
    if Nbin < F * 2 + 1:
        F = int(Nbin / mf)

    # warping parameter initialize
    w = random.randint(-W, W)
    center = random.randint(W, Nframe - W)

    src = np.asarray(
        [[float(center), 1], [float(center), 0], [float(center), 2], [0, 0], [0, 1], [0, 2], [Nframe - 1, 0],
         [Nframe - 1, 1], [Nframe - 1, 2]])
    dst = np.asarray(
        [[float(center + w), 1], [float(center + w), 0], [float(center + w), 2], [0, 0], [0, 1], [0, 2],
         [Nframe - 1, 0], [Nframe - 1, 1], [Nframe - 1, 2]])
    # print(src,dst)

    # source control points
    xs, ys = src[:, 0], src[:, 1]
    cps = np.vstack([xs, ys]).T
    # target control points
    xt, yt = dst[:, 0], dst[:, 1]
    # construct TT
    TT = makeT(cps)

    # solve cx, cy (coefficients for x and y)
    xtAug = np.concatenate([xt, np.zeros(3)])
    ytAug = np.concatenate([yt, np.zeros(3)])
    cx = nl.solve(TT, xtAug)  # [K+3]
    cy = nl.solve(TT, ytAug)

    # dense grid
    x = np.linspace(0, Nframe - 1, Nframe)
    y = np.linspace(1, 1, 1)
    x, y = np.meshgrid(x, y)

    xgs, ygs = x.flatten(), y.flatten()

    gps = np.vstack([xgs, ygs]).T

    # transform
    pgLift = liftPts(gps, cps)  # [N x (K+3)]
    xgt = np.dot(pgLift, cx.T)
    spec_warped = np.zeros_like(spec)
    for f_ind in range(Nbin):
        spec_tmp = spec[f_ind, :]
        func = interpolate.interp1d(xgt, spec_tmp, fill_value="extrapolate")
        xnew = np.linspace(0, Nframe - 1, Nframe)
        spec_warped[f_ind, :] = func(xnew)

    # sample mt of time mask ranges
    t = np.random.randint(T - 1, size=mt) + 1
    # sample mf of freq mask ranges
    f = np.random.randint(F - 1, size=mf) + 1
    # mask_t : time mask vector
    mask_t = np.ones((Nframe, 1))
    ind = 0
    t_tmp = t.sum() + mt
    for _t in t:
        k = random.randint(ind, Nframe - t_tmp)
        mask_t[k:k + _t] = 0
        ind = k + _t + 1
        t_tmp = t_tmp - (_t + 1)
    mask_t[ind:] = 1

    # mask_f : freq mask vector
    mask_f = np.ones((Nbin, 1))
    ind = 0
    f_tmp = f.sum() + mf
    for _f in f:
        k = random.randint(ind, Nbin - f_tmp)
        mask_f[k:k + _f] = 0
        ind = k + _f + 1
        f_tmp = f_tmp - (_f + 1)
    mask_f[ind:] = 1

    # calculate mean
    mean = np.mean(spec_warped)

    # make spectrum to zero mean
    spec_zero = spec_warped - mean

    spec_masked = ((spec_zero * mask_t.T) * mask_f) + mean
    # spec_masked = ((spec_zero * mask_t).T * mask_f).T

    return spec_masked


def power_to_db(S, amin=1e-16, top_db=80.0):
    """Convert a power-spectrogram (magnitude squared) to decibel (dB) units.
    Computes the scaling ``10 * log10(S / max(S))`` in a numerically
    stable way.
    Based on:
    https://librosa.github.io/librosa/generated/librosa.core.power_to_db.html
    """

    def _tf_log10(x):
        numerator = tf.math.log(x)
        denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator

    # Scale magnitude relative to maximum value in S. Zeros in the output
    # correspond to positions where S == ref.
    ref = tf.reduce_max(S)

    log_spec = 10.0 * _tf_log10(tf.maximum(amin, S))
    log_spec -= 10.0 * _tf_log10(tf.maximum(amin, ref))

    log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)

    return log_spec