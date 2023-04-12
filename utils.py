from logger import logger
import concurrent.futures as cf
import math
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import torchvision as tv
from time import time
import os
import re
from skimage import color
from vidgear.gears import CamGear
import asyncio as aio
import itertools
import sys
import os
# add path to parent directory in sys.path list
sys.path.append('C:\\Users\\shash\\Desktop\\btp\\main\\model\\')
c, loaded_model, model = None, None, None


def set_models(c_, loaded_model_, model_):
    global c, loaded_model, model
    c = c_
    loaded_model = loaded_model_
    model = model_
    logger.log('model set')


def get_file_path_list(indir):
    """
    It reads filname containing input pattern recursively if indir exists
    """

    assert os.path.exists(indir), 'indir is not exits.'

    img_file_list = os.listdir(indir)
    img_file_list = sorted(img_file_list,
                           key=lambda k: int(re.match(r'(\d+)', k).group()))
    img_list = []
    for i, img in enumerate(img_file_list):
        if '.png' in img:
            path_ = os.path.join(indir, img)
            img_list.append(path_)
    return img_list


def color_hist(im, col_bins):
    """
    Get color histogram descriptors for RGB and LAB space.
    Input: im: (h,w,c): 0-255: np.uint8
    Output: descriptor: (col_bins*6,)
    """
    assert im.ndim == 3 and im.shape[2] == 3, "image should be rgb"
    arr = np.concatenate((im, color.rgb2lab(im)), axis=2).reshape((-1, 6))
    desc = np.zeros((col_bins * 6,), dtype=np.float)
    for i in range(3):
        desc[i * col_bins:(i + 1) * col_bins], _ = np.histogram(
            arr[:, i], bins=col_bins, range=(0, 255))
        desc[i * col_bins:(i + 1) * col_bins] /= np.sum(
            desc[i * col_bins:(i + 1) * col_bins]) + (
                np.sum(desc[i * col_bins:(i + 1) * col_bins]) < 1e-4)

    # noinspection PyUnboundLocalVariable
    i += 1
    desc[i * col_bins:(i + 1) * col_bins], _ = np.histogram(
        arr[:, i], bins=col_bins, range=(0, 100))
    desc[i * col_bins:(i + 1) * col_bins] /= np.sum(
        desc[i * col_bins:(i + 1) * col_bins]) + (
            np.sum(desc[i * col_bins:(i + 1) * col_bins]) < 1e-4)
    for i in range(4, 6):
        desc[i * col_bins:(i + 1) * col_bins], _ = np.histogram(
            arr[:, i], bins=col_bins, range=(-128, 127))
        desc[i * col_bins:(i + 1) * col_bins] /= np.sum(
            desc[i * col_bins:(i + 1) * col_bins]) + (
                np.sum(desc[i * col_bins:(i + 1) * col_bins]) < 1e-4)
    return desc


def compute_features(im, col_bins):
    """
    Compute features of images: RGB histogram + SIFT
    im: (h,w,c): 0-255: np.uint8
    feat: (d,)
    """
    col_hist = color_hist(im, col_bins=col_bins)

    return col_hist


def calc_scatters(K):
    """
    Calculate scatter matrix:
    scatters[i,j] = {scatter of the sequence with starting frame i and ending frame j}
    """
    n = K.shape[0]
    K1 = np.cumsum([0] + list(np.diag(K)))
    K2 = np.zeros((n+1, n+1))
    # TODO: use the fact that K - symmetric
    K2[1:, 1:] = np.cumsum(np.cumsum(K, 0), 1)

    scatters = np.zeros((n, n))

    diagK2 = np.diag(K2)

    i = np.arange(n).reshape((-1, 1))
    j = np.arange(n).reshape((1, -1))
    scatters = (K1[1:].reshape((1, -1))
                - K1[:-1].reshape((-1, 1))
                - (diagK2[1:].reshape((1, -1)) + diagK2[:-
                   1].reshape((-1, 1)) - K2[1:, :-1].T - K2[:-1, 1:])
                / ((j-i+1).astype(float) + (j == i-1).astype(float)))
    scatters[j < i] = 0
    # code = r"""
    # for (int i = 0; i < n; i++) {
    #    for (int j = i; j < n; j++) {
    #        scatters(i,j) = K1(j+1)-K1(i) - (K2(j+1,j+1)+K2(i,i)-K2(j+1,i)-K2(i,j+1))/(j-i+1);
    #    }
    # }
    # """
    # weave.inline(code, ['K1','K2','scatters','n'], global_dict = \
    #    {'K1':K1, 'K2':K2, 'scatters':scatters, 'n':n}, type_converters=weave.converters.blitz)

    return scatters


def cpd_nonlin(K, ncp, lmin=1, lmax=100000, backtrack=True, verbose=True,
               out_scatters=None):
    """ Change point detection with dynamic programming
    K - square kernel matrix
    ncp - number of change points to detect (ncp >= 0)
    lmin - minimal length of a segment
    lmax - maximal length of a segment
    backtrack - when False - only evaluate objective scores (to save memory)

    Returns: (cps, obj)
        cps - detected array of change points: mean is thought to be constant on [ cps[i], cps[i+1] )
        obj_vals - values of the objective function for 0..m changepoints

    """
    m = int(ncp)  # prevent numpy.int64

    (n, n1) = K.shape
    assert (n == n1), "Kernel matrix awaited."

    assert (n >= (m + 1)*lmin)
    assert (n <= (m + 1)*lmax)
    assert (lmax >= lmin >= 1)

    if verbose:
        # print "n =", n
        logger.log("Precomputing scatters...")
    J = calc_scatters(K)

    if out_scatters != None:
        out_scatters[0] = J

    if verbose:
        logger.log("Inferring best change points...")
    # I[k, l] - value of the objective for k change-points and l first frames
    I = 1e101*np.ones((m+1, n+1))
    I[0, lmin:lmax] = J[0, lmin-1:lmax-1]

    if backtrack:
        # p[k, l] --- "previous change" --- best t[k] when t[k+1] equals l
        p = np.zeros((m+1, n+1), dtype=int)
    else:
        p = np.zeros((1, 1), dtype=int)

    for k in range(1, m+1):
        for l in range((k+1)*lmin, n+1):
            tmin = max(k*lmin, l-lmax)
            tmax = l-lmin+1
            c = J[tmin:tmax, l-1].reshape(-1) + I[k-1, tmin:tmax].reshape(-1)
            I[k, l] = np.min(c)
            if backtrack:
                p[k, l] = np.argmin(c)+tmin

    # code = r"""
    # define max(x,y) ((x)>(y)?(x):(y))
    # for (int k=1; k<m+1; k++) {
    #    for (int l=(k+1)*lmin; l<n+1; l++) {
    #        I(k, l) = 1e100; //nearly infinity
    #        for (int t=max(k*lmin,l-lmax); t<l-lmin+1; t++) {
    #            double c = I(k-1, t) + J(t, l-1);
    #            if (c < I(k, l)) {
    #                I(k, l) = c;
    #                if (backtrack == 1) {
    #                    p(k, l) = t;
    #                }
    #            }
    #        }
    #    }
    # }
    # """

    # weave.inline(code, ['m','n','p','I', 'J', 'lmin', 'lmax', 'backtrack'], \
    #    global_dict={'m':m, 'n':n, 'p':p, 'I':I, 'J':J, \
    #    'lmin':lmin, 'lmax':lmax, 'backtrack': int(1) if backtrack else int(0)},
    #    type_converters=weave.converters.blitz)

    # Collect change points
    cps = np.zeros(m, dtype=int)

    if backtrack:
        cur = n
        for k in range(m, 0, -1):
            cps[k-1] = p[k, cur]
            cur = cps[k-1]

    scores = I[:, n].copy()
    scores[scores > 1e99] = np.inf
    return cps, scores


def cpd_auto(K, ncp, vmax, desc_rate=1, **kwargs):
    """Main interface

    Detect change points automatically selecting their number
        K       - kernel between each pair of frames in video
        ncp     - maximum ncp
        vmax    - special parameter
    Optional arguments:
        lmin     - minimum segment length
        lmax     - maximum segment length
        desc_rate - rate of descriptor sampling (vmax always corresponds to 1x)

    Note:
        - cps are always calculated in subsampled coordinates irrespective to
            desc_rate
        - lmin and m should be in agreement
    ---
    Returns: (cps, costs)
        cps   - best selected change-points
        costs - costs for 0,1,2,...,m change-points

    Memory requirement: ~ (3*N*N + N*ncp)*4 bytes ~= 16 * N^2 bytes
    That is 1,6 Gb for the N=10000.
    """
    m = ncp
    (_, scores) = cpd_nonlin(K, m, backtrack=False, **kwargs)

    N = K.shape[0]
    N2 = N*desc_rate  # length of the video before subsampling

    penalties = np.zeros(m+1)
    # Prevent division by zero (in case of 0 changes)
    ncp = np.arange(1, m+1)
    penalties[1:] = (vmax*ncp/(2.0*N2))*(np.log(float(N2)/ncp)+1)

    costs = scores/float(N) + penalties
    m_best = np.argmin(costs)
    (cps, scores2) = cpd_nonlin(K, m_best, **kwargs)

    return (cps, scores2)


# ------------------------------------------------------------------------------
# Extra functions (currently not used)

def estimate_vmax(K_stable):
    """K_stable - kernel between all frames of a stable segment"""
    n = K_stable.shape[0]
    vmax = np.trace(centering(K_stable)/n)
    return vmax


def centering(K):
    """Apply kernel centering"""
    mean_rows = np.mean(K, 1)[:, np.newaxis]
    return K - mean_rows - mean_rows.T + np.mean(mean_rows)


def eval_score(K, cps):
    """ Evaluate unnormalized empirical score
        (sum of kernelized scatters) for the given change-points """
    N = K.shape[0]
    cps = [0] + list(cps) + [N]
    V1 = 0
    V2 = 0
    for i in range(len(cps)-1):
        K_sub = K[cps[i]:cps[i+1], :][:, cps[i]:cps[i+1]]
        V1 += np.sum(np.diag(K_sub))
        V2 += np.sum(K_sub) / float(cps[i+1] - cps[i])
    return (V1 - V2)


def eval_cost(k, cps, score, vmax):
    """ Evaluate cost function for automatic number of change points selection
    K      - kernel between all frames
    cps    - selected change-points
    score  - unnormalized empirical score (sum of kernelized scatters)
    vmax   - vmax parameter"""

    n = k.shape[0]
    penalty = (vmax*len(cps)/(2.0*n))*(np.log(float(n)/len(cps))+1)
    return score/float(n) + penalty


def get_frames(filename, n_frames=-1):
    frames = []
    actual_frames = []
    v_cap = cv2.VideoCapture(filename)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames == -1:
        n_frames = v_len
    else:
        n_frames = min(n_frames, v_len)

    frame_list = np.linspace(0, v_len-1, n_frames+1, dtype=np.int16)
    for fn in range(v_len):
        success, frame = v_cap.read()
        if success is False:
            continue
        if (fn in frame_list):
            actual_frames.append(frame)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frames.append(frame)
    v_cap.release()
    return frames, v_len, actual_frames


def get_frames_with_sound(filename, n_frames=-1):
    actual_frames = []
    v_cap = cv2.VideoCapture(filename)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames == -1:
        n_frames = v_len
    else:
        n_frames = min(n_frames, v_len)
    audio_index = [0]
    frame_list = np.linspace(0, v_len-1, n_frames+1, dtype=np.int16)
    for fn in range(v_len):
        success, frame = v_cap.read()
        if success is False:
            continue
        if (fn in frame_list):
            actual_frames.append(frame)
            cur_time = int(v_cap.get(cv2.CAP_PROP_POS_MSEC))
            audio_index.append(cur_time)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frames.append(frame)
    v_cap.release()
    cv2.destroyAllWindows()
    return actual_frames, audio_index


def get_frames_videogear(video_path):
    stream = CamGear(source=video_path).start()
    frames = []
    while True:
        # read frames from stream
        frame = stream.read()
        # check for frame if Nonetype
        if frame is None:
            break
        frames.append(frame)
    # safely close video stream
    stream.stop()
    return frames


def extractFeatures(input_image):

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    # create a mini-batch as expected by the model
    input_batch = input_tensor.unsqueeze(0)

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        global model
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
        output_2darray = output.cpu().numpy()  # changed here

        final_output = np. reshape(output_2darray, -1, order='F')

        return final_output


async def extract_range(frames, start, end):
    features = []
    im = None
    for frame in frames[start:end]:
        im = Image.fromarray(frame)

        frame_feature = extractFeatures(im)
        features.append(frame_feature)

    return features


def extract_range_sync(frames, start, end):
    features = []
    im = None
    for frame in frames[start:end]:
        im = Image.fromarray(frame)

        frame_feature = extractFeatures(im)
        features.append(frame_feature)
    return features


async def get_features_fast(frames):
    no_of_threads = 10
    mxx = math.ceil(len(frames) / no_of_threads)
    coros = [0 for i in range(no_of_threads)]
    loop = aio.get_event_loop()
    v_len = len(frames)
    for i in range(no_of_threads):
        coros[i] = loop.create_task(extract_range(
            frames, i * mxx, min(v_len, (i+1) * mxx)), name=str(i))
    results = await aio.wait(coros)
    results = list(results[0])
    results.sort(key=lambda x: int(x.get_name()))
    return list(itertools.chain(*[r.result() for r in results]))
# ->223


def get_features_thread(frames):
    no_of_threads = 20
    v_len = len(frames)
    mxx = math.ceil(len(frames) / no_of_threads)
    coros = [0 for i in range(no_of_threads)]
    ret = []
    with cf.ThreadPoolExecutor() as executor:
        for i in range(no_of_threads):
            logger.log(f'running thread {i}')
            coros[i] = executor.submit(
                extract_range_sync, frames, i * mxx, min(v_len, (i+1) * mxx))
        # cf.wait(coros)
        logger.log('collecting result')
        ret = list(itertools.chain(*[coro.result() for coro in coros]))
        executor.shutdown(wait=True, cancel_futures=True)

    logger.log(len(ret))
    return ret
# 20->123
# 50->124
# 10->143
# 25->131

# modified


def get_video_shots(features, max_shots=5, vmax=0.6, col_bins=40):
    """
    Convert a given video into number of shots
    img_seq: (n,h,w,c): 0-255: np.uint8: RGB
    shot_idx: (k,): start Index of shot: 0-indexed
    shotScore: (k,): First change ../lib/kts/cpd_auto.py return value to
                     scores2 instead of costs (a bug)
    """
    x = features.cpu().numpy()

    k = np.dot(x, x.T)
    shot_idx, _ = cpd_auto(k, max_shots - 1, vmax)
    shot_idx = np.concatenate(([0], shot_idx))
    return shot_idx


def knapSack(W, wt, val, n):
    """ Maximize the value that a knapsack of capacity W can hold. You can either put the item or discard it, there is
    no concept of putting some part of item in the knapsack.

    :param int W: Maximum capacity -in frames- of the knapsack.
    :param list[int] wt: The weights (lengths -in frames-) of each video shot.
    :param list[float] val: The values (importance scores) of each video shot.
    :param int n: The number of the shots.
    :return: A list containing the indices of the selected shots.
    """
    K = [[0 for _ in range(W + 1)] for _ in range(n + 1)]

    # Build table K[][] in bottom up manner
    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif wt[i - 1] <= w:
                K[i][w] = max(val[i - 1] + K[i - 1]
                              [w - wt[i - 1]], K[i - 1][w])
            else:
                K[i][w] = K[i - 1][w]

    selected = []
    w = W
    for i in range(n, 0, -1):
        if K[i][w] != K[i - 1][w]:
            selected.insert(0, i - 1)
            w -= wt[i - 1]

    return selected
