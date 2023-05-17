import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import precision_recall_curve, accuracy_score
import pandas as pd
import json
import yaml
import os
import scipy
import time
import cv2
from math import sqrt

from angel_system.ptg_eval.common.load_data import time_from_name
from angel_system.ptg_eval.common.load_data import activities_from_dive_csv

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter, \
        AutoMinorLocator
    HAS_MATLOTLIB = True
except ModuleNotFoundError:
    HAS_MATLOTLIB = False


def accuracy_from_true_false_examples(true_example_scores,
                                      false_example_scores):
    s = np.hstack([true_example_scores, false_example_scores]).T
    y_tue = np.hstack([np.ones(len(true_example_scores), dtype=bool),
                       np.zeros(len(false_example_scores), dtype=bool)]).T
    s.shape = (-1, 1)
    y_tue.shape = (-1, 1)
    return accuracy_score(y_tue, s)


def plot_precision_recall(true_example_scores, false_example_scores,
                          label=None, title_str=None):
    fig = plt.figure(num=None, figsize=(14, 10), dpi=80)
    plt.rc('font', **{'size': 22})
    plt.rc('axes', linewidth=4)

    s = np.hstack([true_example_scores, false_example_scores]).T
    y_tue = np.hstack([np.ones(len(true_example_scores), dtype=bool),
                       np.zeros(len(false_example_scores), dtype=bool)]).T
    s.shape = (-1, 1)
    y_tue.shape = (-1, 1)
    precision, recall, thresholds = precision_recall_curve(y_tue, s)
    thresholds = np.hstack([thresholds[0], thresholds])
    auc = -np.trapz(precision, recall)

    plt.plot(recall, precision, linewidth=6, label=label)

    plt.xlabel('Recall', fontsize=40)
    plt.ylabel('Precision', fontsize=40)

    if title_str is not None:
        plt.title(title_str, fontsize=40)

    plt.xlim([0, 1.01])
    plt.ylim([0, 1.01])
    fig.tight_layout()

    plt.gca().yaxis.set_minor_locator(AutoMinorLocator(2))
    plt.tick_params(
            axis="y",
            which="major",
            grid_color='lightgrey')
    plt.tick_params(
            axis="y",
            which="minor",
            grid_linestyle='--',
            grid_color='lightgrey')
    plt.grid(axis='y', which='both')
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(2))
    plt.tick_params(
            axis="x",
            which="major",
            grid_color='lightgrey')
    plt.tick_params(
            axis="x",
            which="minor",
            grid_linestyle='--',
            grid_color='lightgrey')
    plt.grid(axis='x', which='both')

    if label is not None:
        plt.legend(fontsize=20, loc=0)


def save_matrix_image(mat, fname, min_w=100, max_w=8000, aspect_ratio=4,
                      first_ind=0, col_labels=False,
                      colormap=cv2.COLORMAP_JET):
    """Save image of num_class x num_results matrix.

    first_ind : int
        Integer of the first row to start counting from.
    """
    assert min_w <= max_w

    h, w = mat.shape
    num_class = h

    if w > max_w:
        w = max_w
    elif w < min_w:
        w = min_w

    h2 = int(w/aspect_ratio)
    th0 = 0.6*h2/num_class

    def err(s):
        font_scale = s
        font_thickness = 2

        th = cv2.getTextSize(str(num_class+1), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                             fontScale=font_scale, thickness=font_thickness)[0][1]
        return np.abs(th - th0)

    s = np.logspace(-3, 2, 1000)
    er = [err(s_) for s_ in s]
    s = s[np.argmin(er)]


    font_scale = s*1.25
    font_thickness = sqrt(s)*2

    font_scale = max([font_scale, 0.25])
    font_thickness = int(max([font_thickness, 1]))

    out = cv2.resize(mat, (w, h2), interpolation=cv2.INTER_NEAREST)
    out = np.round(out*255).astype(np.uint8)
    out = cv2.applyColorMap(out, colormap)

    textSize = cv2.getTextSize(str(num_class+1),
                               fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                               fontScale=font_scale, thickness=font_thickness)

    r1 = int(textSize[0][0]*1.05)

    if col_labels:
        textSize = cv2.getTextSize(str(num_class+1),
                               fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                               fontScale=font_scale, thickness=font_thickness)

        r2 = int(textSize[0][1]*1.05)

        out = cv2.copyMakeBorder(out, top=r2, bottom=0, left=0, right=0,
                                borderType=cv2.BORDER_CONSTANT,
                                value=[0, 0, 0])
        xs = np.linspace(1, out.shape[1] - 1, mat.shape[1] + 1).astype(int)
        for i in range(len(xs) - 1):
            dx = int((xs[i+1] - xs[i]))
            cv2.line(out, (xs[i], 0), (xs[i], out.shape[0]), (255, 255, 255),
                     thickness=2)

            textSize = cv2.getTextSize(str(i + first_ind),
                                       fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                       fontScale=font_scale,
                                       thickness=font_thickness)
            y = int((r2 - textSize[0][1])/2 - 2)
            x = int(xs[i + 1] - (dx - textSize[0][1])/2)

            cv2.putText(out, str(i + first_ind), (x, y),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=font_scale,
                        color=(255, 255, 255),
                        thickness=font_thickness, lineType=1)
    else:
        r2 = 0

    out = cv2.copyMakeBorder(out, top=0, bottom=0, left=r1, right=0,
                                borderType=cv2.BORDER_CONSTANT,
                                value=[0, 0, 0])

    cv2.line(out, (0, 0), (out.shape[1], 0), (255, 255, 255), thickness=2)
    ys = np.linspace(r2 + 1, out.shape[0] - 1, num_class + 1).astype(int)
    for i in range(len(ys) - 1):
        dy = int((ys[i+1] - ys[i]))
        cv2.line(out, (0, ys[i] ), (out.shape[1], ys[i] ), (255, 255, 255),
                 thickness=2)

        textSize = cv2.getTextSize(str(i + first_ind),
                                   fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                   fontScale=font_scale,
                                   thickness=font_thickness)
        x = int((r1 - textSize[0][0])/2 - 2)
        y = int(ys[i + 1] - (dy - textSize[0][1])/2)

        cv2.putText(out, str(i + first_ind), (x, y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale,
                    color=(255, 255, 255),
                    thickness=font_thickness, lineType=1)

    cv2.line(out, (r1, 0), (r1, out.shape[0]), (255, 255, 255), thickness=2)

    out = cv2.copyMakeBorder(out, top=4, bottom=4, left=4, right=4,
                             borderType=cv2.BORDER_CONSTANT,
                             value=[255, 255, 255])

    cv2.imwrite(fname, out)