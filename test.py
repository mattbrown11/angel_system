import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import pandas as pd
import json
import yaml
import os
import scipy
import time
import matplotlib.pyplot as plt
import json
import matplotlib.pyplot as plt
import importlib.util
import yaml
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from random import shuffle
import pickle
import cv2
import copy

# angel imports
from angel_system.impls.detect_activities.detections_to_activities.utils import obj_det2d_set_to_feature
from angel_system.activity_hmm.eval import save_matrix_image


pred_train_fname = '/angel_workspace/ros_bags/m2_with_lab_cleaned_fixed_data_with_inter_and_before_finished_steps_results_train_activity.mscoco.json'
pred_test_fname = '/angel_workspace/ros_bags/m2_with_lab_cleaned_fixed_data_with_inter_and_before_finished_steps_results_test.mscoco.json'
activity_fname = ''
act_label_yaml = '/angel_workspace/config/activity_labels/medical_tourniquet.yaml'

with open(act_label_yaml, 'r') as stream:
    act_labels = yaml.safe_load(stream)

if False:
    with open(trimmed_recipe_yaml, 'r') as stream:
        trimmed_recipe = yaml.safe_load(stream)

    trim_act_map = {}
    for step in trimmed_recipe['steps']:
        ids = step['activity_id']
        try:
            ids = ids.split(',')
            for i in ids:
                trim_act_map[int(i)] = int(step['id'])
        except:
            trim_act_map[int(ids)] = int(step['id'])


def sanitize_str(txt):
    txt = txt.lower()

    # remove (step 1)
    try:
        txt = txt[:txt.index('(') - 1]
    except ValueError:
        pass

    if txt[-1] == '.':
        txt = txt[:-1]

    return txt


# Description to ID map.
act_map = {}
inv_act_map = {}
for step in act_labels['labels']:
    act_map[sanitize_str(step['label'])] = step['id']
    inv_act_map[step['id']] = step['label']

if 0 not in act_map.values():
    act_map['background'] = 0
    inv_act_map[0] = 'Background'

act_str_list = [inv_act_map[key] for key in range(max(inv_act_map.keys()) + 1)]


with open(pred_train_fname, 'r') as f:
    dat = json.load(f)

with open(pred_test_fname, 'r') as f:
    dat.update(json.load(f))

if False:
    activities = set()
    for im in dat['images']:
        try:
            activities.add(im['activity_gt'])
        except KeyError:
            pass


counts = {}
for im in dat['images']:
    if im['activity_gt'] not in counts:
        counts[im['activity_gt']] = 1
    else:
        counts[im['activity_gt']] += 1


image_activity_gt = {}
image_id_to_dataset = {}
for im in dat['images']:
    image_id_to_dataset[im['id']] = os.path.split(im['file_name'])[0]

    if im['activity_gt'] is None:
        continue

    image_activity_gt[im['id']] = (act_map[sanitize_str(im['activity_gt'])])

dsets = sorted(list(set(image_id_to_dataset.values())))
image_id_to_dataset = {i: dsets.index(image_id_to_dataset[i]) for i in image_id_to_dataset}


set(image_activity_gt.values())


min_cat = min([cat['id'] for cat in dat['categories']])
num_act = len(dat['categories'])
label_to_ind = {cat['name']: cat['id'] - min_cat  for cat in dat['categories']}
act_id_to_str = {cat['id']: cat['name'] for cat in dat['categories']}


ann_by_image = {}
for ann in dat['annotations']:
    if ann['image_id'] not in ann_by_image:
        ann_by_image[ann['image_id']] = [ann]
    else:
        ann_by_image[ann['image_id']].append(ann)


X = []
y = []
dataset_id = []
last_dset = 0
for image_id in ann_by_image:
    label_vec = []
    left = []
    right = []
    top = []
    bottom = []
    label_confidences = []
    obj_obj_contact_state = []
    obj_obj_contact_conf = []
    obj_hand_contact_state = []
    obj_hand_contact_conf = []

    for ann in ann_by_image[image_id]:
        label_vec.append(act_id_to_str[ann['category_id']])
        left.append(ann['bbox'][0])
        right.append(ann['bbox'][1])
        top.append(ann['bbox'][2])
        bottom.append(ann['bbox'][3])
        label_confidences.append(ann['confidence'])
        obj_obj_contact_state.append(ann['obj-obj_contact_state'])
        obj_obj_contact_conf.append(ann['obj-obj_contact_conf'])
        obj_hand_contact_state.append(ann['obj-hand_contact_state'])
        obj_hand_contact_conf.append(ann['obj-hand_contact_conf'])

    feature_vec = obj_det2d_set_to_feature(label_vec, left, right, top, bottom,
                                           label_confidences, None,
                                           obj_obj_contact_state,
                                           obj_obj_contact_conf,
                                           obj_hand_contact_state,
                                           obj_hand_contact_conf, label_to_ind,
                                           version=1)

    X.append(feature_vec.ravel())
    try:
        dataset_id.append(image_id_to_dataset[image_id])
        last_dset = dataset_id[-1]
    except:
        dataset_id.append(last_dset)

    try:
        #y.append(trim_act_map[image_activity_gt[image_id]])
        y.append(image_activity_gt[image_id])
    except:
        y.append(0)

X = np.array(X)
y = np.array(y)
dataset_id = np.array(dataset_id)
ind = list(range(len(y)))
shuffle(ind)
X = X[ind]
y = y[ind]
dataset_id = dataset_id[ind]

plt.imshow(np.cov(X.T))

y_ = y.tolist()
x_ = list(range(max(y) + 1))
counts = [y_.count(i) for i in x_]
plt.close('all')
fig = plt.figure(num=None, figsize=(14, 10), dpi=80)
plt.rc('font', **{'size': 22})
plt.rc('axes', linewidth=4)
plt.bar(x_, counts)
plt.xticks(x_)
plt.ylabel('Counts', fontsize=34)
plt.xlabel('Ground Truth Steps', fontsize=34)
plt.tight_layout()
plt.savefig('/angel_workspace/ros_bags/gt_counts.png')



def fit(clf, n_splits=2, n_components=50):
    group_kfold = GroupKFold(n_splits=n_splits)
    y_test = []
    ypred_test = []
    for i, (train_index, test_index) in enumerate(group_kfold.split(X, y, dataset_id)):
        #print(f"Fold {i}:")
        #print(f"  Train: index={train_index}, group={dataset_id[train_index]}")
        #print(f"  Test:  index={test_index}, group={dataset_id[test_index]}")
        X_ = X[train_index]
        y_ = y[train_index]

        if True:
            #X_ = PCA(whiten=True, n_components='mle').fit_transform(X_)
            #pipe = make_pipeline(PCA(whiten=True, n_components='mle'), clf)
            pipe = make_pipeline(PCA(whiten=True, n_components=n_components),
                                 clf)
        else:
            pipe = make_pipeline(RobustScaler(), clf)

        if isinstance(clf, MLPClassifier):
            clf.solver = 'adam'
            clf.warm_start = False
            clf.max_iter = 200
            pipe.fit(X_, y_)
            clf.solver = 'lbfgs'
            clf.warm_start = True
            clf.max_iter = 10000
            pipe.fit(X_, y_)
        else:
            pipe.fit(X_, y_)

        print(accuracy_score(y[test_index], pipe.predict(X[test_index])))

        y_test.append(y[test_index])
        ypred_test.append(pipe.predict(X[test_index]))

    y_test = np.hstack(y_test)
    ypred_test = np.hstack(ypred_test)
    return accuracy_score(y_test, ypred_test), y_test, ypred_test


# Logistic regression.
clf = LogisticRegression(random_state=0, max_iter=1000)
C = np.logspace(-6, 2, 20)
err = []
for C_ in C:
    clf.C = C_
    err.append(fit(clf)[0])
    print(C_, err[-1])

plt.semilogx(C[:len(err)], err, 'ro')

clf.C = C[np.argmax(err)]

y_test, ypred_test = fit(clf, n_splits=5)[1:]


scores = []
for n_components in range(5, 50):
    score_ = fit(clf, n_components=n_components)[0]
    print(n_components, score_)
    scores.append(score_)

plt.plot(range(5, 50), scores)

# ----------------------------------------------------------------------------
# GaussianNB
best_score = 0
best_model = None

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
kernel = 1.0 * RBF(0.1)
clf = GaussianProcessClassifier(kernel=kernel, random_state=0)

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
print(fit(clf)[0])

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
clf = QuadraticDiscriminantAnalysis(reg_param=2)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=None, random_state=0, n_estimators=1000,
                             bootstrap=True)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)

from sklearn.ensemble import AdaBoostClassifier
clf = LogisticRegression(random_state=0, max_iter=1000)
clf = AdaBoostClassifier(clf, n_estimators=100, random_state=0)

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
clf = SVC(gamma=0.025)
clf = LinearSVC(random_state=0, tol=1e-5)

C = np.logspace(-2, 2, 20)
score = []
for C_ in C:
    clf.C = C_
    score_ = fit(clf)[0]
    score.append(score_)
    print(score_)
    if score_ > best_score:
        best_score = score_
        best_model = copy.deepcopy(clf)

    print(C_, score[-1])

plt.semilogx(C[:len(score)], score, 'ro')
#plt.semilogx(C[:len(score)], score, 'ro', label='gamma=%s' % clf.gamma)
plt.legend()
# ----------------------------------------------------------------------------



pipe = make_pipeline(PCA(whiten=True, n_components=15), clf)
pipe.fit(X, y)
pipe.predict(X[0].reshape(1, -1))

fname = '/angel_workspace/model_files/recipe_m2_apply_tourniquet_v0.052.pkl'
with open(fname, 'wb') as of:
    pickle.dump([label_to_ind, 1, pipe, act_str_list], of)











N = 15
hidden_layer_sizes = [N, N, N, N, N, N, N, N]
alpha = 1e-6
activation = 'relu'
solver = 'adam'
solver = 'lbfgs'
clf = MLPClassifier(solver=solver, alpha=alpha, activation=activation,
                    hidden_layer_sizes=hidden_layer_sizes,
                    learning_rate_init=0.001, max_iter=100,
                    tol=1e-3, n_iter_no_change=np.inf,
                    early_stopping=False, verbose=True)
#pipe = make_pipeline(StandardScaler(), clf)
#pipe.fit(X, y)
#print(pipe.score(X, y))

alphas = np.logspace(-4, 4, 50)
shuffle(alphas)
errs = []
for alpha in alphas:
    clf.alpha = alpha
    err, y_test, ypred_test = fit(clf)
    errs.append(err)
    print(alpha, err)
    fname = '/mnt/data10tb/projects/ptg/new_contact_process/models/%s.p' % str(int(err*1000000)).zfill(6)
    with open(fname, 'wb') as of:
        pickle.dump(clf, of)

plt.semilogx(alphas[:len(errs)], errs, 'ro')


fname = '/mnt/data10tb/projects/ptg/new_contact_process/models/432301.p'
with open(fname, 'rb') as of:
    clf2 = pickle.load(of)


def fit_mlp(x, y):
    solver = 'adam'
    alpha = 10**(-(np.random.rand(1))*10)
    #ind = int(np.random.randint(low=0, high=3))

    ind = 2
    if ind == 0:
        activation = 'logistic'
    elif ind == 1:
        activation = 'tanh'
    elif ind == 2:
        activation = 'relu'

    alpha = 1e-8
    hidden_layer_sizes = [50]*np.random.randint(1, 4)
    clf = MLPClassifier(solver=solver, alpha=alpha, activation=activation,
                        hidden_layer_sizes=hidden_layer_sizes,
                        learning_rate_init=0.01, max_iter=200,
                        tol=1e-6, n_iter_no_change=np.inf,
                        early_stopping=True, verbose=True)
    pipe = make_pipeline(StandardScaler(), clf)
    pipe.fit(X, y)

    clf.solver = 'lbfgs'
    #clf.solver = 'adam'
    clf.verbose = True
    #clf.verbose = False
    clf.tol = 1e-6
    #clf.verbose = False
    clf.warm_start = True
    clf.max_iter = 10000

    while True:
        pipe.fit(X, y)
        print(pipe.score(X, y))

y_pred = pipe.predict(X)


A = confusion_matrix(y_test, ypred_test, labels=range(max(y)+1), normalize='pred')

plt.close('all')
fig = plt.figure(num=None, figsize=(14, 10), dpi=80)
plt.rc('font', **{'size': 22})
plt.rc('axes', linewidth=4)
plt.imshow(A)
cbar = plt.colorbar()
cbar.set_label('Percent', rotation=270, fontsize=34)
cbar.set_label('Percent', labelpad=+30, rotation=270, fontsize=34)
plt.xticks(range(0, len(A)))
plt.yticks(range(0, len(A)))
plt.ylabel('True Steps', fontsize=34)
plt.xlabel('Predicted Steps', fontsize=34)
plt.tight_layout()
plt.savefig('/angel_workspace/ros_bags/tourniquet_step_classification.png')
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
fname = '/angel_workspace/ros_bags/kitware_test_results_test.mscoco.json'
with open(pred_train_fname, 'r') as f:
    dat = json.load(f)

ann_by_image = {}
for ann in dat['annotations']:
    if ann['image_id'] not in ann_by_image:
        ann_by_image[ann['image_id']] = [ann]
    else:
        ann_by_image[ann['image_id']].append(ann)

# Generate features using code above.

fname = '/angel_workspace/model_files/recipe_m2_apply_tourniquet_v0.052.pkl'
with open(fname, 'rb') as of:
    label_to_ind, feat_ver, clf, act_str_list = pickle.load(of)


label_to_ind = {cat['name']: cat['id'] - min_cat  for cat in dat['categories']}
act_id_to_str = {cat['id']: cat['name'] for cat in dat['categories']}


X = []
y = []
dataset_id = []
last_dset = 0
for image_id in ann_by_image:
    label_vec = []
    left = []
    right = []
    top = []
    bottom = []
    label_confidences = []
    obj_obj_contact_state = []
    obj_obj_contact_conf = []
    obj_hand_contact_state = []
    obj_hand_contact_conf = []

    for ann in ann_by_image[image_id]:
        label_vec.append(act_id_to_str[ann['category_id']])
        left.append(ann['bbox'][0])
        right.append(ann['bbox'][1])
        top.append(ann['bbox'][2])
        bottom.append(ann['bbox'][3])
        label_confidences.append(ann['confidence'])
        obj_obj_contact_state.append(ann['obj-obj_contact_state'])
        obj_obj_contact_conf.append(ann['obj-obj_contact_conf'])
        obj_hand_contact_state.append(ann['obj-hand_contact_state'])
        obj_hand_contact_conf.append(ann['obj-hand_contact_conf'])

    feature_vec = obj_det2d_set_to_feature(label_vec, left, right, top, bottom,
                                           label_confidences, None,
                                           obj_obj_contact_state,
                                           obj_obj_contact_conf,
                                           obj_hand_contact_state,
                                           obj_hand_contact_conf, label_to_ind,
                                           version=1)

    X.append(feature_vec.ravel())
    try:
        dataset_id.append(image_id_to_dataset[image_id])
        last_dset = dataset_id[-1]
    except:
        dataset_id.append(last_dset)

    try:
        #y.append(trim_act_map[image_activity_gt[image_id]])
        y.append(image_activity_gt[image_id])
    except:
        y.append(0)

X = np.array(X)
y = np.array(y)
dataset_id = np.array(dataset_id)



# ----------------------------------------------------------------------------
y_pred = clf.predict_proba(X)

fname = '/angel_workspace/ros_bags/det_step_class.png'
save_matrix_image(X.T, fname, max_w=8000, aspect_ratio=1, first_ind=1)

fname = '/angel_workspace/ros_bags/step_class.png'
save_matrix_image(clf.predict_proba(X).T, fname, max_w=8000, aspect_ratio=4, first_ind=0)


#h2 = w



# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
import angel_system
from angel_system.activity_hmm.core import ActivityHMMRos

config_fname = '/angel_workspace/config/tasks/task_steps_config-recipe_m2_apply_tourniquet_v0.052.yaml'
print(f'Loading HMM with recipe {config_fname}')
live_model = ActivityHMMRos(config_fname)

live_model.class_std_conf = np.maximum(live_model.class_std_conf, 5)

if False:
    fname = '/angel_workspace/ros_bags/hhm_class_mean_conf.png'
    save_matrix_image(live_model.class_mean_conf, fname, max_w=4000, aspect_ratio=1, first_ind=0)

    fname = '/angel_workspace/ros_bags/hhm_class_std_conf.png'
    a = live_model.class_mean_conf/live_model.class_std_conf
    a = a/np.max(a)
    save_matrix_image(a, fname, max_w=4000, aspect_ratio=1, first_ind=0)


t = 0
dt = 0.025
step_finished_confs = []
i = 0
avg_steps = 50
while i < len(y_pred):
    print(i)
    conf_vec = np.max(y_pred[i:i+avg_steps], axis=0)
    live_model.add_activity_classification(range(live_model.num_activities),
                                           conf_vec, t, t + dt)
    t += dt
    step_finished_conf = live_model.analyze_current_state()[3]
    step_finished_confs.append(step_finished_conf)
    i += avg_steps


step_finished_confs = np.array(step_finished_confs).T

fname = '/angel_workspace/ros_bags/hhm_step_finished_confs.png'
save_matrix_image(step_finished_confs, fname, max_w=4000, aspect_ratio=4,
                  first_ind=0, colormap=cv2.COLORMAP_HOT)
