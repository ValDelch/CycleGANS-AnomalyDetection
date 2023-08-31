"""
Main code used to obtain the results from the paper:
"Industrial and Medical Anomaly Detection Through Cycle-Consistent Adversarial Networks"

Dataset should all match the following structure:

data_path
|
|__ pcam
|   |__ normal
|   |__ abnormal
|
|__ brain
|   |__ normal
|   |__ abnormal
|
...

Attention is paid to keep balanced testing sets, with min([len(normal images), len(abnormal images)]) / 2. images per class.
"""

import json
import os
import numpy as np
import torch
from load_data import return_dataset
from training_setup import return_training_setup
from train import return_trained_model
from infer import infer
from metrics import conf_mat, metric_conf, auc_roc_score

import argparse
parser = argparse.ArgumentParser(description='Training and inference for AD')
parser.add_argument('data_path', type=str, help='path to the datasets')
parser.add_argument('save_dir', type=str, help='directory used to save the results')
parser.add_argument('model', type=str, help='model to use among cgan256|cgan64|ganomaly|patchcore|padim')
parser.add_argument('dataset', type=str, help='dataset to use among pcam|brain|breast|oct|wood|tile|hazelnut|screw')
parser.add_argument('run_id', type=int, help='id for the run')
parser.add_argument('--infer', action='store_true', default=False, help='only perform inference by loading weights from save_dir')
args = parser.parse_args()

with open('./configs/models_config.json', 'r') as json_file:
    model_config = json.load(json_file)[args.model]
with open('./configs/datasets_config.json', 'r') as json_file:
    dataset_config = json.load(json_file)[args.dataset]

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

""" Instantiate the dataset """
data = return_dataset(args.dataset, 
                      args.data_path, 
                      model_config, 
                      dataset_config, 
                      args.run_id)
print('\n>> Dataset loaded\n')

""" Instantiate the model """
training_setup = return_training_setup(args.model, 
                                       model_config,
                                       dataset_config,
                                       device)
print('\n>> Model loaded\n')

""" Call the training loop """
model = return_trained_model(args.save_dir, 
                             args.model, 
                             args.dataset,
                             args.infer,
                             args.run_id,
                             data[0], 
                             training_setup, 
                             model_config,
                             dataset_config, 
                             device)
print('\n>> Model trained\n')

""" Call the inference loop """
maps_normal, maps_abnormal = infer(args.model, 
                                   model, 
                                   data[1], 
                                   data[2], 
                                   device)
print('\n>> Inference performed\n')

# Save the maps
with open(os.path.join(args.save_dir, args.dataset, args.model, str(args.run_id), 'maps_normal.json'), 'w') as json_file:
    json.dump(maps_normal, json_file, indent=4)
with open(os.path.join(args.save_dir, args.dataset, args.model, str(args.run_id), 'maps_abnormal.json'), 'w') as json_file:
    json.dump(maps_abnormal, json_file, indent=4)

""" Save the results """
result_acc = []
fp,tp,fn,tn = {},{},{},{}
fp_50,tp_50,fn_50,tn_50 = {},{},{},{}
accuracy,precision,specificity,specificity,fpr,fnr = {},{},{},{},{},{}
accuracy_50,precision_50,specificity_50,specificity_50,fpr_50,fnr_50,auc_score = {},{},{},{},{},{},{}
accuracy_scores = dict.fromkeys(maps_normal.keys(), [])
data =  {}
accuracies, max_accuracy, max_accuracy_threshold =  {},{},{}

for metric in maps_normal.keys():
    fp[metric],tp[metric],fn[metric],tn[metric] = conf_mat(maps_normal[metric], maps_abnormal[metric], min(maps_abnormal[metric]))
    accuracy[metric],precision[metric],specificity[metric],specificity[metric],fpr[metric],fnr[metric] = metric_conf(fp[metric],tp[metric],fn[metric],tn[metric])

    data[metric] = sorted(maps_normal[metric]+maps_abnormal[metric])
    for thresh in data[metric]:
        fp_50[metric],tp_50[metric],fn_50[metric],tn_50[metric] = conf_mat(maps_normal[metric], maps_abnormal[metric], thresh)
        accuracy_50[metric] = round((len(tp_50[metric]) + len(tn_50[metric])) / (len(tp_50[metric]) + len(fn_50[metric]) + len(fp_50[metric]) + len(tn_50[metric])),4)
        accuracy_scores[metric].append(accuracy_50[metric])
    accuracies[metric] = np.array(accuracy_scores[metric])
    max_accuracy[metric] = accuracies[metric].max() 
    max_accuracy_threshold[metric] =  data[metric][accuracies[metric].argmax()]

    fp_50[metric],tp_50[metric],fn_50[metric],tn_50[metric] = conf_mat(maps_normal[metric], maps_abnormal[metric], max_accuracy_threshold[metric])
    accuracy_50[metric],precision_50[metric],specificity_50[metric],specificity_50[metric],fpr_50[metric],fnr_50[metric] = metric_conf(fp_50[metric],tp_50[metric],fn_50[metric],tn_50[metric])

    auc_score[metric] = auc_roc_score(maps_normal[metric], maps_abnormal[metric])

# Saving the results
with open(os.path.join(args.save_dir, args.dataset, args.model, str(args.run_id), 'accuracy.json'), 'w') as json_file:
    json.dump(accuracy, json_file, indent=4)
with open(os.path.join(args.save_dir, args.dataset, args.model, str(args.run_id), 'accuracy_50.json'), 'w') as json_file:
    json.dump(accuracy_50, json_file, indent=4)
with open(os.path.join(args.save_dir, args.dataset, args.model, str(args.run_id), 'auc_score.json'), 'w') as json_file:
    json.dump(auc_score, json_file, indent=4)
with open(os.path.join(args.save_dir, args.dataset, args.model, str(args.run_id), 'max_accuracy_threshold.json'), 'w') as json_file:
    json.dump(max_accuracy_threshold, json_file, indent=4)
print('\n>> Results saved\n')