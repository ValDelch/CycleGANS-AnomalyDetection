"""
Main code used to obtain the results for the paper:
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

Attention is paid to keep balanced testing sets, with 50% of the available abnormal images
"""

import json
from load_data import return_dataset

import argparse
parser = argparse.ArgumentParser(description='Training and inference for AD')
parser.add_argument('data_path', type=str, help='path to the datasets')
parser.add_argument('save_dir', type=str, help='directory used to save the results')
parser.add_argument('model', type=str, help='model to use among cgan256|cgan64|patchcore|padim')
parser.add_argument('dataset', type=str, help='dataset to use among pcam|brain|breast|oct|wood|tile|hazelnut|screw')
parser.add_argument('run_id', type=int, help='id for the run')
parser.add_argument('--infer', action='store_true', default=False, help='only perform inference by loading weights from save_dir')
args = parser.parse_args()

with open('models_config.json', 'r') as json_file:
    model_config = json.load(json_file)[args.model]

""" Instantiate the dataset """
data = return_dataset(args.dataset, 
                      args.data_path, 
                      model_config['always_RGB'], 
                      model_config['max_img_size'], 
                      model_config['data_loader'],
                      args.run_id)

""" Instantiate the model """


""" Call the training loop """


""" Call the inference loop """


""" Save the results """