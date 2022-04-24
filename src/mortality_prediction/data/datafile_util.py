import os
from os.path import join
import json

root_dir, _ = os.path.split(os.path.abspath(__file__))
root_dir = os.path.dirname(root_dir)
root_dir = os.path.dirname(root_dir)
root_dir = os.path.dirname(root_dir)

embeddings_basedir = 'outputs/concept_embedding/'


def tesa_dict():
    tesa_dict_file = fullpath('dataset/processed/mimic3_dict.json')
    with open(tesa_dict_file, 'r', encoding='utf-8') as f:
        tesa_dict = json.load(f)
    return tesa_dict


def fullpath(filepath):
    return join(root_dir, embeddings_basedir, filepath)

