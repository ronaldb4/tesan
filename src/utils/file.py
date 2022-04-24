import json
import pickle
import os

def load_squad_dataset(filePath):
    with open(filePath, 'r', encoding='utf-8') as data_file:
        line = data_file.readline()
        dataset = json.loads(line)

    return dataset['data']

def save_file(data, filePath, dataName = 'data', mode='pickle'):
    if mode == 'pickle':
        with open(filePath, 'wb') as f:
            pickle.dump(obj=data,file=f)
    elif mode == 'json':
        with open(filePath, 'w', encoding='utf-8') as f:
            json.dump(obj=data,fp=f)
    else:
        raise(ValueError,'Function save_file does not have mode %s' % (mode))


def load_file(filePath, dataName = 'data', mode='pickle'):
    data = None
    is_load = False
    if os.path.isfile(filePath):
        if mode == 'pickle':
            with open(filePath, 'rb') as f:
                data = pickle.load(f)
                is_load = True
        elif mode == 'json':
            with open(filePath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                is_load = True
        else:
            raise (ValueError, 'Function save_file does not have mode %s' % (mode))
    return is_load, data


def save_nn_model(modelFilePath, allParams, epoch):
    with open(modelFilePath,'wb') as f:
        pickle.dump(obj=[[param.get_value() for param in allParams ],
                         epoch],
                    file = f)

def load_nn_model(modelFilePath):
    allParamValues = None
    epoch = 1
    isLoaded = False
    if os.path.isfile(modelFilePath):
        with open(modelFilePath, 'rb') as f:
            data = pickle.load(f)
            allParamValues = data[0]
            epoch = data[1]
            isLoaded = True
    return isLoaded, allParamValues, epoch