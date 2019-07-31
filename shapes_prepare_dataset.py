import torch
from torch.autograd import Variable
import torch.nn as nn
from glob import glob
import os
import numpy as np
from math import floor
from random import shuffle
from shutil import copyfile

def split_into_train_valid_and_test(dirName, ext):
    allFiles = list()
    for root, dirs, files in os.walk(dirName):
        for file in files:
            if file.endswith(ext):
                allFiles.append(os.path.join(root, file))

    shuffle(allFiles)

    split_ratio_for_testing = 0.8
    split_index = floor(len(allFiles) * split_ratio_for_testing)
    training_plus_validation = allFiles[:split_index]
    testing = allFiles[split_index:]

    split_ratio_for_validation = 0.8
    split_index = floor(len(training_plus_validation) * split_ratio_for_validation)
    training = training_plus_validation[:split_index]
    validation = training_plus_validation[split_index:]

    return training, validation, testing

def split_into_train_and_test(dirName, ext):
    allFiles = list()
    for root, dirs, files in os.walk(dirName):
        for file in files:
            if file.endswith(ext):
                allFiles.append(os.path.join(root, file))

    shuffle(allFiles)

    split_ratio_for_testing = 0.8
    split_index = floor(len(allFiles) * split_ratio_for_testing)
    training_plus_validation = allFiles[:split_index]
    testing = allFiles[split_index:]

    return training_plus_validation, testing

#object_list = ['circle', 'square', 'star', 'triangle']
object_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#path_target = 'images/'
path_target = '/Users/macbookpro/Documents/PythonProjects/Sign_Language_Digits/images/'

for object in object_list:

    #path = '/Users/macbookpro/Documents/PythonProjects/shapes/' + object + '/'
    path = '/Users/macbookpro/Documents/PythonProjects/Sign_Language_Digits/dataset/' + object + '/'

    #training, validation, testing = split_into_train_valid_and_test(path, ".png")
    #print(f'number of images: training {len(training)}, validation {len(validation)}, testing {len(testing)}')

    training, testing = split_into_train_and_test(path, ".JPG")
    print(f'number of images: training {len(training)}, testing {len(testing)}')

    for file in training:
        directory = path_target + "train/" + object + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = os.path.basename(file)
        copyfile(file, directory + filename)

    '''
    for file in validation:
        directory = path_target + "valid/" + object + "/"
            if not os.path.exists(directory):
                os.makedirs(directory)
        filename = os.path.basename(file)
        copyfile(file, directory + filename)
    '''

    for file in testing:
        directory = path_target + "test/" + object + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = os.path.basename(file)
        copyfile(file, directory + filename)


create_directories = False
if create_directories:
    os.mkdir(os.path.join(path_target, 'train'))
    os.mkdir(os.path.join(path_target, 'valid'))
    os.mkdir(os.path.join(path_target, 'test'))
    for t in ['train', 'valid', 'test']:
        for folder in ['circle/', 'square/', 'star/', 'triangle/']:
            os.mkdir(os.path.join(path_target, t, folder))


