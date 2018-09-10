# -*- coding: utf-8 -*-
#!/usr/bin/env python

from Training.Make_Model import *
# import openpyxl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type= str, )
parser.add_argument('--test_path', type= str)
parser.add_argument('--tfrecord_filename', type= str)
parser.add_argument('--fileread_mode', type= str)
parser.add_argument('--learning_rate', type= float)
parser.add_argument('--batch_size', type= int)
parser.add_argument('--n_classes', type= int)
parser.add_argument('--pos_weight', type= float)
parser.add_argument('--target_leak_ratio', type= float)
parser.add_argument('--n_epochs', type= int)
parser.add_argument('--model_nm', type= str)


args = parser.parse_args()

inf = Inference_v1(train_path = args.test_path,
                   test_path = args.test_path,
                   tfrecord_filename = args.tfrecord_filename,
                   fileread_mode = args.fileread_mode,
                   learning_rate = args.learning_rate,
                   batch_size = args.batch_size,
                   batch_size = args.batch_size,
                   n_classes = args.n_classes,
                   pos_weight = args.pos_weight,
                   target_leak_ratio = args.target_leak_ratio,
                   n_epochs = args.n_epochs,
                   model_nm = args.model_nm
                              )

inf.build()
inf.inference()