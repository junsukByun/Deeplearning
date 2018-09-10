# -*- coding: utf-8 -*-
#!/usr/bin/env python
from Inference.Tensorflow_inference import *
# import openpyxl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test_path', type= str, help="path_of_inference")
parser.add_argument('--model_path', type= str)
parser.add_argument('--batch_size', type= int)
parser.add_argument('--name', type= str)
parser.add_argument('--save_path', type= str)
parser.add_argument('--threshold', type= float)
parser.add_argument('--multicrop', type= str)
# parser.add_argument('--mode', type=str)

args = parser.parse_args()


# a = AutoInference_v4(test_path = '/home/storage/hdd2/data/LGD_P7/Factory_confirm/R1_SD_final/20180626_11_multi_crop/RPC/',    
#                             model_path = '/home/storage/ssd/DIGITS/digits/jobs/20180529-095032-ce41',
#                             model_name = 'R1_train_augmentation___R1_SD_final_20180626',
#                             mode = 'P7')
inf = Inference_v1(test_path = args.test_path,    
                            model_path = args.model_path,
                            batch_size = args.batch_size,
                            name = args.name,
                            save_path = args.save_path,
                            threshold = args.threshold,
                            multicrop = args.multicrop  
                              )

inf.build()
inf.inference()