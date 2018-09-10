# -*- coding: utf-8 -*-
from Helper.Library import *
from Helper.Make_matrix import *
import tensorflow as tf
import tensorflow.contrib.image.ops
from PIL import Image
import time
import shutil
from Inference.Read_inference_data import *
import numpy as np
np.random.seed(1111)
import glob
import openpyxl
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

class Inference_v1():
    
    def __init__(self, test_path, model_nm, batch_size, name, multicrop= False):
        self.test_path = test_path
        self.model_path = 'checkpoints' +'/'+ model_nm
        self.batch_size = batch_size
        self.name = name
        self.multicrop = multicrop
        self.index_in_epoch = 0
        self.target_leak_ratio = 0.8
        self.layer = 'default'
        self.epoch = -1
        self.inference_start_time = time.time()

    # Input data loading
    def Data_prepration(self):
        self.test_list, self.X_test, self.Y_test = Data_preprocessing(self.test_path)
        self.test_filenames = image_name_parser(self.test_list)

    def next_batch(self):
        self.start = self.index_in_epoch
        self.index_in_epoch += self.batch_size
        self.end = self.index_in_epoch
        return self.X_test[self.start:self.end], self.Y_test[self.start:self.end]

    # Mode loading
    def Model_load(self):
        list_iter = sorted(glob.glob(self.model_path +'/*00001'), key=lambda x: int(x.split('.data')[0].split('-')[-1]))
        list_iter_num = [int(x.split('.data')[0].split('-')[-1]) for x in list_iter]
        print(list_iter_num)
        chosen_iter_num = int(input('Please choose a model:')) 
        self.epoch = list_iter_num.index(chosen_iter_num)
        print('Epoch' + ' ' + str(list_iter_num.index(chosen_iter_num)) + ' is selected')
        print('Model selection finished')
    
    def build(self):
        self.Data_prepration()
        self.Model_load()
             
    def inference(self):
        saver = tf.train.import_meta_graph(glob.glob(self.model_path +'/*meta')[0])    # 726
        graph = tf.get_default_graph()         
#         for op in graph.get_operations():
#             print(op.name)
        input_img = graph.get_tensor_by_name("get_data_1/Batch/cond/Merge_1:0")
#         input_img = graph.get_tensor_by_name("get_data/IteratorGetNext:1")
        softmax = graph.get_tensor_by_name("predict/Softmax:0")    
        start_time = time.time()        
        self.total_softmax = []
        self.total_labels = []
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        with tf.Session(config=config) as sess :
            saver.restore(sess,tf.train.get_checkpoint_state(self.model_path).all_model_checkpoint_paths[self.epoch])
            print('model restored')
            while True:
                X_test_batch, Y_test_batch = self.next_batch()
                
                self.softmax_batch = sess.run(softmax, feed_dict={input_img:X_test_batch})
                self.total_softmax.extend(list(self.softmax_batch))
                self.total_labels.extend(list(Y_test_batch))
                
                if self.end >= self.X_test.shape[0]:
                    print("Infernece finished")
                    print('Inference time: {0} seconds'.format(time.time() - start_time)) 
                    break
            
            s_time = time.time()        
            self.df, self.df_target = AutoFinder(self.total_softmax,self.total_labels, self.test_filenames, self.multicrop, self.target_leak_ratio)
            try:
                self.best_threshold = round(self.df[self.leak_ratio <= self.target_leak_ratio].sort_values('leak_ratio').iloc[0,:].threshold, 4)
            except:
                self.best_threshold = self.df.iloc[-1,:]['threshold']
            print("Threshold with leak_ratio target_leak%%: %.4f" %self.best_threshold)
            newpath = 'Inference_result' +'/'+ self.name
            if not os.path.exists('Inference_result'):
                os.makedirs('Inference_result')    
            writer = pd.ExcelWriter(newpath + '_threshold.xlsx')
            self.df.to_excel(writer,self.layer)
            writer.save() 
            print("Make files of Inference time : {0} seconds" .format(time.time() - s_time))
            
            
            print('Target stats: Accuracy: {0},  Leak_ratio: {1}, Threshold: {2}'.format(self.df_target['accuracy'], 
                                                                                         self.df_target['leak_ratio'], 
                                                                                         self.df_target['threshold']))
            print('Total inference time: {0} seconds'.format(time.time() - self.inference_start_time ))

    