# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from Helper.Library import *
from Training.Read_train_data import *
from Training.TFrecord_generation import *
from Helper.Augmentation import *
from Training.Read_TFrecord import *
from Helper.Make_matrix import *
from Helper.Network import *
from tensorflow.python.ops import variable_scope
import numpy as np
np.random.seed(1111)
import tensorflow as tf
import time
import pandas as pd
    
class LGD_non_equip_model(object):

    def __init__(self, train_path, test_path, Killer_batch_ratio, tfrecord_filename, file_read_mode, network, learning_rate, batch_size, n_classes, pos_weight, target_leak_ratio, n_epochs, model_nm):
        tf.reset_default_graph()
        self.train_path = train_path
        self.test_path = test_path
        self.tfrecord_filename = tfrecord_filename
        self.file_read_mode = file_read_mode
        self.network = network
        self.lr = learning_rate
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.pos_weight = pos_weight  ## Refer to https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits        
        self.target_leak_ratio = target_leak_ratio
        self.n_epochs = n_epochs
        self.model_nm = model_nm        
        self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.skip_steps = 30
        self.Killer_batch_size = int((self.batch_size) * (Killer_batch_ratio))
        self.Nonkiller_batch_size = int(self.batch_size - self.Killer_batch_size)
      
            
    def write_data(self):
        with tf.name_scope('write_data'):
            
            self.tf_filenm = self.tfrecord_filename
            
            if self.file_read_mode == 'write':
                self.n_train, self.n_test, self.data_list = read_datalist(self.train_path, self.test_path)     
                make_dataset(self.data_list, self.tf_filenm, self.file_read_mode)
                self.train_kl, self.train_nk, self.valid_kl, self.valid_nk = len(self.data_list[('train', 'Killer')]), len(self.data_list[('train', 'Nonkiller')]), len(self.data_list[('test', 'Killer')]), len(self.data_list[('test', 'Nonkiller')])
              
                with open('./TFrecords/' + self.tfrecord_filename + '_information.txt', 'a') as f:
                    f.writelines('train path for making TFrecord : {0} \n test path for making TFrecord : {1}'.format(self.train_path, self.test_path))
                               
            else:
                with open('TFrecords/'+self.tf_filenm+ '_information.txt', 'r') as f:
                    data = f.readlines()
                    train_kl = int(data[0].split(':')[1])
                    train_nk = int(data[1].split(':')[1])
                    valid_kl = int(data[2].split(':')[1])
                    valid_nk = int(data[3].split(':')[1])
                print('Since you want to read TFrecord, [train_path] and [test_path] dont affect this training')
                print('Load saved TFrecord')
                print(' train Nonkiller:{}\n train Killer:{}\n test Nonkiller:{}\n test Killer:{}'.format(train_nk, train_kl, valid_nk, valid_kl))
#                 print('In every batch, Killer data is {} and NonKiller data is {}. In one epoch, used  
                self.n_train = train_kl + train_nk
                self.n_test = valid_kl + valid_nk
                self.train_kl, self.train_nk, self.valid_kl, self.valid_nk = train_kl, train_nk, valid_kl, valid_nk
                pass
            
    def Killer_get_next(self):
        labels_Killer, images_Killer = self.iterator_Killer.get_next()        
        return labels_Killer, images_Killer
    
    def Nonkiller_get_next(self):ㅊ
        labels_Nonkiller, images_Nonkiller = self.iterator_Nonkiller.get_next()
        return labels_Nonkiller, images_Nonkiller
    
    def return_train_get_next(self):
        labels_Nonkiller, images_Nonkiller = self.Nonkiller_get_next()
        labels_Killer, images_Killer = self.Killer_get_next()
        return tf.concat([labels_Killer, labels_Nonkiller], axis=0), tf.concat([images_Killer, images_Nonkiller], axis=0) 

    def return_test_get_next(self):
        labels_test, images_test = self.iterator_test.get_next()
        return labels_test, images_test
                    
    def get_data_under_construction_v2(self):          
        with tf.name_scope('data_get'):

            # tf.dataset 구성
            self.train_dataset, self.test_dataset = read_train_and_test(self.tf_filenm)                        

            # Train Killer dataset 구성
            with tf.name_scope('Train_Killer'):
                self.train_Killer = self.train_dataset.filter(lambda x,y: tf.reshape(tf.cast(x[0], tf.bool), []))# default 1:1 
                self.train_Killer = self.train_Killer.shuffle(buffer_size=(int(self.n_train)*2 + 3 * self.batch_size))
                self.train_Killer = self.train_Killer.batch(self.Killer_batch_size)
                self.train_Killer = self.train_Killer.prefetch(1)
                self.train_Killer = self.train_Killer.cache()
                self.train_Killer = self.train_Killer.repeat() 
                self.iterator_Killer = self.train_Killer.make_initializable_iterator()

            with tf.name_scope('Train_Nonkiller'):
                self.train_Nonkiller = self.train_dataset.filter(lambda x,y: tf.reshape(tf.cast(x[1], tf.bool), [])) 
                self.train_Nonkiller = self.train_Nonkiller.shuffle(buffer_size=(int(self.n_train)*2 + 3 * self.batch_size))            
                self.train_Nonkiller = self.train_Nonkiller.batch(self.Nonkiller_batch_size)
                self.train_Nonkiller = self.train_Nonkiller.prefetch(1)
                self.train_NonKiller = self.train_Nonkiller.cache()
                self.iterator_Nonkiller = self.train_Nonkiller.make_initializable_iterator()
           
            with tf.name_scope('Test'):            
                self.test_dataset = self.test_dataset.batch(10)
                self.test_dataset = self.test_dataset.prefetch(1)
                self.test_dataset = self.test_dataset.cache()
                self.iterator_test = self.test_dataset.make_initializable_iterator()
            
            with tf.name_scope('Batch'):
                #참조: https://stackoverflow.com/questions/46622490/how-to-use-tensorflows-tf-cond-with-two-different-dataset-iterators-without-i
                self.labels, self.images = tf.cond(self.is_training, lambda : self.return_train_get_next(), lambda : self.return_test_get_next())
            
    def network_structure(self):
        if self.network == 'inception_resnet_v2':
            emb = inception_resnet_v2(inputs = self.images, is_training=self.is_training)
        elif self.network == 'inception_v1':
            emb = inception_v1(inputs = self.images, is_training=self.is_training)
        elif self.network == 'inception_v2':
            emb = inception_v2(inputs = self.images, is_training=self.is_training)
        elif self.network == 'inception_v3':
            emb = inception_v3(inputs = self.images, is_training=self.is_training)
        else:
            raise ValueError('network name is not in list')

        self.logits = slim.fully_connected(emb, self.n_classes, activation_fn=None, scope='logits')

    def loss(self):
        with tf.name_scope('loss'):
            if self.pos_weight != 1:
                entropy = tf.nn.weighted_cross_entropy_with_logits(targets = self.labels, logits = self.logits, pos_weight = self.pos_weight)
            else:
                entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits) 
            self.loss = tf.reduce_mean(entropy, name='loss')
                
    def optimize(self):
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.gstep)
        
    def summary(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.histogram('histogram loss', self.loss)
            self.summary_op = tf.summary.merge_all()
     
    def eval(self):
        with tf.name_scope('predict'):
            self.preds = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(self.preds, 1)
            actuals = tf.argmax(self.labels, 1)
            correct_preds = tf.equal(self.predictions, actuals)
            self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
            
            leak_preds = tf.logical_and(tf.equal(actuals, tf.zeros_like(actuals)), tf.equal(self.predictions, tf.ones_like(self.predictions)))
            self.leak_ratio = tf.reduce_sum(tf.cast(leak_preds, tf.float32))

    def build(self):
        build_start_time = time.time()
        self.write_data()
        self.get_data_under_construction_v2()
#         self.get_data()   # Data 로딩
        self.network_structure()  # Model 정의
        self.loss()       # Loss 정의
        self.optimize()   # Optimizer 정의
        self.eval()       # Evaluation 정의
        self.summary()    # Summary (tensorboard 용) 정의 
        self.build_time = time.time() - build_start_time
#         print('Build model time(TFrecord time + readTFrecord + other settings) : {0}'.format(self.build_time))
    
    def train_one_epoch(self, sess, saver, writer, epoch, step, save_path):        
        start_time = time.time()
        sess.run([self.iterator_Killer.initializer, self.iterator_Nonkiller.initializer]) 
        total_loss = 0
        n_batches = 0
        
        try:
            while True:           
                self.images_vals, self.labels_vals, self.logits_vals, _, l, summaries = sess.run([self.images, self.labels, self.logits, self.opt, self.loss, self.summary_op], feed_dict = {self.is_training: True})
#                 print('fetch time took: %.2f seconds' % (time.time() - start_time))                                                                                   
                writer.add_summary(summaries, global_step=step)
#                 if step % self.skip_steps == 0:
                print('Loss at epoch %d (step: {%d}): %.2f' % (epoch,step,l))
                step += 1
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        
        saver.save(sess, 'checkpoints/'+ save_path +'/' + save_path, step, write_meta_graph = True)        
#         print('Average loss at epoch %d (step: %d) : %.2f' % (epoch, step, total_loss/n_batches))
        print('Training took: %.2f seconds' % (time.time() - start_time))
        self.train_one_time = time.time() - start_time
        return step     

    def eval_once(self, sess, writer, epoch, step):
        start_time = time.time()
        sess.run(self.iterator_test.initializer)
        self.total_preds = []
        self.total_labels = []
        self.df_target = pd.DataFrame()
        val_total_loss = 0
        total_correct_preds = 0
        total_leak_preds = 0

        try:
            while True:  #발견함! end of sequence 에러 발생..
                _, labels_batch, preds_batch, accuracy_batch, leak_batch, summaries, loss_batch = sess.run([self.images, self.labels, self.preds, self.accuracy, self.leak_ratio, self.summary_op,  self.loss], feed_dict = {self.is_training: False})
                writer.add_summary(summaries, global_step=step)
                self.total_preds.extend(list(preds_batch))
                self.total_labels.extend(list(labels_batch))
                total_correct_preds += accuracy_batch
                total_leak_preds += leak_batch
                val_total_loss += float(loss_batch * self.batch_size)

        except tf.errors.OutOfRangeError:
            pass
        
        self.df, self.df_target = AutoFinder(self.total_preds, self.total_labels, [], False, self.target_leak_ratio)
    
        print('Accuracy at epoch {0} (step: {1}) : {2} '.format(epoch, step, total_correct_preds/self.n_test * 100))
        print('Leak ratio at epoch {0} (step: {1}): {2}'.format(epoch, step, total_leak_preds/self.n_test * 100))
        print('Target stats at epoch {0} (step: {1}):  Accuracy: {2},  Leak_ratio: {3}, Threshold: {4}'.format(epoch, step, self.df_target['accuracy'], self.df_target['leak_ratio'], self.df_target['threshold']))
        train_one_epoch_time = time.time() - start_time + self.train_one_time
        print('train one epoch time: {0} seconds'.format(train_one_epoch_time))
        remaining_time = train_one_epoch_time * (self.n_epochs - 1 - epoch)
        hour = int(remaining_time/3600)
        minute = int((remaining_time - hour * 3600)/60)
        second = int(remaining_time) % 60
        print('Remaining train time: {} hours {} minutes {} seconds'.format(hour, minute, second))

    def train(self):
        
        start_time = time.time()
        save_path = self.model_nm
        safe_mkdir('checkpoints')
        safe_mkdir('checkpoints/' + save_path)
        writer = tf.summary.FileWriter('./graphs/' + save_path, tf.get_default_graph())
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config= config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=100)
            directory = 'checkpoints/' + self.model_nm + '/checkpoint'
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/' + self.model_nm + '/checkpoint'))        
            
            if ckpt and ckpt.model_checkpoint_path: ## 이미 모델이 만들어지고 재학습하는 경우
                with open(directory, 'r') as f:
                    backup = f.readlines()
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("<<<<<<< Continous training mode >>>>>>>")  # 이전 checkpoint backup 필요
                
                with open('checkpoints/' + self.model_nm + '/Modeling_information.txt', 'a') as f:
                    f.writelines('-----start train again----- \n' )
                    f.writelines('  train Killer : {}\n, train NonKiller : {}\n, valid Killer : {}\n, valid NonKiller : {}\n, '.format(self.train_kl, self.train_nk, self.valid_kl, self.valid_nk))
                    f.writelines('train_data : {}\n, valid_data : {}\n, killer_batch_size : {}\n, Nonkiller_batch_size : {}\n, used_TFrecord : {}\n, file_read_mode : {}\n, network: {}\n, learning_rate: {}\n, pos_weight : {}\n, target_leak_ratio : {}\n, batch_size : {}\n, epochs : {}\n'.format( self.train_path, self.test_path,self.Killer_batch_size, self.Nonkiller_batch_size,  self.self.tfrecord_filename, self.file_read_mode, self.network, self.lr, self.pos_weight, self.target_leak_ratio, self.batch_size, self.n_epochs))
                                
            else:  ## 새로 학습하는 경우
                print("<<<<<<< New training mode >>>>>>>")
                backup = None
                
                with open('checkpoints/' + self.model_nm + '/Modeling_information.txt', 'w') as f:
                    f.writelines('-----First train----- \n' )
                    f.writelines('  train Killer : {}\n, train NonKiller : {}\n, valid Killer : {}\n, valid NonKiller : {}\n, '.format(self.train_kl, self.train_nk, self.valid_kl, self.valid_nk))
                    f.writelines('train_data : {}\n, valid_data : {}\n, killer_batch_size : {}\n, Nonkiller_batch_size : {}\n, used_TFrecord : {}\n, file_read_mode : {}\n, network: {}\n, learning_rate: {}\n, pos_weight : {}\n, target_leak_ratio : {}\n, batch_size : {}\n, epochs : {}\n'.format( self.train_path, self.test_path, self.Killer_batch_size, self.Nonkiller_batch_size,  self.tfrecord_filename, self.file_read_mode, self.network, self.lr, self.pos_weight, self.target_leak_ratio, self.batch_size, self.n_epochs))
            step = self.gstep.eval()

            for epoch in range(self.n_epochs):
                print("<<<<<<< Epoch %d / %d Training started >>>>>>>" % (epoch+1, self.n_epochs))
                step = self.train_one_epoch(sess, saver, writer, epoch, step, save_path) #self.train_init
                print('epoch : {} '.format(epoch))
                print('evaluation start')
                self.eval_once(sess, writer, epoch, step)# self.test_init
            
            if backup is not None:
                with open(directory, 'r') as f1:
                    add = f1.readlines()
                with open(directory, 'w') as f2:
                    f2.writelines(add[0])
                    for b in backup[1:]:
                        f2.writelines(b)
                    for a in add[1:]:
                        f2.writelines(a)
        
        print("<<<<<<< Training finished >>>>>>>")
        total_time = time.time() - start_time + self.build_time
        print('Total Training time: {0} seconds'.format(total_time))
        with open('checkpoints/' + self.model_nm + '/Modeling_information.txt', 'a') as f:
            f.writelines(", build time : {}\n, train time : {}\n, total time : {}\n ".format(self.build_time, total_time - self.build_time, total_time))
            
        writer.close()

        
