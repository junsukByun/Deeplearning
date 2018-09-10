# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import openpyxl

def matrix():
    if (df['pred'] == 1) & (df['true'] == 1):
        return TP
    elif (df['pred'] == 0) & (df['true'] == 1):
        return FN
    elif (df['pred'] == 1) & (df['true'] == 0):
        return FP
    elif (df['pred'] == 0) & (df['true'] == 0):
        return TN    


def AutoCalRatio(preds, labels, threshold, filenames, multicrop):       

    # 각 이미지별 예측
    prob = np.array(list(map(lambda x: x[1], preds))) # 양품 확률
    pred_c = np.array(list(map(lambda x:1 if x[1] > threshold else 0, preds)))    # 양품: 1, 불량: 0
    true = np.array([np.argmax(element) for element in labels]) 
#     print(len(true))
    
    filename_parsed = [('_').join(filename.split('_')[:-1]) for filename in filenames]
    if multicrop:
        # 이미지 group별 예측
        df_pred_true = pd.DataFrame({'filename_org': filenames, 'filename': filename_parsed, 'prob': prob, 'pred_c': pred_c, 'true': true})
        df_pred_true = df_pred_true.groupby("filename")[['pred_c', 'true']].min()    
        true = np.array(df_pred_true['true'])
        pred_c = np.array(df_pred_true['pred_c'])

    leak = len(np.where((true==0) & (pred_c==1))[0])   #양품: 1 
    leak_ratio = round(len(np.where((true==0) & (pred_c==1))[0]) / float(len(true)) * 100,2)
    true_ok = len(np.where((true==1) & (pred_c==1))[0])
    true_ok_ratio = round(len(np.where((true==1) & (pred_c==1))[0]) / float(len(true)) * 100,2)
    TP = int(len(np.where((true==1) & (pred_c==1))[0]))  #실제양품 - 양품예측
    FP = int(len(np.where((true==0) & (pred_c==1))[0]))  #실제불량 - 양품예측
    FN = int(len(np.where((true==1) & (pred_c==0))[0]))  #실제양품 - 불량예측
    TN = int(len(np.where((true==0) & (pred_c==0))[0]))  #실제불량 - 불량예측
    
    acc = float((TP+TN))/(TP+TN+FP+FN) * 100
    contribution = float((TP+FP))/(TP+TN+FP+FN) * 100
    pass_acc = float(TP)/(TP+FN) * 100
    fail_acc = float(TN)/(TN+FP) * 100

    return (acc, leak, true_ok, leak_ratio, contribution, pass_acc, fail_acc, TP, FN, FP, TN)


def AutoFinder(preds, labels, filenames, multicrop, target_leak = 0.8):
    df = pd.DataFrame({'threshold': [0], 'contribution':[0], 'pass_acc':[0], 'fail_acc':[0], 'accuracy': [0], 'leak_ratio': [0], 
                       'leak': [0], 'TP':[0], 'FN':[0], 'FP':[0], 'TN':[0]})

    for i in list(np.arange(0.1, 1, 0.001)):      
        acc, leak, true_ok, leak_ratio,contribution, pass_acc, fail_acc, TP, FN, FP, TN = AutoCalRatio(preds, labels, i, filenames, multicrop)            
        df_element = pd.DataFrame({'threshold': [i], 'contribution':[contribution], 'pass_acc':[pass_acc], 'fail_acc':[fail_acc], 'accuracy': [acc], 
                                   'leak_ratio': [leak_ratio], 'leak': [leak], 'TP':[TP], 'FN':[FN], 'FP':[FP], 'TN':[TN]})
        df = pd.concat([df, df_element])   

    df = df.iloc[1:, ]
    try:
        diff_min = 0.5    
        for val in list(df['leak_ratio']):
            diff = abs(val- target_leak)
            if diff <= diff_min:
                closest_val = val
                diff_min = diff
        df_target = df[df['leak_ratio']== closest_val].iloc[0,]
    
    except:
        df_target = df[df['leak_ratio']== df['leak_ratio'].min()].iloc[0,]
        print("No closet values (diff: under 0.5) exist")
       

    return df, df_target