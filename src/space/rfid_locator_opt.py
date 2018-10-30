'''
Created on Oct 29, 2018

@author: Inwoo Chung (gutomitai@gmail.com)
'''

import sys
import numpy as np
import os
import argparse

import pandas as pd
import time

num_iter = 1200

def calProposalDistVal(pre, cur, cov):
    '''
        Calculate a proposal distribution value.
        @param pre:
        @param cur:
        @param cov:
    '''
    
    # Calculate the numerator and denominator.
    numerator = np.exp(-0.5 * np.dot(np.dot((cur - pre).T, np.linalg.inv(cov)), (cur - pre)))
    denominator = np.sqrt(np.power(2.0 * np.pi, 2.0) * np.linalg.det(cov))
    
    return numerator / denominator

def optimizeViaSA(rawDataPath, hps, modelLoading):
        
    # Search for hyper-parameters optimizing score.
    np.random.seed(int(time.time()))
    t0 = 1.0 #?
    tf = 0.01 #?
    
    numHps = 2
    covar = np.identity(numHps, dtype=np.float32) * t0 #?
    mean = np.zeros(numHps, dtype=np.float32)
        
    # Initial state.
    hpState = np.random.multivariate_normal(mean, covar)
    
    # dense1_dim: 8 ~ 1024.
    hps['dense1_dim'] = int(np.maximum(np.minimum(np.abs(hpState[0])*400, 1024), 8))
    
    # num_layer: 1 ~ 12.
    hps['num_layer'] = int(np.maximum(np.minimum(np.abs(hpState[1])*4, 12), 1))
                    
    preHpState = hpState
        
    # Train.
    argvText = '--mode' + ' ' + 'train' + ' '
    argvText += '--raw_data_path' + ' ' + rawDataPath + ' '
    argvText += '--num_layers' + ' ' + str(hps['num_layers']) + ' '
    argvText += '--dense1_dim' + ' ' + str(hps['dense1_dim']) + ' '
    argvText += '--dropout1_rate' + ' ' + str(hps['dropout1_rate']) + ' '
    argvText += '--lr' + ' ' + str(hps['lr']) + ' '
    argvText += '--beta_1' + ' ' + str(hps['beta_1']) + ' '
    argvText += '--beta_2' + ' ' + str(hps['beta_2']) + ' '
    argvText += '--decay' + ' ' + str(hps['decay']) + ' '
    argvText += '--epochs' + ' ' + str(hps['epochs']) + ' '
    argvText += '--batch_size' + ' ' + str(hps['batch_size'])
    argvText += '--val_ratio' + ' ' + str(hps['val_ratio']) + ' '
    argvText += '--model_load' + ' ' + str(modelLoading)
    
    command = 'python rfid_locator.py ' + argvText
    print(command)

    os.system(command) 
                 
    # Evaluate and get previous score.
    argvText = '--mode' + ' ' + 'evaluate' + ' '
    argvText += '--raw_data_path' + ' ' + rawDataPath + ' '
    argvText += '--num_layers' + ' ' + str(hps['num_layers']) + ' '
    argvText += '--dense1_dim' + ' ' + str(hps['dense1_dim']) + ' '
    argvText += '--dropout1_rate' + ' ' + str(hps['dropout1_rate']) + ' '
    argvText += '--lr' + ' ' + str(hps['lr']) + ' '
    argvText += '--beta_1' + ' ' + str(hps['beta_1']) + ' '
    argvText += '--beta_2' + ' ' + str(hps['beta_2']) + ' '
    argvText += '--decay' + ' ' + str(hps['decay']) + ' '
    argvText += '--epochs' + ' ' + str(hps['epochs']) + ' '
    argvText += '--batch_size' + ' ' + str(hps['batch_size'])
    argvText += '--val_ratio' + ' ' + str(hps['val_ratio']) + ' '
    argvText += '--model_load' + ' ' + str(modelLoading)
    
    command = 'python rfid_locator.py ' + argvText
    print(command)

    os.system(command) 

    # Read evaluation score.
    scoresDF = pd.read_csv('final_scores.csv', header=None)
    preScore = scoresDF.iloc[0, 0] + 1.0e-7

    scores = [preScore]
    dense1_dims = [hps['dense1_dim']]
    num_layers_list = [hps['num_layers']]
      
    for i in range(hps['num_iter']):
        print(i)
    
        t = t0 * np.power((tf / t0), i/hps['num_iter'])
        covar = np.identity(numHps, dtype=np.float32) * t #?
    
        # Sample a current state and get a current score.
        proHpState = np.random.multivariate_normal(preHpState, covar)
        
        # dense1_dim: 8 ~ 1024.
        hps['dense1_dim'] = int(np.maximum(np.minimum(np.abs(proHpState[0])*400, 1024), 8))
        
        # num_layer: 1 ~ 12.
        hps['num_layer'] = int(np.maximum(np.minimum(np.abs(proHpState[1])*4, 12), 1))
                        
        preHpState = proHpState
            
        # Train.
        argvText = '--mode' + ' ' + 'train' + ' '
        argvText += '--raw_data_path' + ' ' + rawDataPath + ' '
        argvText += '--num_layers' + ' ' + str(hps['num_layers']) + ' '
        argvText += '--dense1_dim' + ' ' + str(hps['dense1_dim']) + ' '
        argvText += '--dropout1_rate' + ' ' + str(hps['dropout1_rate']) + ' '
        argvText += '--lr' + ' ' + str(hps['lr']) + ' '
        argvText += '--beta_1' + ' ' + str(hps['beta_1']) + ' '
        argvText += '--beta_2' + ' ' + str(hps['beta_2']) + ' '
        argvText += '--decay' + ' ' + str(hps['decay']) + ' '
        argvText += '--epochs' + ' ' + str(hps['epochs']) + ' '
        argvText += '--batch_size' + ' ' + str(hps['batch_size'])
        argvText += '--val_ratio' + ' ' + str(hps['val_ratio']) + ' '
        argvText += '--model_load' + ' ' + str(modelLoading)
        
        command = 'python rfid_locator.py ' + argvText
        print(command)
    
        os.system(command) 
                     
        # Evaluate and get previous score.
        argvText = '--mode' + ' ' + 'evaluate' + ' '
        argvText += '--raw_data_path' + ' ' + rawDataPath + ' '
        argvText += '--num_layers' + ' ' + str(hps['num_layers']) + ' '
        argvText += '--dense1_dim' + ' ' + str(hps['dense1_dim']) + ' '
        argvText += '--dropout1_rate' + ' ' + str(hps['dropout1_rate']) + ' '
        argvText += '--lr' + ' ' + str(hps['lr']) + ' '
        argvText += '--beta_1' + ' ' + str(hps['beta_1']) + ' '
        argvText += '--beta_2' + ' ' + str(hps['beta_2']) + ' '
        argvText += '--decay' + ' ' + str(hps['decay']) + ' '
        argvText += '--epochs' + ' ' + str(hps['epochs']) + ' '
        argvText += '--batch_size' + ' ' + str(hps['batch_size'])
        argvText += '--val_ratio' + ' ' + str(hps['val_ratio']) + ' '
        argvText += '--model_load' + ' ' + str(modelLoading)
        
        command = 'python rfid_locator.py ' + argvText
        print(command)
    
        os.system(command) 
    
        # Read evaluation score.
        scoresDF = pd.read_csv('final_scores.csv', header=None)
        curScore = scoresDF.iloc[0, 0] + 1.0e-7
    
        # Check exception.
        if curScore == 0.0:
            continue
    
        u = np.random.rand()
    
        vSA = np.minimum(1.0, (np.power(curScore,1.0/t)*calProposalDistVal(preHpState, proHpState, covar))
                         /(np.power(preScore,1.0/t)*calProposalDistVal(proHpState, preHpState, covar))) #?
        
        if u <= vSA:
            preHpState = proHpState
        
        print(curScore)
        scores.append(curScore)
        dense1_dims.append(hps['dense1_dim'])
        num_layers_list.append(hps['num_layers'])

    # Save scores.
    scores = pd.DataFrame({'scores':scores, 'dense1_dim': dense1_dims, 'num_layers': num_layers_list})
    scores.to_csv('evaluation_scores.csv')
                                    
def main(args):
    '''
        Main.
        @param args: Arguments.
    '''
    
    hps = {}
    
    # Get arguments.
    rawDataPath = args.raw_data_path
    
    # hps.
    hps['num_layers'] = int(args.num_layers)
    hps['dense1_dim'] = int(args.dense1_dim)
    hps['dropout1_rate'] = float(args.dropout1_rate)
    hps['lr'] = float(args.lr)
    hps['beta_1'] = float(args.beta_1)
    hps['beta_2'] = float(args.beta_2)
    hps['decay'] = float(args.decay) 
    hps['epochs'] = int(args.epochs) 
    hps['batch_size'] = int(args.batch_size) 
    hps['val_ratio'] = float(args.val_ratio)
    
    modelLoading = int(args.model_load)       
    
    hps['num_iter'] = int(args.num_iter)
    
    # Optimize the model.
    optimizeViaSA(rawDataPath, hps, modelLoading)
            
if __name__ == '__main__':
    
    # Parse arguments.
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode')
    parser.add_argument('--raw_data_path')
    parser.add_argument('--num_layers')
    parser.add_argument('--dense1_dim')
    parser.add_argument('--dropout1_rate')
    parser.add_argument('--lr')
    parser.add_argument('--beta_1')
    parser.add_argument('--beta_2')
    parser.add_argument('--decay')
    parser.add_argument('--epochs')
    parser.add_argument('--batch_size')
    parser.add_argument('--val_ratio')
    parser.add_argument('--model_load')
    parser.add_argument('--num_iter')
    args = parser.parse_args()
    
    main(args)