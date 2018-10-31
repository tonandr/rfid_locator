'''
Created on Oct 17, 2018

@author: Inwoo Chung (gutomitai@gmail.com)
'''

import os
import argparse
import time

import pandas as pd
import numpy as np

from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras import optimizers
from keras.utils import multi_gpu_model
import keras

import ipyparallel as ipp

# Constant.
TIME_SLOT = pd.Timedelta(1, unit='h')

CATEGORY_NONE = -1
CATEGORY_MARKER = 0
CATEGORY_COMMUNITY = 1
CATEGORY_CTB = 2

CONFIDENCE_MARKER = 6.
CONFIDENCE_COMMUNITY_RATIO = 0.1
CONFIDENCE_CTB = 12.

RMIN_MARKER = 48.0
RMIN_COMMUNITY = 48.0
RMIN_CTB = 12.0

IS_MULTI_GPU = False
NUM_GPUS = 4

IS_DEBUG = False

markerLocRefDF = None
commLocRefDF = None
ctbLocRefDF = None

def createTrValData(rawDataPath):
    '''
        Create training and validation data.
        @param rawDataPath: Raw data path.
    '''
    
    global markerLocRefDF, commLocRefDF, ctbLocRefDF
    
    # Make location references by each category such as the marker, community, CTB.
    # Marker.
    markerLocRefDF = pd.read_csv(os.path.join(rawDataPath, 'misc', 'marker_tag_locations.txt'))
    
    # Community.
    commLocRef = []
    commLocRawDF = pd.read_csv(os.path.join(rawDataPath, 'misc', 'community_locations.txt'))
    
    # Get tag ids for each community.
    tagsByComm = {}
    with open(os.path.join(rawDataPath, 'misc', 'communities.txt'), 'r') as f:
        lines = f.readlines()
        
        for i in range(int(len(lines)/2)):
            commName = lines[i*2][0:-1]
            tags = [int(v) for v in lines[i*2+1][1:-2].split(',')]
            
            tagsByComm[commName] = tags
    
    # Create the location table for each tag.
    for commName in tagsByComm:
        df = commLocRawDF[commLocRawDF.id == commName]
        tags = tagsByComm[commName]
        
        for tag_id in tags:
        
            # Get location.                
            commLocRef.append({'epc_id': tag_id
                               , 'x': df.x.iloc[0]
                               , 'y': df.y.iloc[0]
                               , 'z': df.z.iloc[0]})
    
    commLocRefDF = pd.DataFrame(commLocRef)
    
    # CTB.
    ctbLocRefDF = pd.DataFrame(columns = ['epc_id','x', 'y', 'z', 'start', 'end'])
    ctbLocRawDF = pd.read_csv(os.path.join(rawDataPath, 'misc', 'ctb_locations.txt'))
    
    # Get tag ids for each ctb and time slot.
    tagsByctb = {}
    with open(os.path.join(rawDataPath, 'misc', 'ctb_tags.txt'), 'r') as f:
        lines = f.readlines()
        
        for i in range(int(len(lines)/2)):
            ctbName = lines[i*2][0:-1]
            tags = [int(v) for v in lines[i*2+1][1:-2].split(',')]
            
            tagsByctb[ctbName] = tags
    
    # Create the location and time slot table for each tag.
    for ctbName in tagsByctb:
        tags = tagsByctb[ctbName]
        
        for tag_id in tags:
            
            # Get location and time slot.
            df = ctbLocRawDF[ctbLocRawDF.id == int(ctbName)]
            df = df[['x', 'y', 'z', 'start', 'end']]
            df.index = [idx for idx in range(df.shape[0])]
            df = pd.concat([pd.Series([tag_id for _ in range(df.shape[0])], name='epc_id'), df], axis = 1)
            ctbLocRefDF = ctbLocRefDF.append(df)
    
    # Create the location table for each sample tag.
    trValDataDF = pd.DataFrame(columns = ['category'
                                          , 'epc_id'
                                          , 'start'
                                          , 'end'
                                          , 'a1_id'
                                          , 'a2_id'
                                          , 'a3_id'
                                          , 'rssi1'
                                          , 'rssi2'
                                          , 'rssi3'
                                          , 'x'
                                          , 'y'
                                          , 'z'])
    
    vals = getTrainRawDFGList(rawDataPath)
    
    '''
    pClient = ipp.Client()
    pView = pClient[:]
    
    pView.push({'markerLocRefDF': markerLocRefDF, 'commLocRawDF': commLocRawDF, 'ctbLocRefDF': ctbLocRefDF})

    trValDataDFs = pView.map(calLocationTableForEachTag, vals, block=True)    
    '''
    
    trValDataDFs = []
    
    for val in vals:
        ft = time.time()
        trValDataDFs.append(calLocationTableForEachTag(val))
        et = time.time()
        
        print(val[1], val[2], et - ft)
        
    trValDataDF = trValDataDFs[0]
    
    for i in range(1, len(trValDataDFs)):
        trValDataDF = trValDataDF.append(trValDataDFs[i])
        
    # Save training data.
    trValDataDF.to_csv('train.csv')    

def getTrainRawDFGList(rawDataPath):
    '''
        Get train raw DFG list.
        @param rawDataPath: Raw data path.
    '''
    
    trainRawDFGList = []

    trainRawDF = pd.read_csv(os.path.join(rawDataPath, 'rfid_raw_readings_train.txt'), sep='\t')
    #trainRawDF = pd.read_csv(os.path.join(rawDataPath, 'rfid_raw_task-example.txt'), sep='\t')
    
    # Group data by tag ids.
    trainRawDFG = trainRawDF.groupby('epc_id')
    tagIds = list(trainRawDFG.groups.keys())
    numTagIds = len(tagIds)
    
    # Get group list.
    for tagId in tagIds:
        trainRawDFGList.append(trainRawDFG.get_group(tagId))
    
    vals = []
    
    for v in zip(trainRawDFGList, tagIds, range(numTagIds)):
        vals.append(v)
    
    return vals

def calLocationTableForEachTag(val):
    '''
        Calculate the location table for each tag.
        @param val: Train raw DF, tag id, tag id number.
    '''
    
    global markerLocRefDF, commLocRefDF, ctbLocRefDF
    
    rawDF = val[0]
    tagId = val[1]
    tagIdNum = val[2]
    
    trValDataDF = pd.DataFrame(columns = ['category'
                                          , 'epc_id'
                                          , 'start'
                                          , 'end'
                                          , 'a1_id'
                                          , 'a2_id'
                                          , 'a3_id'
                                          , 'rssi1'
                                          , 'rssi2'
                                          , 'rssi3'
                                          , 'x'
                                          , 'y'
                                          , 'z'])

    rawDF = rawDF.sort_values(by = 'date')
    rawDF.index = pd.to_datetime(rawDF.date)
    rawDF = rawDF[['epc_id', 'antenna_id', 'rssi', 'freq', 'phase', 'power', 'cnt']]
    
    # Check a valid tag id.
    # Marker.
    resDF = markerLocRefDF[markerLocRefDF.epc_id == tagId]
    
    if resDF.shape[0] != 0:
        category = CATEGORY_MARKER
        x, y, z = resDF.x.iloc[0], resDF.y.iloc[0], resDF.z.iloc[0]
    else:
        
        # Community.
        resDF = commLocRefDF[commLocRefDF.epc_id == tagId]
        
        if resDF.shape[0] != 0:
            category = CATEGORY_COMMUNITY
            x, y, z = resDF.x.iloc[0], resDF.y.iloc[0], resDF.z.iloc[0]
        else:
            
            # CTB.
            resDF = ctbLocRefDF[ctbLocRefDF.epc_id == tagId]
            
            if resDF.shape[0] != 0:
                
                # Extract data relevant to allowable time slots.
                if tagId == 13127:
                    rawDF = rawDF[pd.Timestamp(resDF.start.iloc[0]):pd.Timestamp(resDF.end.iloc[0])]
                    
                    # Check exception.
                    if rawDF.shape[0] == 0:
                        return trValDataDF
                    
                    category = CATEGORY_CTB
                    x, y, z = resDF.x.iloc[0], resDF.y.iloc[0], resDF.z.iloc[0]
                else:
                    rawDFs, xs, ys, zs = [], [], [], []
                    category = CATEGORY_CTB
                    
                    for i in range(3):
                        rawDFUnit = rawDF[pd.Timestamp(resDF.start.iloc[i]):pd.Timestamp(resDF.end.iloc[i])]
                                                    
                        rawDFs.append(rawDFUnit)
                        x, y, z = resDF.x.iloc[i], resDF.y.iloc[i], resDF.z.iloc[i]                            
                        xs.append(x)
                        ys.append(y)
                        zs.append(z)
                    
                    # Check exception.
                    if (rawDFs[0].shape[0] == 0) and (rawDFs[1].shape[0] == 0) and (rawDFs[2].shape[0] == 0):
                        return trValDataDF
            else:
                # Non-category?                    
                return trValDataDF         
    
    if (tagId != 13127) and (category == CATEGORY_CTB):            
        for rawDF, x, y, z in zip(rawDFs, xs, ys, zs):
            
            # Check exception.
            if rawDF.shape[0] == 0: continue
            
            st = rawDF.index[0]
            et = rawDF.index[-1]
            
            ist = st
            iet = ist + TIME_SLOT        
            
            while ist < et:
                rawDF_ts = rawDF[ist:iet]
                rawDF_ts_g = rawDF_ts.groupby('antenna_id')
                
                rssi_by_aid = {}
                
                for aid in list(rawDF_ts_g.groups.keys()):
                    df = rawDF_ts_g.get_group(aid)
                    rssi = df.rssi.median() #?
                    rssi_by_aid[aid] = rssi
                
                aids = list(rssi_by_aid.keys())
                
                for f_id, f_aid in enumerate(aids):
                    for s_id, s_aid in enumerate(aids):
                        if s_id <= f_id: continue
                        
                        for t_id, t_aid in enumerate(aids):
                            if t_id <= s_id: continue
                                                    
                            trValData = {'category': [category]
                                              , 'epc_id': [tagId]
                                              , 'start': [ist]
                                              , 'end': [iet]
                                              , 'a1_id': [f_aid]
                                              , 'a2_id': [s_aid]
                                              , 'a3_id': [t_aid]
                                              , 'rssi1': [rssi_by_aid[f_aid]]
                                              , 'rssi2': [rssi_by_aid[s_aid]]
                                              , 'rssi3': [rssi_by_aid[t_aid]]
                                              , 'x': [x]
                                              , 'y': [y]
                                              , 'z': [z]}
                            
                            trValDataDF = trValDataDF.append(pd.DataFrame(trValData))
                
                ist = iet
                iet = ist + TIME_SLOT                
    else:        
        st = rawDF.index[0]
        et = rawDF.index[-1]
        
        ist = st
        iet = ist + TIME_SLOT        
        
        while ist < et:
            rawDF_ts = rawDF[ist:iet]
            rawDF_ts_g = rawDF_ts.groupby('antenna_id')
            
            rssi_by_aid = {}
            
            for aid in list(rawDF_ts_g.groups.keys()):
                df = rawDF_ts_g.get_group(aid)
                rssi = df.rssi.median() #?
                rssi_by_aid[aid] = rssi
            
            aids = list(rssi_by_aid.keys())
            
            for f_id, f_aid in enumerate(aids):
                for s_id, s_aid in enumerate(aids):
                    if s_id <= f_id: continue
                    
                    for t_id, t_aid in enumerate(aids):
                        if t_id <= s_id: continue
                                                
                        trValData = {'category': [category]
                                          , 'epc_id': [tagId]
                                          , 'start': [ist]
                                          , 'end': [iet]
                                          , 'a1_id': [f_aid]
                                          , 'a2_id': [s_aid]
                                          , 'a3_id': [t_aid]
                                          , 'rssi1': [rssi_by_aid[f_aid]]
                                          , 'rssi2': [rssi_by_aid[s_aid]]
                                          , 'rssi3': [rssi_by_aid[t_aid]]
                                          , 'x': [x]
                                          , 'y': [y]
                                          , 'z': [z]}
                        
                        trValDataDF = trValDataDF.append(pd.DataFrame(trValData))
            
            ist = iet
            iet = ist + TIME_SLOT 
        
    return trValDataDF

class ISSRFIDLocator(object):
    '''
        ISS rfid locator.
    '''
    
    def __init__(self, rawDataPath):
        '''
            Constructor.
            @param rawDataPath: Raw data path.
        '''
        
        self.rawDataPath = rawDataPath
        
        # Initialize.
        # Create 3 antenna combinations.
        self.aids = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
        self.combs = []
        
        for f_id, f_aid in enumerate(self.aids):
            for s_id, s_aid in enumerate(self.aids):
                if s_id <= f_id: continue
                
                for t_id, t_aid in enumerate(self.aids):
                    if t_id <= s_id: continue
                    self.combs.append((f_aid, s_aid, t_aid))

        # Load antenna information.
        self.antennaInfoDF = pd.read_csv(os.path.join(rawDataPath, 'misc', 'antenna_locations.csv'), header=0)
        self.antennaInfoDF = self.antennaInfoDF.iloc[1:,:]
        self.antennaInfoDF.index = self.antennaInfoDF.id
        self.antennaInfoDF = self.antennaInfoDF[['reader_id', 'antenna_id', 'location_x', 'location_y', 'location_z']]  

        # Make location references by each category such as the marker, community, CTB.
        # Marker.
        self.markerLocRefDF = pd.read_csv(os.path.join(self.rawDataPath, 'misc', 'marker_tag_locations.txt'))
        self.markerIds =list(self.markerLocRefDF.epc_id) #?
        
        # Community.
        commLocRef = []
        commLocRawDF = pd.read_csv(os.path.join(rawDataPath, 'misc', 'community_locations.txt'))
        
        # Get tag ids for each community.
        self.tagsByComm = {}
        with open(os.path.join(rawDataPath, 'misc', 'communities.txt'), 'r') as f:
            lines = f.readlines()
            
            for i in range(int(len(lines)/2)):
                commName = lines[i*2][0:-1]
                tags = [int(v) for v in lines[i*2+1][1:-2].split(',')]
                
                self.tagsByComm[commName] = tags
        
        # Create the location table for each tag.
        for commName in self.tagsByComm:
            df = commLocRawDF[commLocRawDF.id == commName]
            tags = self.tagsByComm[commName]
            
            for tag_id in tags:
                
                # Get a location and confidence.                
                cr = CONFIDENCE_COMMUNITY_RATIO \
                    *np.median([np.sqrt(np.power(df.x.iloc[0] - float(self.antennaInfoDF.iloc[i].location_x), 2.0) \
                                          + np.power(df.y.iloc[0] - float(self.antennaInfoDF.iloc[i].location_y), 2.0) \
                                          + np.power(df.z.iloc[0] - float(self.antennaInfoDF.iloc[i].location_z), 2.0)) \
                                          for i in range(24)])
                
                commLocRef.append({'epc_id': [tag_id]
                                   , 'x': [df.x.iloc[0]]
                                   , 'y': [df.y.iloc[0]]
                                   , 'z': [df.z.iloc[0]]
                                   , 'cr': [cr]})
        
        self.commLocRefDF = pd.DataFrame(commLocRef)

        # CTB.
        self.ctbLocRefDF = pd.DataFrame(columns = ['epc_id','x', 'y', 'z', 'start', 'end'])
        ctbLocRawDF = pd.read_csv(os.path.join(rawDataPath, 'misc', 'ctb_locations.txt'))
        
        # Get tag ids for each ctb and time slot.
        tagsByctb = {}
        with open(os.path.join(rawDataPath, 'misc', 'ctb_tags.txt'), 'r') as f:
            lines = f.readlines()
            
            for i in range(int(len(lines)/2)):
                ctbName = lines[i*2][0:-1]
                tags = [int(v) for v in lines[i*2+1][1:-2].split(',')]
                
                tagsByctb[ctbName] = tags
        
        # Create the location and time slot table for each tag.
        for ctbName in tagsByctb:
            tags = tagsByctb[ctbName]
            
            for tag_id in tags:
                
                # Get location and time slot.
                df = ctbLocRawDF[ctbLocRawDF.id == int(ctbName)]
                df = df[['x', 'y', 'z', 'start', 'end']]
                df.index = [idx for idx in range(df.shape[0])]
                df = pd.concat([pd.Series([tag_id for _ in range(df.shape[0])], name='epc_id'), df], axis = 1)
                self.ctbLocRefDF = self.ctbLocRefDF.append(df)
                
    def train(self, hps, modelLoading = False):
        '''
            Train.
            @param hps: Hyper-parameters.
            @param modelLoading: Model loading flag.
        '''
        
        self.hps = hps
        
        if modelLoading == True:
            print('Load the pre-trained model...')
            
            if IS_MULTI_GPU == True:
                self.model = multi_gpu_model(keras.models.load_model('iss_rfid_locator.h5'), gpus = NUM_GPUS) 
            else:

                self.model = keras.models.load_model('iss_rfid_locator.h5')
        else:
            
            # Design the model.
            print('Design the model.')
            
            # Input: 2024 x 3 (3 antenna combinations) 
            input = Input(shape=(2024, 3))
            x = Dense(self.hps['dense1_dim'], activation='relu', use_bias=False, name='dense1_' + str(0))(input) #?
            
            for i in range(1, hps['num_layers']):
                x = Dense(self.hps['dense1_dim'], activation='relu', use_bias=False, name='dense1_' + str(i))(x) #?
            
            output = Dense(3, activation='linear', use_bias=False, name='dense1_last')(x)
                                            
            # Create the model.
            if IS_MULTI_GPU == True:
                self.model = multi_gpu_model(Model(inputs=[input]
                                                   , outputs=[output]), gpus = NUM_GPUS)
            else:
                self.model = Model(inputs=[input], outputs=[output])  
        
        # Compile the model.
        optimizer = optimizers.Adam(lr=self.hps['lr']
                                    , beta_1=self.hps['beta_1']
                                    , beta_2=self.hps['beta_2']
                                    , decay=self.hps['decay'])
        self.model.compile(optimizer=optimizer, loss='mse')
        self.model.summary()        
        
        # Create training and validation data.
        tr, val = self.__createTrValData__(hps)
        rssiM, posM = tr
        
        # Train the model.
        hists = []
        
        hist = self.model.fit([rssiM], [posM]
                      , epochs=self.hps['epochs']
                      , batch_size=self.hps['batch_size']
                      , verbose=1)
            
        hists.append(hist)
            
        # Print loss.
        print(hist.history['loss'][-1])
        
        print('Save the model.')            
        self.model.save('iss_rfid_locator.h5')
        
        # Calculate loss.
        lossList = list()
        
        for h in hists:
            lossList.append(h.history['loss'][-1])
            
        lossArray = np.asarray(lossList)        
        lossMean = lossArray.mean()
        
        print('Each mean loss: {0:f} \n'.format(lossMean))
        
        with open('losses.csv', 'a') as f:
            f.write('{0:f} \n'.format(lossMean))
                
        with open('loss.csv', 'w') as f:
            f.write(str(lossMean) + '\n') #?
            
        return lossMean                
        
    def __createTrValData__(self, hps):
        '''
            Create training and validation data.
            @param hps: Hyper-parameters.
        '''
        
        # Load raw data.
        rawDatasDF = pd.read_csv('train.csv').groupby('category')
        rawDatasDF = rawDatasDF.get_group(0)

        # Training data.
        trRawDatasDF = rawDatasDF.iloc[:int(rawDatasDF.shape[0]*(1.0 - hps['val_ratio'])), :]
        
        # Make the rssi value matrix and x, y, z matrix according to 3 antenna combinations.
        rssiRawM = np.ndarray(shape=(31, 31, 31), dtype=np.object)
        posRawM = np.ndarray(shape=(31, 31, 31), dtype=np.object)
        
        for i in range(31):
            for j in range(31):
                for k in range(31):
                    rssiRawM[i,j,k] = []
                    posRawM[i,j,k] = []
        
        for i in range(trRawDatasDF.shape[0]):
            rawDataDF = trRawDatasDF.iloc[i, :]
            
            rssiRawM[rawDataDF.a1_id, rawDataDF.a2_id, rawDataDF.a3_id]\
            .append((rawDataDF.rssi1, rawDataDF.rssi2, rawDataDF.rssi3))
            
            posRawM[rawDataDF.a1_id, rawDataDF.a2_id, rawDataDF.a3_id]\
            .append((rawDataDF.x, rawDataDF.y, rawDataDF.z))
                
        # Extract data relevant to 3 antenna combinations.
        rssiM = np.ndarray(shape=(2024), dtype=np.object)
        posM = np.ndarray(shape=(2024), dtype=np.object)                                                                                               
        
        for i in range(2024):
            rssiM[i] = np.asarray(rssiRawM[self.combs[i]])
            posM[i] = np.asarray(posRawM[self.combs[i]])
            
        # Adjust batch size to same size.
        # Get maximum batch length.
        lengths = [rssiM[i].shape[0] for i in range(2024)]
        maxLength = lengths[np.argmax(lengths)]
        rssiMList = []
        posMList = []
            
        for i in range(2024):
            
            # Check and process exception.
            if rssiM[i].shape[0] == 0:
                rssiMList.append(np.zeros(shape=(maxLength, 3)).T)
                posMList.append(np.zeros(shape=(maxLength, 3)).T) 
            else:
                rssiMList.append(np.concatenate([rssiM[i]
                                                 , np.repeat(rssiM[i][-1,:][np.newaxis, :]
                                                             , (maxLength - rssiM[i].shape[0])
                                                             , axis=0)]).T)
                posMList.append(np.concatenate([posM[i]
                                                , np.repeat(posM[i][-1,:][np.newaxis, :]
                                                            , (maxLength - rssiM[i].shape[0])
                                                            , axis=0)]).T)
        
        rssiM = np.asarray(rssiMList)
        posM = np.asarray(posMList)
        
        rssiM = np.asarray([rssiM[:,:,i] for i in range(maxLength)])
        posM = np.asarray([posM[:,:,i] for i in range(maxLength)])
        
        # Save data.
        rssiM.tofile('rssiM_tr.nd')
        posM.tofile('posM_tr.nd')
        
        #rssiM = np.fromfile('rssiM_tr.nd').reshape((5028, 2024, 3))
        #posM = np.fromfile('posM_tr.nd').reshape((5028, 2024, 3))
        
        tr = (rssiM, posM)
        
        # Validation data.
        valRawDatasDF = rawDatasDF.iloc[int(rawDatasDF.shape[0]*(1.0 - hps['val_ratio'])):, :]
        
        # Make the rssi value matrix and x, y, z matrix according to 3 antenna combinations.
        rssiRawM = np.ndarray(shape=(31, 31, 31), dtype=np.object)
        posRawM = np.ndarray(shape=(31, 31, 31), dtype=np.object)
        
        for i in range(31):
            for j in range(31):
                for k in range(31):
                    rssiRawM[i,j,k] = []
                    posRawM[i,j,k] = []
        
        for i in range(valRawDatasDF.shape[0]):
            rawDataDF = valRawDatasDF.iloc[i, :]
            
            rssiRawM[rawDataDF.a1_id, rawDataDF.a2_id, rawDataDF.a3_id]\
            .append((rawDataDF.rssi1, rawDataDF.rssi2, rawDataDF.rssi3))
            
            posRawM[rawDataDF.a1_id, rawDataDF.a2_id, rawDataDF.a3_id]\
            .append((rawDataDF.x, rawDataDF.y, rawDataDF.z))
        
        # Extract data relevant to 3 antenna combinations.
        rssiM = np.ndarray(shape=(2024), dtype=np.object)
        posM = np.ndarray(shape=(2024), dtype=np.object)                                                                                               
        
        for i in range(2024):
            rssiM[i] = np.asarray(rssiRawM[self.combs[i]])
            posM[i] = np.asarray(posRawM[self.combs[i]])
            
        # Adjust batch size to same size.
        # Get maximum batch length.
        lengths = [rssiM[i].shape[0] for i in range(2024)]
        maxLength = lengths[np.argmax(lengths)]
        rssiMList = []
        posMList = []
            
        for i in range(2024):
            
            # Check and process exception.
            if rssiM[i].shape[0] == 0:
                rssiMList.append(np.zeros(shape=(maxLength, 3)).T)
                posMList.append(np.zeros(shape=(maxLength, 3)).T) 
            else:
                rssiMList.append(np.concatenate([rssiM[i]
                                                 , np.repeat(rssiM[i][-1,:][np.newaxis, :]
                                                             , (maxLength - rssiM[i].shape[0])
                                                             , axis=0)]).T)
                posMList.append(np.concatenate([posM[i]
                                                , np.repeat(posM[i][-1,:][np.newaxis, :]
                                                            , (maxLength - rssiM[i].shape[0])
                                                            , axis=0)]).T)
        
        rssiM = np.asarray(rssiMList)
        posM = np.asarray(posMList)
        
        rssiM = np.asarray([rssiM[:,:,i] for i in range(maxLength)])
        posM = np.asarray([posM[:,:,i] for i in range(maxLength)])
        
        # Save data.
        rssiM.tofile('rssiM_val.nd')
        posM.tofile('posM_val.nd')
        
        #rssiM = np.fromfile('rssiM_val.nd').reshape((948, 2024, 3))
        #posM = np.fromfile('posM_val.nd').reshape((948, 2024, 3))
        
        val = (rssiM, posM)        
        
        return tr, val
    
    def evaluate(self, hps, modelLoading = True):
        '''
            Evaluate the performance of the iss rfid locator.
            @param hps: Hyper-parameters.
            @param modelLoading: Model loading flag.
        '''
        
        self.hps = hps
        
        if modelLoading == True:
            print('Load the pre-trained model...')
            
            if IS_MULTI_GPU == True:
                self.model = multi_gpu_model(keras.models.load_model('iss_rfid_locator.h5'), gpus = NUM_GPUS) 
            else:

                self.model = keras.models.load_model('iss_rfid_locator.h5')
        
        # Load raw data.
        rawDatasDF = pd.read_csv('train.csv')

        # Calculate the bias calibration factor using marker tags.
        rawDatasDFG = rawDatasDF.groupby('epc_id')
        xOffs, yOffs, zOffs = [], [], []
        
        for id in self.markerIds:
            try:
                df = rawDatasDFG.get_group(id)
            except Exception:
                continue
            
            rssiVals = np.zeros(shape=(1,2024,3))
            valIndexes = set()
            
            for i in range(1, 4):
                dfG = df.groupby('a' + str(i) + '_id') # Antenna id?
                
                # Get rssi values.
                rssi_by_aid = {}
                
                for aid in list(dfG.groups.keys()):
                    df = dfG.get_group(aid)
                    rssi = df.loc[:, 'rssi' + str(i)].median() #?
                    rssi_by_aid[aid] = rssi
                
                aids = list(rssi_by_aid.keys())
                
                for f_id, f_aid in enumerate(aids):
                    for s_id, s_aid in enumerate(aids):
                        if s_id <= f_id: continue
                        
                        for t_id, t_aid in enumerate(aids):
                            if t_id <= s_id: continue        
                            
                            rssiVals[0, self.combs.index((f_aid, s_aid, t_aid)), 0] = rssi_by_aid[f_aid] # Last value?
                            rssiVals[0, self.combs.index((f_aid, s_aid, t_aid)), 1] = rssi_by_aid[s_aid]
                            rssiVals[0, self.combs.index((f_aid, s_aid, t_aid)), 2] = rssi_by_aid[t_aid]
                            
                            valIndexes.add(self.combs.index((f_aid, s_aid, t_aid)))
            
            # Check exception.
            if len(valIndexes) == 0:
                continue
            
            # Predict a position.
            valIndexes = list(valIndexes)
            pos = self.model.predict(rssiVals) # Dimension?
            x = np.median(pos[0, valIndexes, 0])
            y = np.median(pos[0, valIndexes, 1])
            z = np.median(pos[0, valIndexes, 2])
            
            # Calculate offset.
            resDF = self.markerLocRefDF[self.markerLocRefDF.epc_id == id]
            xt, yt, zt = resDF.x.iloc[0], resDF.y.iloc[0], resDF.z.iloc[0] #?
                        
            xOffs.append(xt - x)
            yOffs.append(yt - y)
            zOffs.append(zt - z)
            
            print(x, y, z, xt, yt, zt, xt - x, yt - y, zt - z)
        
        # Check exception.
        if len(xOffs) == 0:
            xOff, yOff, zOff = 0., 0., 0.
        
        xOff, yOff, zOff = (np.median(xOffs), np.median(yOffs), np.median(zOffs))  

        # Training data.
        trResults = []
        trRawDatasDF = rawDatasDF.iloc[:int(rawDatasDF.shape[0]*(1.0 - hps['val_ratio'])), :]
        
        # Group data according to tag id.
        trRawDatasDFG = trRawDatasDF.groupby('epc_id')
        tagIds = list(trRawDatasDFG.groups.keys())
                
        for i, id in enumerate(tagIds):
            
            # Predict a position and radius confidence.
            df = trRawDatasDFG.get_group(id)
            xt = df.x.iloc[0]
            yt = df.y.iloc[0]
            zt = df.z.iloc[0]
            
            rssiVals = np.zeros(shape=(1,2024,3))
            valIndexes = set()
            
            for i in range(1, 4):
                dfG = df.groupby('a' + str(i) + '_id') # Antenna id?
                
                # Get rssi values.
                rssi_by_aid = {}
                
                for aid in list(dfG.groups.keys()):
                    df = dfG.get_group(aid)
                    rssi = df.loc[:, 'rssi' + str(i)].median() #?
                    rssi_by_aid[aid] = rssi
                
                aids = list(rssi_by_aid.keys())
                
                for f_id, f_aid in enumerate(aids):
                    for s_id, s_aid in enumerate(aids):
                        if s_id <= f_id: continue
                        
                        for t_id, t_aid in enumerate(aids):
                            if t_id <= s_id: continue        
                            
                            rssiVals[0, self.combs.index((f_aid, s_aid, t_aid)), 0] = rssi_by_aid[f_aid] # Last value?
                            rssiVals[0, self.combs.index((f_aid, s_aid, t_aid)), 1] = rssi_by_aid[s_aid]
                            rssiVals[0, self.combs.index((f_aid, s_aid, t_aid)), 2] = rssi_by_aid[t_aid]
                            
                            valIndexes.add(self.combs.index((f_aid, s_aid, t_aid)))
            
            # Check exception.
            if len(valIndexes) == 0:
                continue
            
            # Predict a position.
            valIndexes = list(valIndexes)
            pos = self.model.predict(rssiVals) # Dimension?
            x = np.median(pos[0, valIndexes, 0])
            y = np.median(pos[0, valIndexes, 1])
            z = np.median(pos[0, valIndexes, 2])
            
            # Calibrate bias.
            #x += xOff
            #y += yOff
            #z += zOff 
            
            print(x, y, z, xt, yt, zt, xt - x, yt - y, zt - z)
            
            # Check a tag id category, and determine confidence.
            category = self.__checkTagIdCategory__(id)
            
            # Get confidence radius.
            if category == CATEGORY_MARKER:
                if xOff == 0.:
                    cr = CONFIDENCE_MARKER
                else:
                    cr = np.sqrt(np.power(CONFIDENCE_MARKER, 2.0) + np.power(CONFIDENCE_MARKER, 2.0)) 
            elif category == CATEGORY_COMMUNITY:
                if xOff == 0.:
                    resDF = self.commLocRefDF[self.commLocRefDF.epc_id == id]
                    cr = resDF.cr.iloc[0]
                else:
                    cr = np.sqrt(np.power(resDF.cr.iloc[0], 2.0) + np.power(CONFIDENCE_MARKER, 2.0))                 
            else:
                if xOff == 0.:
                    cr = CONFIDENCE_CTB
                else:
                    cr = np.sqrt(np.power(CONFIDENCE_CTB, 2.0) + np.power(CONFIDENCE_MARKER, 2.0)) #?
            
            trResults.append({'category': category
                            ,'epc_id': id
                            , 'x': x
                            , 'y': y
                            , 'z': z
                            , 'cr': cr
                            , 'xt': xt
                            , 'yt': yt
                            , 'zt': zt})       
        
        # Calculate score.
        trResults = pd.DataFrame(trResults)
        markerScores, commScores, ctbScores = [], [], []
        
        for i in range(trResults.shape[0]):
            category = trResults.loc[i, 'category']
            epc_id = trResults.loc[i, 'epc_id']
            x = trResults.loc[i, 'x']
            y = trResults.loc[i, 'y']
            z = trResults.loc[i, 'z']
            cr = trResults.loc[i, 'cr']
            xt = trResults.loc[i, 'xt']
            yt = trResults.loc[i, 'yt']
            zt = trResults.loc[i, 'zt']
            
            # Check category.
            if category == CATEGORY_MARKER:
                
                # Calculate score according to category.
                E = np.sqrt(np.power(xt - x, 2.) + np.power(yt - y, 2.) + np.power(zt - z, 2.))
                
                if E > cr:
                    markerScores.append(0.0)
                else:
                    markerScores.append(np.min([RMIN_MARKER/(cr + 1e-7), 1.])) #?
            elif category == CATEGORY_COMMUNITY:
                
                # Calculate weight.
                for commName in list(self.tagsByComm.keys()):
                    try:
                        self.tagsByComm[commName].index(epc_id) #?
                    except Exception:
                        continue
                    
                    weight = 1/len(self.tagsByComm[commName])
                    break
                
                # Calculate score according to category.
                E = np.sqrt(np.power(xt - x, 2.) + np.power(yt - y, 2.) + np.power(zt - z, 2.))
                
                if E > cr:
                    commScores.append(0.0)
                else:
                    commScores.append(weight * np.min([RMIN_COMMUNITY/(cr + 1e-7), 1.])) #?
            else:
                
                # Calculate weight.
                if epc_id == 13127:
                    weight = 1.
                else:
                    weight = 1/13
                
                # Calculate score according to category.
                E = np.sqrt(np.power(xt - x, 2.) + np.power(yt - y, 2.) + np.power(zt - z, 2.))
                
                if E > cr:
                    ctbScores.append(0.0)
                else:
                    ctbScores.append(weight * np.min([RMIN_CTB/(cr + 1e-7), 1.])) #?            
        
        # Check exception.
        if len(markerScores) == 0:
            markerScores.append(0.)
        if len(commScores) == 0:
            commScores.append(0.)
        if len(ctbScores) == 0:
            ctbScores.append(0.)
        
        finalScoreTr = (np.asarray(markerScores).mean() + np.asarray(commScores).mean() + 4. * np.asarray(ctbScores).mean()) / 6. * 1000000.        
                
        print('Training data score {0:f}'.format(finalScoreTr))    
        
        # Validation data.
        valResults = []
        valRawDatasDF = rawDatasDF.iloc[int(rawDatasDF.shape[0]*(1.0 - hps['val_ratio'])):, :]

        # Group data according to tag id.
        valRawDatasDFG = valRawDatasDF.groupby('epc_id')
        tagIds = list(valRawDatasDFG.groups.keys())
                
        for i, id in enumerate(tagIds):
            
            # Predict a position and radius confidence.
            df = valRawDatasDFG.get_group(id)
            xt = df.x.iloc[0]
            yt = df.y.iloc[0]
            zt = df.z.iloc[0]
            
            rssiVals = np.zeros(shape=(1,2024,3))
            valIndexes = set()
            
            for i in range(1, 4):
                dfG = df.groupby('a' + str(i) + '_id') # Antenna id?
                
                # Get rssi values.
                rssi_by_aid = {}
                
                for aid in list(dfG.groups.keys()):
                    df = dfG.get_group(aid)
                    rssi = df.loc[:, 'rssi' + str(i)].median() #?
                    rssi_by_aid[aid] = rssi
                
                aids = list(rssi_by_aid.keys())
                
                for f_id, f_aid in enumerate(aids):
                    for s_id, s_aid in enumerate(aids):
                        if s_id <= f_id: continue
                        
                        for t_id, t_aid in enumerate(aids):
                            if t_id <= s_id: continue        
                            
                            rssiVals[0, self.combs.index((f_aid, s_aid, t_aid)), 0] = rssi_by_aid[f_aid] # Last value?
                            rssiVals[0, self.combs.index((f_aid, s_aid, t_aid)), 1] = rssi_by_aid[s_aid]
                            rssiVals[0, self.combs.index((f_aid, s_aid, t_aid)), 2] = rssi_by_aid[t_aid]
                            
                            valIndexes.add(self.combs.index((f_aid, s_aid, t_aid)))
            
            # Check exception.
            if len(valIndexes) == 0:
                continue
            
            # Predict a position.
            valIndexes = list(valIndexes)
            pos = self.model.predict(rssiVals) # Dimension?
            x = np.median(pos[0, valIndexes, 0])
            y = np.median(pos[0, valIndexes, 1])
            z = np.median(pos[0, valIndexes, 2])
            
            # Calibrate bias.
            #x += xOff
            #y += yOff
            #z += zOff 
            
            print(x, y, z, xt, yt, zt, xt - x, yt - y, zt - z)
            
            # Check a tag id category, and determine confidence.
            category = self.__checkTagIdCategory__(id)
            
            # Get confidence radius.
            if category == CATEGORY_MARKER:
                if xOff == 0.:
                    cr = CONFIDENCE_MARKER
                else:
                    cr = np.sqrt(np.power(CONFIDENCE_MARKER, 2.0) + np.power(CONFIDENCE_MARKER, 2.0)) 
            elif category == CATEGORY_COMMUNITY:
                if xOff == 0.:
                    resDF = self.commLocRefDF[self.commLocRefDF.epc_id == id]
                    cr = resDF.cr.iloc[0]
                else:
                    cr = np.sqrt(np.power(resDF.cr.iloc[0], 2.0) + np.power(CONFIDENCE_MARKER, 2.0))                 
            else:
                if xOff == 0.:
                    cr = CONFIDENCE_CTB
                else:
                    cr = np.sqrt(np.power(CONFIDENCE_CTB, 2.0) + np.power(CONFIDENCE_MARKER, 2.0)) #?
            
            valResults.append({'category': category
                            ,'epc_id': id
                            , 'x': x
                            , 'y': y
                            , 'z': z
                            , 'cr': cr
                            , 'xt': xt
                            , 'yt': yt
                            , 'zt': zt})       
        
        # Calculate score.
        valResults = pd.DataFrame(valResults)
        markerScores, commScores, ctbScores = [], [], []
        
        for i in range(valResults.shape[0]):
            category = valResults.loc[i, 'category']
            epc_id = valResults.loc[i, 'epc_id']
            x = valResults.loc[i, 'x']
            y = valResults.loc[i, 'y']
            z = valResults.loc[i, 'z']
            cr = valResults.loc[i, 'cr']
            xt = valResults.loc[i, 'xt']
            yt = valResults.loc[i, 'yt']
            zt = valResults.loc[i, 'zt']
            
            # Check category.
            if category == CATEGORY_MARKER:
                
                # Calculate score according to category.
                E = np.sqrt(np.power(xt - x, 2.) + np.power(yt - y, 2.) + np.power(zt - z, 2.))
                
                if E > cr:
                    markerScores.append(0.0)
                else:
                    markerScores.append(np.min([RMIN_MARKER/(cr + 1e-7), 1.])) #?
            elif category == CATEGORY_COMMUNITY:
                
                # Calculate weight.
                for commName in list(self.tagsByComm.keys()):
                    try:
                        self.tagsByComm[commName].index(epc_id) #?
                    except Exception:
                        continue
                    
                    weight = 1/len(self.tagsByComm[commName])
                    break
                
                # Calculate score according to category.
                E = np.sqrt(np.power(xt - x, 2.) + np.power(yt - y, 2.) + np.power(zt - z, 2.))
                
                if E > cr:
                    commScores.append(0.0)
                else:
                    commScores.append(weight * np.min([RMIN_COMMUNITY/(cr + 1e-7), 1.])) #?
            else:
                
                # Calculate weight.
                if epc_id == 13127:
                    weight = 1.
                else:
                    weight = 1/13
                
                # Calculate score according to category.
                E = np.sqrt(np.power(xt - x, 2.) + np.power(yt - y, 2.) + np.power(zt - z, 2.))
                
                if E > cr:
                    ctbScores.append(0.0)
                else:
                    ctbScores.append(weight * np.min([RMIN_CTB/(cr + 1e-7), 1.])) #?            

        # Check exception.
        if len(markerScores) == 0:
            markerScores.append(0.)
        if len(commScores) == 0:
            commScores.append(0.)
        if len(ctbScores) == 0:
            ctbScores.append(0.)

        finalScoreVal = (np.asarray(markerScores).mean() + np.asarray(commScores).mean() + 4. * np.asarray(ctbScores).mean()) / 6. * 1000000.        
                
        print('Validation data score {0:f}'.format(finalScoreVal))         
        
        # Save final scores.
        finalScoreDF = pd.DataFrame({'final_score_tr': [finalScoreTr], 'final_score_val': [finalScoreVal]})
        finalScoreDF.to_csv('final_scores.csv') #?
        
        return (finalScoreTr, finalScoreVal)
                    
    def test(self, hps, modelLoading = True):
        '''
            Test.
            @param hps: Hyper-parameters.
            @param modelLoading: Model loading flag.
        '''

        self.hps = hps
        
        if modelLoading == True:
            print('Load the pre-trained model...')
            
            if IS_MULTI_GPU == True:
                self.model = multi_gpu_model(keras.models.load_model('iss_rfid_locator.h5'), gpus = NUM_GPUS) 
            else:

                self.model = keras.models.load_model('iss_rfid_locator.h5')
        
        for i in range(1, 21):
            if i < 10:
                numStr = '0' + str(i)
            else:
                numStr = str(i)
            
            taskTagIds = list(pd.read_csv(os.path.join(self.rawDataPath
                                                  , 'tasks', 'task-' + numStr + '.txt')
                                                  , header=None).iloc[:,0])
            taskRFIDDataDF = pd.read_csv(os.path.join(self.rawDataPath
                                                  , 'tasks', 'rfid_raw_task-' + numStr + '.txt')
                                                  , sep='\t'
                                                  , header=None)
            
            if i == 1:
                predResultDF = self.predictRFIDLocations(i, taskTagIds, taskRFIDDataDF)
            else:
                predResultDF = predResultDF.append(self.predictRFIDLocations(i, taskTagIds, taskRFIDDataDF))
        
        # Save.
        predResultDF.to_csv('iss_rfid_location_result.txt', header=0) #?
            
    def predictRFIDLocations(self, taskId, taskTagIds, taskRFIDDataDF):
        '''
            Predict rfid locations.
            @param taskId: Task id.
            @param taskTagIds: Task tag ids.
            @param taskRFIDDataDF: Task rfid data DF.
        '''
        
        results = []
        taskRFIDDataDFG = taskRFIDDataDF.groupby(1) # By epc id.
        
        # Get the bias calibration factor.
        xOff, yOff, zOff = self.__calBiasCalFactor__(taskRFIDDataDFG) 
        
        for i, id in enumerate(taskTagIds):
            
            # Check a tag id category, and determine position and confidence.
            category = self.__checkTagIdCategory__(id)
            
            if category == CATEGORY_MARKER:
                
                # Get the position of a marker tag.
                resDF = self.markerLocRefDF[self.markerLocRefDF.epc_id == id]
                cr = CONFIDENCE_MARKER
                x, y, z = float(resDF.x.iloc[0]), float(resDF.y.iloc[0]), float(resDF.z.iloc[0]) #?
                
                results.append({'task-id': taskId
                                , 'epc_id': id
                                , 'x': x
                                , 'y': y
                                , 'z': z
                                , 'cr': cr})
            elif category == CATEGORY_COMMUNITY:
                
                # Get the position of a community tag.
                resDF = self.commLocRefDF[self.commLocRefDF.epc_id == id]
                
                x, y, z = float(resDF.x.iloc[0]), float(resDF.y.iloc[0]), float(resDF.z.iloc[0]) #?
                cr = resDF.cr.iloc[0]
                
                results.append({'task-id': taskId
                                , 'epc_id': id
                                , 'x': x
                                , 'y': y
                                , 'z': z
                                , 'cr': cr})                
            else:
                
                # Predict a position and radius confidence.
                try:
                    df = taskRFIDDataDFG.get_group(id) # No signal?
                except Exception:
                    continue
                
                dfG = df.groupby(2) # Antenna id. #?
                   
                # Get rssi values.
                rssiVals = np.zeros(shape=(1,2024,3))    
                rssi_by_aid = {}
                
                for aid in list(dfG.groups.keys()):
                    df = dfG.get_group(aid)
                    rssi = df.loc[:, 3].median() #?
                    rssi_by_aid[aid] = rssi
                
                aids = list(rssi_by_aid.keys())
                valIndexes = set()
                
                for f_id, f_aid in enumerate(aids):
                    for s_id, s_aid in enumerate(aids):
                        if s_id <= f_id: continue
                        
                        for t_id, t_aid in enumerate(aids):
                            if t_id <= s_id: continue        
                            
                            rssiVals[0, self.combs.index((f_aid, s_aid, t_aid)), 0] = rssi_by_aid[f_aid]
                            rssiVals[0, self.combs.index((f_aid, s_aid, t_aid)), 1] = rssi_by_aid[s_aid]
                            rssiVals[0, self.combs.index((f_aid, s_aid, t_aid)), 2] = rssi_by_aid[t_aid]
                
                            valIndexes.add(self.combs.index((f_aid, s_aid, t_aid)))
                
                # Check exception.
                if len(valIndexes) == 0: #?
                    continue
                
                # Predict a position.
                valIndexes = list(valIndexes)
                pos = self.model.predict(rssiVals) # Dimension?
                x = np.median(pos[0, valIndexes, 0])
                y = np.median(pos[0, valIndexes, 1])
                z = np.median(pos[0, valIndexes, 2])
                
                # Calibrate bias.
                x += xOff
                y += yOff
                z += zOff 
                
                # Get confidence radius.
                if xOff == 0.:
                    cr = CONFIDENCE_CTB
                else:
                    cr = np.sqrt(np.power(CONFIDENCE_CTB, 2.0) + np.power(CONFIDENCE_MARKER, 2.0)) #?
                
                results.append({'task-id': taskId
                                , 'epc_id': id
                                , 'x': x
                                , 'y': y
                                , 'z': z
                                , 'cr': cr})
        
        return pd.DataFrame(results)
    
    def __calBiasCalFactor__(self, taskRFIDDataDFG):
        '''
            Calculate the bias calibration factor using marker tags.
            @param taskRFIDDataDFG: Task rfid data dataframe group by epc id.
        '''
        
        xOffs, yOffs, zOffs = [], [], []
        
        for id in self.markerIds:
            try:
                df = taskRFIDDataDFG.get_group(id)
            except Exception:
                continue
            
            dfG = df.groupby(2) # Antenna id?
            
            # Get rssi values.
            rssiVals = np.zeros(shape=(1,2024,3))
            rssi_by_aid = {}
            
            for aid in list(dfG.groups.keys()):
                df = dfG.get_group(aid)
                rssi = df.loc[:, 3].median() #?
                rssi_by_aid[aid] = rssi
            
            aids = list(rssi_by_aid.keys())
            valIndexes = set()
            
            for f_id, f_aid in enumerate(aids):
                for s_id, s_aid in enumerate(aids):
                    if s_id <= f_id: continue
                    
                    for t_id, t_aid in enumerate(aids):
                        if t_id <= s_id: continue        
                        
                        rssiVals[0, self.combs.index((f_aid, s_aid, t_aid)), 0] = rssi_by_aid[f_aid]
                        rssiVals[0, self.combs.index((f_aid, s_aid, t_aid)), 1] = rssi_by_aid[s_aid]
                        rssiVals[0, self.combs.index((f_aid, s_aid, t_aid)), 2] = rssi_by_aid[t_aid]
            
                        valIndexes.add(self.combs.index((f_aid, s_aid, t_aid)))
            
            # Predict a position.
            valIndexes = list(valIndexes)
            pos = self.model.predict(rssiVals) # Dimension?
            x = np.median(pos[0, valIndexes, 0])
            y = np.median(pos[0, valIndexes, 1])
            z = np.median(pos[0, valIndexes, 2])
            
            # Calculate offset.
            resDF = self.markerLocRefDF[self.markerLocRefDF.epc_id == id]
            xt, yt, zt = resDF.x.iloc[0], resDF.y.iloc[0], resDF.z.iloc[0] #?            
            
            xOffs.append(xt - x)
            yOffs.append(yt - y)
            zOffs.append(zt - z)
        
        # Check exception.
        if len(xOffs) == 0:
            return (0., 0., 0.)
        
        return (np.median(xOffs), np.median(yOffs), np.median(zOffs))            
                
    def __checkTagIdCategory__(self, tagId):
        '''
            Check a tag id category.
            @param tagId: Tag id.
        '''
        
        # Marker.
        resDF = self.markerLocRefDF[self.markerLocRefDF.epc_id == tagId]
        
        if resDF.shape[0] != 0: #?
            category = CATEGORY_MARKER
        else:
            
            # Community.
            resDF = self.commLocRefDF[self.commLocRefDF.epc_id == tagId]
            
            if resDF.shape[0] != 0: #?
                category = CATEGORY_COMMUNITY
            else:
                
                # CTB.
                resDF = self.ctbLocRefDF[self.ctbLocRefDF.epc_id == tagId]
                
                if resDF.shape[0] != 0: #?
                    category = CATEGORY_CTB
                else:
                    category = CATEGORY_NONE

        return category

def main(args):
    '''
        Main.
        @param args: Arguments.
    '''
    
    hps = {}
    
    if args.mode == 'data':
        
        # Get arguments.
        rawDataPath = args.raw_data_path
        
        # Create training and validation data.
        createTrValData(rawDataPath)
    elif args.mode == 'train':
        
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
        
        modelLoading = False if int(args.model_load) == 0 else True        
        
        # Train.
        rfidLocator = ISSRFIDLocator(rawDataPath)
        
        rfidLocator.train(hps, modelLoading)
    elif args.mode == 'evaluate':
        
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
        
        modelLoading = False if int(args.model_load) == 0 else True        
        
        # Train.
        rfidLocator = ISSRFIDLocator(rawDataPath)
        
        rfidLocator.evaluate(hps, modelLoading) #?
    elif args.mode == 'test':
        
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
        
        modelLoading = False if int(args.model_load) == 0 else True        
        
        # Train.
        rfidLocator = ISSRFIDLocator(rawDataPath)
        
        rfidLocator.test(hps, modelLoading) #?           
        
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
    args = parser.parse_args()
    
    main(args)