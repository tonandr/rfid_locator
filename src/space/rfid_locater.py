'''
Created on Oct 17, 2018

@author: Inwoo Chung (gutomitai@gmail.com)
'''

import os

import pandas as pd
import numpy as np

from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras import optimizers
from keras.utils import multi_gpu_model
import keras

# Constant.
TIME_SLOT = pd.Timedelta(1, unit='h')

CATEGORY_NONE = -1
CATEGORY_MARKER = 0
CATEGORY_COMMUNITY = 1
CATEGORY_CTB = 2

CONFIDENCE_MARKER = 6.
CONFIDENCE_COMMUNITY_RATIO = 0.1
CONFIDENCE_CTB = 12.

IS_MULTI_GPU = False
NUM_GPUS = 4

IS_DEBUG = False

def createTrValData(rawDataPath):
    '''
        Create training and validation data.
        @param rawDataPath: Raw data path.
    '''
    
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
        
        for i in range(len(lines)/2):
            commName = lines[i][0:-1]
            tags = [int(v) for v in lines[i+1][2:-2].split(',')]
            
            tagsByComm[commName] = tags
    
    # Create the location table for each tag.
    for commName in tagsByComm:
        tags = tagsByComm[commName]
        
        for tag_id in tags:
            
            # Get location.
            df = commLocRawDF[commLocRawDF == commName]
            
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
        
        for i in range(len(lines)/2):
            ctbName = lines[i][0:-1]
            tags = [int(v) for v in lines[i+1][2:-2].split(',')]
            
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
                                          , 'rssi1_mean'
                                          , 'rssi2_mean'
                                          , 'rssi3_mean'
                                          , 'x'
                                          , 'y'
                                          , 'z'])
    
    trainRawDF = pd.read_csv(os.path.join(rawDataPath, 'rfid_raw_readings_train.txt'))
    
    # Group data by tag ids.
    trainRawDFG = trainRawDF.groupby('epc_id')
    tagIds = list(trainRawDFG.groups.keys())
    
    for tagId in tagIds:
        
        # Get the raw DF of a tag id.                               
        rawDF = trainRawDFG.get_group(tagId)
        rawDF = rawDF.sort_values(by = 'date')
        
        for i in range(rawDF.shape[0]): rawDF.date.iat[i] = pd.Timestamp(rawDF.date.iloc[i]) 
        
        rawDF.index = rawDF.date
        rawDF = rawDF[['epc_id', 'antenna_id', 'rssi', 'freq', 'phase', 'power', 'cnt']]
        
        # Check a valid tag id.
        # Marker.
        resDF = markerLocRefDF[markerLocRefDF.epc_id == tagId]
        
        if resDF.shape[0] != 0: #?
            category = CATEGORY_MARKER
            x, y, z = resDF.x.iloc[0], resDF.y.iloc[0], resDF.z.iloc[0]
        else:
            
            # Community.
            resDF = commLocRefDF[commLocRefDF.epc_id == tagId]
            
            if resDF.shape[0] != 0: #?
                category = CATEGORY_COMMUNITY
                x, y, z = resDF.x.iloc[0], resDF.y.iloc[0], resDF.z.iloc[0]
            else:
                
                # CTB.
                resDF = ctbLocRefDF[ctbLocRefDF.epc_id == tagId]
                
                if resDF.shape[0] != 0: #?
                    
                    # Extract data relevant to allowable time slots.
                    if tagId == 13127:
                        rawDF = rawDF[pd.Timestamp(resDF.start.iloc[0]):pd.Timestamp(resDF.end.iloc[0])]
                        
                        # Check exception.
                        if rawDF.shape[0] == 0:
                            continue
                        
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
                            continue
                else:
                    continue         
        
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
                                                        
                                trValData = {'category': category
                                                  , 'epc_id': tagId
                                                  , 'start': ist
                                                  , 'end': iet
                                                  , 'a1_id': f_aid
                                                  , 'a2_id': s_aid
                                                  , 'a3_id': t_aid
                                                  , 'rssi1': rssi_by_aid[f_aid]
                                                  , 'rssi2': rssi_by_aid[s_aid]
                                                  , 'rssi3': rssi_by_aid[t_aid]
                                                  , 'x': x
                                                  , 'y': y
                                                  , 'z': z}
                                
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
                                                    
                            trValData = {'category': category
                                              , 'epc_id': tagId
                                              , 'start': ist
                                              , 'end': iet
                                              , 'a1_id': f_aid
                                              , 'a2_id': s_aid
                                              , 'a3_id': t_aid
                                              , 'rssi1': rssi_by_aid[f_aid]
                                              , 'rssi2': rssi_by_aid[s_aid]
                                              , 'rssi3': rssi_by_aid[t_aid]
                                              , 'x': x
                                              , 'y': y
                                              , 'z': z}
                            
                            trValDataDF = trValDataDF.append(pd.DataFrame(trValData))
                
                ist = iet
                iet = ist + TIME_SLOT
        
    # Save training data.
    trValDataDF.to_csv('train.csv')    

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
        self.antennaInfoDF = self.antennaInfoDF[1:,:]
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
        tagsByComm = {}
        with open(os.path.join(rawDataPath, 'misc', 'communities.txt'), 'r') as f:
            lines = f.readlines()
            
            for i in range(len(lines)/2):
                commName = lines[i][0:-1]
                tags = [int(v) for v in lines[i+1][2:-2].split(',')]
                
                tagsByComm[commName] = tags
        
        # Create the location table for each tag.
        for commName in tagsByComm:
            tags = tagsByComm[commName]
            
            for tag_id in tags:
                
                # Get a location and confidence.
                df = commLocRawDF[commLocRawDF == commName]
                
                cr = CONFIDENCE_COMMUNITY_RATIO \
                    *np.median([np.sqrt(np.power(df.x.iloc[0] - self.antennaInfoDF.iloc[i].location_x.iloc[0], 2.0) \
                                          + np.power(df.y.iloc[0] - self.antennaInfoDF.iloc[i].location_y.iloc[0], 2.0) \
                                          + np.power(df.z.iloc[0] - self.antennaInfoDF.iloc[i].location_z.iloc[0], 2.0)) \
                                          for i in range(24)])
                
                commLocRef.append({'epc_id': tag_id
                                   , 'x': df.x.iloc[0]
                                   , 'y': df.y.iloc[0]
                                   , 'z': df.z.iloc[0]
                                   , 'cr': cr})
        
        self.commLocRefDF = pd.DataFrame(commLocRef)
                
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
            
            x = Dense(self.hps['dense1_dim'], activation='relu', name='dense1')(input)
            output = Dense(self.hps['dense2_dim'], activation='linear', name='dense_2')(x)
                                            
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
                
        with open('score.csv', 'w') as f:
            f.write(str(lossMean) + '\n') #?
            
        return lossMean                
        
    def __createTrValData__(self, hps):
        '''
            Create training and validation data.
            @param hps: Hyper-parameters.
        '''
        
        # Load raw data.
        rawDatasDF = pd.read_csv('train.csv')
        
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
            
            rssiRawM[rawDataDF.a1_id.iloc[0], rawDataDF.a2_id.iloc[0], rawDataDF.a3_id.iloc[0]]\
            .append((rawDataDF.rssi1.iloc[0], rawDataDF.rssi2.iloc[0], rawDataDF.rssi3.iloc[0]))
            
            posRawM[rawDataDF.a1_id.iloc[0], rawDataDF.a2_id.iloc[0], rawDataDF.a3_id.iloc[0]]\
            .append((rawDataDF.x.iloc[0], rawDataDF.y.iloc[0], rawDataDF.z.iloc[0]))
        
        # Extract data relevant to 3 antenna combinations.
        rssiM = np.ndarray(shape=(2024), dtype=np.object)
        posM = np.ndarray(shape=(2024), dtype=np.object)                                                                                               
        
        for i in range(2024):
            rssiM[i] = np.asarray(rssiRawM[self.combs[i]])
            posM[i] = np.asarray(posRawM[self.combs[i]])
            
        # Adjust batch size to same size.
        # Get maximum batch length.
        lengths = [rssiM[i].shape[0] for i in range(24)]
        maxLength = lengths[np.argmax(lengths)]
        rssiMList = []
        posMList = []
            
        for i in range(2024):
            rssiMList.append(np.concatenate([rssiM[i], np.repeat(rssiM[i][-1,:][np.newaxis, :], (maxLength - rssiM[i].shape[0]))]).T)
            posMList.append(np.concatenate([posM[i], np.repeat(posM[i][-1,:][np.newaxis, :], (maxLength - rssiM[i].shape[0]))]).T)
        
        rssiM = np.asarray(rssiMList)
        posM = np.asarray(posMList)
        
        rssiM = np.asarray([rssiM[:,:,i] for i in range(maxLength)])
        posM = np.asarray([posM[:,:,i] for i in range(maxLength)])
        
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
            
            rssiRawM[rawDataDF.a1_id.iloc[0], rawDataDF.a2_id.iloc[0], rawDataDF.a3_id.iloc[0]]\
            .append((rawDataDF.rssi1.iloc[0], rawDataDF.rssi2.iloc[0], rawDataDF.rssi3.iloc[0]))
            
            posRawM[rawDataDF.a1_id.iloc[0], rawDataDF.a2_id.iloc[0], rawDataDF.a3_id.iloc[0]]\
            .append((rawDataDF.x.iloc[0], rawDataDF.y.iloc[0], rawDataDF.z.iloc[0]))
        
        # Extract data relevant to 3 antenna combinations.
        rssiM = np.ndarray(shape=(2024), dtype=np.object)
        posM = np.ndarray(shape=(2024), dtype=np.object)                                                                                               
        
        for i in range(2024):
            rssiM[i] = np.asarray(rssiRawM[self.combs[i]])
            posM[i] = np.asarray(posRawM[self.combs[i]])
            
        # Adjust batch size to same size.
        # Get maximum batch length.
        lengths = [rssiM[i].shape[0] for i in range(24)]
        maxLength = lengths[np.argmax(lengths)]
        rssiMList = []
        posMList = []
            
        for i in range(2024):
            rssiMList.append(np.concatenate([rssiM[i], np.repeat(rssiM[i][-1,:][np.newaxis, :], (maxLength - rssiM[i].shape[0]))]).T)
            posMList.append(np.concatenate([posM[i], np.repeat(posM[i][-1,:][np.newaxis, :], (maxLength - rssiM[i].shape[0]))]).T)
        
        rssiM = np.asarray(rssiMList)
        posM = np.asarray(posMList)
        
        rssiM = np.asarray([rssiM[:,:,i] for i in range(maxLength)])
        posM = np.asarray([posM[:,:,i] for i in range(maxLength)])
        
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
            
            rssiRawM[rawDataDF.a1_id.iloc[0], rawDataDF.a2_id.iloc[0], rawDataDF.a3_id.iloc[0]]\
            .append((rawDataDF.rssi1.iloc[0], rawDataDF.rssi2.iloc[0], rawDataDF.rssi3.iloc[0]))
            
            posRawM[rawDataDF.a1_id.iloc[0], rawDataDF.a2_id.iloc[0], rawDataDF.a3_id.iloc[0]]\
            .append((rawDataDF.x.iloc[0], rawDataDF.y.iloc[0], rawDataDF.z.iloc[0]))
        
        # Extract data relevant to 3 antenna combinations.
        rssiM = np.ndarray(shape=(2024), dtype=np.object)
        posM = np.ndarray(shape=(2024), dtype=np.object)                                                                                               
        
        for i in range(2024):
            rssiM[i] = np.asarray(rssiRawM[self.combs[i]])
            posM[i] = np.asarray(posRawM[self.combs[i]])
            
        # Adjust batch size to same size.
        # Get maximum batch length.
        lengths = [rssiM[i].shape[0] for i in range(24)]
        maxLength = lengths[np.argmax(lengths)]
        rssiMList = []
        posMList = []
            
        for i in range(2024):
            rssiMList.append(np.concatenate([rssiM[i], np.repeat(rssiM[i][-1,:][np.newaxis, :], (maxLength - rssiM[i].shape[0]))]).T)
            posMList.append(np.concatenate([posM[i], np.repeat(posM[i][-1,:][np.newaxis, :], (maxLength - rssiM[i].shape[0]))]).T)
        
        rssiM = np.asarray(rssiMList)
        posM = np.asarray(posMList)
        
        rssiM = np.asarray([rssiM[:,:,i] for i in range(maxLength)])
        posM = np.asarray([posM[:,:,i] for i in range(maxLength)])
        
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
            
            rssiRawM[rawDataDF.a1_id.iloc[0], rawDataDF.a2_id.iloc[0], rawDataDF.a3_id.iloc[0]]\
            .append((rawDataDF.rssi1.iloc[0], rawDataDF.rssi2.iloc[0], rawDataDF.rssi3.iloc[0]))
            
            posRawM[rawDataDF.a1_id.iloc[0], rawDataDF.a2_id.iloc[0], rawDataDF.a3_id.iloc[0]]\
            .append((rawDataDF.x.iloc[0], rawDataDF.y.iloc[0], rawDataDF.z.iloc[0]))
        
        # Extract data relevant to 3 antenna combinations.
        rssiM = np.ndarray(shape=(2024), dtype=np.object)
        posM = np.ndarray(shape=(2024), dtype=np.object)                                                                                               
        
        for i in range(2024):
            rssiM[i] = np.asarray(rssiRawM[self.combs[i]])
            posM[i] = np.asarray(posRawM[self.combs[i]])
            
        # Adjust batch size to same size.
        # Get maximum batch length.
        lengths = [rssiM[i].shape[0] for i in range(24)]
        maxLength = lengths[np.argmax(lengths)]
        rssiMList = []
        posMList = []
            
        for i in range(2024):
            rssiMList.append(np.concatenate([rssiM[i], np.repeat(rssiM[i][-1,:][np.newaxis, :], (maxLength - rssiM[i].shape[0]))]).T)
            posMList.append(np.concatenate([posM[i], np.repeat(posM[i][-1,:][np.newaxis, :], (maxLength - rssiM[i].shape[0]))]).T)
        
        rssiM = np.asarray(rssiMList)
        posM = np.asarray(posMList)
        
        rssiM = np.asarray([rssiM[:,:,i] for i in range(maxLength)])
        posM = np.asarray([posM[:,:,i] for i in range(maxLength)])
        
        val = (rssiM, posM)                
        
        
        
            
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
                                                  , sep=','
                                                  , header=None)
            
            if i == 0:
                predResultDF = self.predictRFIDLocations(taskTagIds, taskRFIDDataDF)
            else:
                predResultDF = predResultDF.append(self.predictRFIDLocations(taskTagIds, taskRFIDDataDF))
        
        # Save.
        predResultDF.to_csv('iss_rfid_location_result.txt') #?
            
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
                x, y, z = resDF.x.iloc[0], resDF.y.iloc[0], resDF.z.iloc[0] #?
                
                results.append({'task-id': taskId, 'epc_id': id, 'x': x, 'y': y, 'z': z, 'confidence_radius': cr})
            elif category == CATEGORY_COMMUNITY:
                
                # Get the position of a community tag.
                resDF = self.commLocRefDF[self.commLocRefDF.epc_id == id]
                
                x, y, z = resDF.x.iloc[0], resDF.y.iloc[0], resDF.z.iloc[0] #?
                cr = resDF.cr.iloc[0]
                
                results.append({'task-id': taskId, 'epc_id': id, 'x': x, 'y': y, 'z': z, 'confidence_radius': cr})                
            else:
                
                # Predict a position and radius confidence.
                df = taskRFIDDataDFG.get_group(id)
                df = df.sort_values(by=2) 
                   
                # Get rssi values.
                rssiVals = np.zeros(shape=(1,2024,3))    
                rssi_by_aid = {}
                
                for aid in list(taskRFIDDataDFG .groups.keys()):
                    df = taskRFIDDataDFG.get_group(aid)
                    rssi = df.rssi.median() #?
                    rssi_by_aid[aid] = rssi
                
                aids = list(rssi_by_aid.keys())
                
                for f_id, f_aid in enumerate(aids):
                    for s_id, s_aid in enumerate(aids):
                        if s_id <= f_id: continue
                        
                        for t_id, t_aid in enumerate(aids):
                            if t_id <= s_id: continue        
                            
                            rssiVals[0, self.combs.index((f_aid, s_aid, t_aid)), 0] = rssi_by_aid[f_aid]
                            rssiVals[0, self.combs.index((f_aid, s_aid, t_aid)), 1] = rssi_by_aid[s_aid]
                            rssiVals[0, self.combs.index((f_aid, s_aid, t_aid)), 2] = rssi_by_aid[t_aid]
                
                # Predict a position.
                pos = self.model(rssiVals) # Dimension?
                x = np.median(pos[0, :, 0])
                y = np.median(pos[0, :, 1])
                z = np.median(pos[0, :, 2])
                
                # Calibrate bias.
                x += xOff
                y += yOff
                z += zOff 
                
                # Get confidence radius.
                if xOff == 0.:
                    cr = CONFIDENCE_CTB
                else:
                    cr = np.sqrt(np.power(CONFIDENCE_CTB, 2.0) + np.power(CONFIDENCE_MARKER, 2.0)) #?
                
                results.append({'task-id': taskId, 'epc_id': id, 'x': x, 'y': y, 'z': z, 'confidence_radius': cr})
        
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
            
            df = df.sort_values(by=2)
            
            # Get rssi values.
            rssiVals = np.zeros(shape=(1,2024,3))
            rssi_by_aid = {}
            
            for aid in list(taskRFIDDataDFG .groups.keys()):
                df = taskRFIDDataDFG.get_group(aid)
                rssi = df.rssi.median() #?
                rssi_by_aid[aid] = rssi
            
            aids = list(rssi_by_aid.keys())
            
            for f_id, f_aid in enumerate(aids):
                for s_id, s_aid in enumerate(aids):
                    if s_id <= f_id: continue
                    
                    for t_id, t_aid in enumerate(aids):
                        if t_id <= s_id: continue        
                        
                        rssiVals[0, self.combs.index((f_aid, s_aid, t_aid)), 0] = rssi_by_aid[f_aid]
                        rssiVals[0, self.combs.index((f_aid, s_aid, t_aid)), 1] = rssi_by_aid[s_aid]
                        rssiVals[0, self.combs.index((f_aid, s_aid, t_aid)), 2] = rssi_by_aid[t_aid]
            
            # Predict a position.
            pos = self.model(rssiVals) # Dimension?
            x = np.median(pos[0, :, 0])
            y = np.median(pos[0, :, 1])
            z = np.median(pos[0, :, 2])
            
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
        
if __name__ == '__main__':
    pass