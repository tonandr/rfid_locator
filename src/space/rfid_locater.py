'''
Created on Oct 17, 2018

@author: Inwoo Chung (gutomitai@gmail.com)
'''

import os

import pandas as pd
import numpy as np

# Constant.
TIME_SLOT = pd.Timedelta(1, unit='h')
CATEGORY_MARKER = 0
CATEGORY_COMMUNITY = 1
CATEGORY_CTB = 2

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


        
if __name__ == '__main__':
    pass