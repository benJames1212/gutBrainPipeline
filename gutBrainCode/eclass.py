import sys

import filesys as fs
import numpy as np
import ephys as ep
import matplotlib.pyplot as pl



def create_meta(ephysFolderPath, ephysFilePath):
    '''create meta file to save identity of each of the channels''' 
    ch_num = ep.getchannelnumber(ephysFilePath)
    version = ep.getversion(ephysFilePath)
    print('detected %d channels' % ch_num)
    print('software version %d' % version)
    
    meta = {}
    if version == 0:
        if ch_num == 23:
            ch_names = ['ch_ep0', 'ch_ep1', 'ch_LS0', 'ch_LS1','ch_NIR0','ch_NIR1', 'ch_UV', 'ch_UVlaserfeedback','ch_velmul','ch_velobs','ch_gain'\
                       ,'ch_trial', 'ch_LED0', 'ch_LED1', 'ch_pulsenum', 'ch_pwidth', 'ch_pcycle', 'ch_gpos' ,'ch_galvoX', 'ch_galvoY', 'ch_femtoOn', 'ch_femtoDuration', 'ch_femtoPressure']
        if ch_num == 24:
            ch_names = ['ch_ep0', 'ch_ep1', 'ch_LS0', 'ch_LS1','ch_NIR0','ch_NIR1', 'ch_UV', 'ch_UVlaserfeedback','ch_velmul','ch_velobs','ch_gain'\
                       ,'ch_trial', 'ch_LED0', 'ch_LED1', 'ch_pulsenum', 'ch_pwidth', 'ch_pcycle', 'ch_stimDuration', 'ch_gpos' ,'ch_galvoX', 'ch_galvoY', 'ch_femtoOn', 'ch_femtoDuration', 'ch_femtoPressure']
        if ch_num == 25:
            ch_names = ['ch_ep0', 'ch_ep1', 'ch_LS0', 'ch_LS1','ch_NIR0','ch_NIR1', 'ch_UV', 'ch_UVlaserfeedback','ch_velmul','ch_velobs','ch_gain'\
                       ,'ch_trial', 'ch_LED0', 'ch_LED1', 'ch_uvtrial','ch_pulsenum', 'ch_pwidth', 'ch_pcycle', 'ch_stimDuration', 'ch_gpos' ,'ch_galvoX', 'ch_galvoY', 'ch_femtoOn', 'ch_femtoDuration', 'ch_femtoPressure']
     
        if ch_num == 25:
            ch_names = ['ch_ep0', 'ch_ep1', 'ch_LS0', 'ch_LS1','ch_NIR0','ch_NIR1', 'ch_galvoX', 'ch_UVlaserfeedback','ch_velmul','ch_velobs','ch_gain'\
                       ,'ch_trial', 'ch_LED0', 'ch_LED1', 'ch_uvtrial','ch_pulsenum', 'ch_pwidth', 'ch_pcycle', 'ch_stimDuration', 'ch_gpos' ,'ch_galvoXs', 'ch_galvoY', 'ch_femtoOn', 'ch_femtoDuration', 'ch_femtoPressure']
      
    if version == 8:
          ch_names = ['ch_ep0', 'ch_ep1', 'ch_LS0', 'ch_LS1','ch_NIR0','ch_NIR1', 'ch_galvoX', 'ch_galvoY', 'ch_UV', 'ch_UVlaserfeedback',\
                      'ch_velmul','ch_velobs', 'ch_gain' ,'ch_trial', 'ch_LED0', 'ch_LED1', 'ch_uvtrial','ch_pulsenum', 'ch_pwidth', 'ch_pcycle', 'ch_stimDuration',\
                      'ch_gpos' , 'ch_femtoOn', 'ch_femtoDuration', 'ch_femtoPressure']

    if version == 9:
          ch_names = ['ch_ep0', 'ch_ep1', 'ch_LS0', 'ch_LS1','ch_LED0', 'ch_LED1', 'ch_NIR0','ch_NIR1', 'ch_galvoX', 'ch_galvoY', 'ch_UV', 'ch_UVlaserfeedback',\
                      'ch_velmul','ch_velobs', 'ch_gain' ,'ch_trial', 'ch_uvtrial','ch_pulsenum', 'ch_pwidth', 'ch_pcycle', 'ch_stimDuration',\
                      'ch_gpos' , 'ch_femtoOn', 'ch_femtoDuration', 'ch_femtoPressure']
    if version == 10:
          ch_names = ['ch_ep0', 'ch_ep1', 'ch_LS0', 'ch_LS1','ch_LED0', 'ch_LED1', 'ch_NIR0','ch_NIR1', 'ch_galvoX', 'ch_galvoY', 'ch_UV', 'ch_UVlaserfeedback',\
                      'ch_473', 'ch_velmul','ch_velobs', 'ch_gain' ,'ch_trial', 'ch_uvtrial','ch_pulsenum', 'ch_pwidth', 'ch_pcycle', 'ch_stimDuration',\
                      'ch_gpos' , 'ch_femtoOn', 'ch_femtoDuration', 'ch_femtoPressure']    
          
    meta['ch_names'] = ch_names
    meta['sr'] = 6000
    for ind, name in enumerate(ch_names):
#         tmp = '_' + str(ind) # old system
#         meta[tmp] = name # old system
        meta[ind] = name
        meta[name] = ind

    meta['LS'] = [0]
    meta['NIR'] = [0]
    ephysFolderPath = ephysFolderPath + 'channel_meta'   
    np.save(ephysFolderPath, meta) 
#     fs.create_meta(meta, path + '.xml') # old system
#     dd = fs.read_meta(path + '.xml') # old system
    print('saved channel information to: %s ' % ephysFolderPath)

    
class Ephys():
    def __init__(self, ephys_path = None, meta = None, dirs = None):
        
        
        if ephys_path is None:
            ephys_path = glob(dirs['ephys'] + '*chFlt')[0]
        self.path = ephys_path
        
        if meta is None:
            fpath = dirs['ephys'] + 'channel_meta.npy'
            meta = np.load(fpath, allow_pickle = True).item()
            
        self.meta = meta
        self.dirs = dirs
        self.initialise()
        self.ep = {}
        
    def initialise(self,):
        self.ch_num = ep.getchannelnumber(self.path) # get number of channels
        self.version = ep.getversion(self.path)
        self.data = ep.load(self.path, num_channels = self.ch_num)
        self.sr = int(self.meta['sr'])
        
        self.meta['sr'] = int(self.meta['sr'])
        self.tmax = self.data.shape[1]/self.sr
        self.xax = np.arange(self.data.shape[1])/self.sr
        print('tmax: %f' % self.tmax)
        if self.dirs is not None:
            self.fig_path = self.dirs['plots'] + 'ephys_'
        else:
            self.fig_path = './ephys_'
        
    def run(self, save = True):
        self.get_lsEpTimes()
        self.get_nirEpTimes()
        self.get_nirStackTimes()
        if save:
            self.save_data()
        

    def save_data(self,):
        fpath = self.dirs['ephys'] + 'ephys_meta.npy'
        np.save(fpath, self.ep, allow_pickle = True) 
        print('\n\n\nsaved ephys analysis to: %s' % fpath)
 

    def process_uvCh(self,):
        self.get_uvEpTimes()   
        self.get_uvNirTimes() 
        self.get_uvStackTimes()
        
        self.get_uvStimParameters() # get pulseNumber, pulseWidth, pulseCycle etc
        self.make_uvStimParameterDictionary() # make summary dictionary
        self.make_473StimParameterDictionary() # make summary dictionary
        self.make_uvtrial_dict()
        
        
#         self.pulseEpTimes = np.argwhere(self.data[self.meta['ch_pulsenum']]>0)
#         self.pulseStackTimes = np.unique(ep.find_frame_by_LS(self.ls0EpTimes, self.pulseEpTimes))
#         self.ep['pulseEpTimes'] = self.pulseEpTimes
#         self.ep['pulseStackTimes'] = self.pulseStackTimes

        
    def process_femtoCh(self,):
        self.get_femtoEpTimes()
        self.get_femtoStackTimes()
        self.get_femtoNirTimes()
        
    def process_ledCh(self,):
        self.get_ledEpTimes()
        self.get_ledStackTimes()
        self.get_ledNirTimes()  
        
    def make_uvtrial_dict(self,):
        trs = np.unique(self.trials)
        tr_dicts = {}
        for i, tmp in enumerate(trs):
            inds = np.argwhere(self.trials == tmp).squeeze()
#             print(inds)
            tr_dicts[i] = {}
            tr_dicts[i]['lsStarts'] = np.array([j[0] for j in self.uvStackTimes[inds]])
            tr_dicts[i]['lsEnds'] = np.array([j[1] for j in self.uvStackTimes[inds]])
            tr_dicts[i]['lsIntervals'] = self.uvStackTimes[inds]
            tr_dicts[i]['epStarts'] = np.array([j[0] for j in self.uvEpTimes[inds]])
            tr_dicts[i]['epEnds'] = np.array([j[1] for j in self.uvEpTimes[inds]])
            tr_dicts[i]['epIntervals'] = self.uvEpTimes[inds]
        self.tr_dicts = tr_dicts
        self.ep['tr_dicts'] = self.tr_dicts
        
    def get_lsEpTimes(self,):
        print('\nprocessing LS0')
        meta = self.meta
        th = [3.9, 100]
        self.ls0EpTimes, self.ls0Rate = ep.detect_LStimes(self.data, meta['ch_LS0'], th = th, sr = self.sr)
        self.ls0N = len(self.ls0EpTimes)
        print('# LS0 pulses: %d' % self.ls0N) 
        
        print('\nprocessing LS1')
        self.ls1EpTimes, self.ls1Rate = ep.detect_LStimes(self.data, meta['ch_LS1'], th = th, sr = self.sr)
        self.ls1N = len(self.ls1EpTimes)
        print('# LS1 pulses: %d' % self.ls1N) 
        
        self.ep['ls0EpTimes'] = self.ls0EpTimes
        self.ep['ls0N'] = self.ls0N
        self.ep['ls0Rate'] = self.ls0Rate
        self.ep['ls1EpTimes'] = self.ls1EpTimes
        self.ep['ls1N'] = self.ls1N
        
        
    def get_gposEpTimes(self,):
        print('\nprocessing gpos')
        meta = self.meta
        th = [.5,100]
        self.gposEpTimes = ep.find_onset_offset_timeseries(self.data, meta['ch_galvoX'], th = th, duration=2)
      #  self.gposEpTimes, self. gposRate = ep.detect_LStimes(self.data, meta['ch_gpos'], th = th, sr = self.sr)
        self.gposN = len(self.gposEpTimes)
        print('# gpos pulses: %d' % self.gposN)
              
        if len(self.gposEpTimes)>0:
            self.gposEpDurations = ep.get_durations(np.array(self.gposEpTimes))
            self.gposEpIntervals = ep.get_intervals(np.array(self.gposEpTimes))
            print('duration of galvoXpos pulses: {0}'.format(self.gposEpDurations))
            
            self.ep['gposEpDurations'] = self.gposEpDurations
              
        self.ep['gposEpTimes'] = self.gposEpTimes
        #self.ep['gposRate'] = self.gposRate
        self.ep['gposN'] = self.gposN
        
        
    def get_nirEpTimes(self,):
        print('\nprocessing NIR0')
        meta = self.meta
        th = [2,100]
        self.nir0EpTimes, self.nir0Rate = ep.detect_LStimes(self.data, meta['ch_NIR0'], th = th, sr = self.sr)
        self.nir0N = len(self.nir0EpTimes)
        print('# NIR0 pulses: %d' % self.nir0N) 
        
        self.ep['nir0EpTimes'] = self.nir0EpTimes
        self.ep['nir0Rate'] = self.nir0Rate
        self.ep['nir0N'] = self.nir0N
        
        if 'ch_NIR1' in meta.keys():
            print('\nprocessing NIR1')
            self.nir1EpTimes, self. nir1Rate = ep.detect_LStimes(self.data, meta['ch_NIR1'], th = th, sr = self.sr)
            self.nir1N = len(self.nir1EpTimes)
            print('# NIR1 pulses: %d' % self.nir1N) 
            self.ep['nir1EpTimes'] = self.nir1EpTimes
            self.ep['nir1Rate'] = self.nir1Rate
            self.ep['nir1N'] = self.nir1N

    def get_nirStackTimes(self,):
        meta = self.meta
        if len(self.nir0EpTimes)>100:
            self.nir0StackTimes = ep.find_frame_by_LS(self.ls0EpTimes, self.nir0EpTimes)
        else:
            self.nir0StackTimes = None
        self.ep['nir0StackTimes'] = self.nir0StackTimes
            
        if 'ch_NIR1' in meta.keys():
            if len(self.nir1EpTimes)>100:
                self.nir1StackTimes = ep.find_frame_by_LS(self.ls0EpTimes, self.nir1EpTimes)
            else:
                self.nir1StackTimes = None
        
        self.ep['nir1StackTimes'] = self.nir1StackTimes
       
    def get_uvEpTimes(self,):
        print('\nprocessing UV')
        meta = self.meta
        th = [3,100] # should be around 3V
        durationpulse = 1
        durationtrain = 3*self.sr
#         self.UVp_eptimes, self.UV_eptimes = ep.estimate_pulsetrain_onset(signal, th, durationpulse, durationtrain)
        self.uvpEpTimes, self.uvEpTimes = ep.find_onset_offset_pulsetimeseries(self.data, meta['ch_UV'], th = th, durationpulse = durationpulse, durationtrain = durationtrain)
        self.uvN = len(self.uvEpTimes)
        print('# UV pulses: %d' % self.uvN) 

        self.ep['uvpEpTimes'] = self.uvpEpTimes
        self.ep['uvEpTimes'] = self.uvEpTimes
        self.ep['uvN'] = self.uvN
        
        if self.version>9:
            self.blLaserpEpTimes, self.blLaserEpTimes = ep.find_onset_offset_pulsetimeseries(self.data, meta['ch_473'], th = th, durationpulse = durationpulse, durationtrain = durationtrain)
            self.blLaserN = len(self.blLaserEpTimes)
            print('# 473 pulses: %d' % self.blLaserN) 
    
            self.ep['blLaserpEpTimes'] = self.blLaserpEpTimes
            self.ep['blLaserEpTimes'] = self.blLaserEpTimes
            self.ep['blLaserN'] = self.blLaserN
        
            
              
    def get_uvStackTimes(self,):
        self.uvStackTimes = ep.find_start_end_frame(self.ls0EpTimes, self.uvEpTimes, var = 2)
                    
        self.uvStackDurations = ep.get_durations(np.array(self.uvStackTimes))
        self.uvStackIntervals = ep.get_intervals(np.array(self.uvStackTimes))
        
        self.ep['uvStackTimes'] = self.uvStackTimes
        self.ep['uvStackDurations'] = self.uvStackDurations
        self.ep['uvStackIntervals'] = self.uvStackIntervals
        
        if self.version>9:
            self.blLaserStackTimes = ep.find_start_end_frame(self.ls0EpTimes, self.blLaserEpTimes, var = 2)
                    
            self.blLaserStackDurations = ep.get_durations(np.array(self.blLaserStackTimes))
            self.blLaserStackIntervals = ep.get_intervals(np.array(self.blLaserStackTimes))
            
            self.ep['blLaserStackTimes'] = self.blLaserStackTimes
            self.ep['blLaserStackDurations'] = self.blLaserStackDurations
            self.ep['blLaserStackIntervals'] = self.blLaserStackIntervals
            
        
        
    def get_uvNirTimes(self,):
        self.uvNir0Times = ep.find_start_end_frame(self.nir0EpTimes, self.uvEpTimes, var = 2)            
        self.ep['uvNir0Times'] = self.uvNir0Times
        self.uvNir1Times = ep.find_start_end_frame(self.nir1EpTimes, self.uvEpTimes, var = 2)
        self.ep['uvNir1Times'] = self.uvNir1Times
        
        if self.version>9:
            self.blLaserNir0Times = ep.find_start_end_frame(self.nir0EpTimes, self.blLaserEpTimes, var = 2)            
            self.ep['blLaserNir0Times'] = self.uvNir0Times
            self.blLaserNir1Times = ep.find_start_end_frame(self.nir1EpTimes, self.blLaserEpTimes, var = 2)
            self.ep['blLaserNir1Times'] = self.blLaserNir1Times
        
    def get_uvStimParameters(self,):
        meta = self.meta
        if self.version > 7:
            micro2milli = 1 #already in millisecond
        else:
            micro2milli = 1000 # as it's in microseconds
        self.uvPWidth = np.array([self.data[meta['ch_pwidth'], i[0]] for i in self.ep['uvEpTimes']])/micro2milli
        self.uvPCycle = np.array([self.data[meta['ch_pcycle'], i[0]] for i in self.ep['uvEpTimes']])/micro2milli
        self.uvPNumber = np.array([self.data[meta['ch_pulsenum'], i[0]] for i in self.ep['uvEpTimes']])
        self.ep['uvPWidth'] = self.uvPWidth
        self.ep['uvPCycle'] = self.uvPCycle
        self.ep['uvPNumber'] = self.uvPNumber
        
        if self.version>9:
            self.blLaserPWidth = np.array([self.data[meta['ch_pwidth'], i[0]] for i in self.ep['blLaserEpTimes']])/micro2milli
            self.blLaserPCycle = np.array([self.data[meta['ch_pcycle'], i[0]] for i in self.ep['blLaserEpTimes']])/micro2milli
            self.blLaserPNumber = np.array([self.data[meta['ch_pulsenum'], i[0]] for i in self.ep['blLaserEpTimes']])
            self.ep['blLaserPWidth'] = self.blLaserPWidth
            self.ep['blLaserPCycle'] = self.blLaserPCycle
            self.ep['blLaserPNumber'] = self.blLaserPNumber
        
        
        secs2milli = 1000
        if 'ch_stimDuration' in meta.keys(): # 24 channel file
            self.uvStimDuration = np.array([self.data[meta['ch_stimDuration'], i[0]] for i in self.ep['uvEpTimes']])
        else:
            self.uvStimDuration = np.array([(i[1]-i[0])/self.sr*secs2milli for i in self.uvEpTimes])
        self.ep['uvStimDuration'] = self.uvStimDuration
        if self.version>9:
            if 'ch_stimDuration' in meta.keys(): # 24 channel file
                self.blLaserStimDuration = np.array([self.data[meta['ch_stimDuration'], i[0]] for i in self.ep['blLaserEpTimes']])
            else:
                self.blLaserStimDuration = np.array([(i[1]-i[0])/self.sr*secs2milli for i in self.blLaserEpTimes])
            self.ep['blLaserStimDuration'] = self.blLaserStimDuration
            

        
    def make_uvStimParameterDictionary(self,):
        print('\nprocessing trials')
        meta = self.meta
        self.trials = np.array([self.data[meta['ch_gpos']][i[0]] for i in self.uvEpTimes])
#         print('trials:', self.trials)
        self.ep['trials'] = self.trials
        
        self.uvTrialsParams = {}
        for ind, i in enumerate(self.uvStackTimes):
            self.uvTrialsParams[ind] = {}
            self.uvTrialsParams[ind]['pCycle'] = self.uvPCycle[ind]
            self.uvTrialsParams[ind]['pWidth'] = self.uvPWidth[ind]
            self.uvTrialsParams[ind]['pNumber'] = self.uvPNumber[ind]
            self.uvTrialsParams[ind]['stimDuration'] = self.uvStimDuration[ind]
            self.uvTrialsParams[ind]['trial'] = self.trials[ind]
            self.uvTrialsParams[ind]['stackTime'] = self.uvStackTimes[ind]
            self.uvTrialsParams[ind]['epTime'] = self.uvEpTimes[ind]
            self.uvTrialsParams[ind]['nir0Time'] = self.uvNir0Times[ind]
            self.uvTrialsParams[ind]['nir1Time'] = self.uvNir1Times[ind]
        self.ep['uvTrialsParams'] = self.uvTrialsParams

    def make_473StimParameterDictionary(self,):
        print('\nprocessing trials')
        meta = self.meta
        self.blLasertrials = np.array([self.data[meta['ch_gpos']][i[0]] for i in self.blLaserEpTimes])
#         print('trials:', self.trials)
        self.ep['blLasertrials'] = self.blLasertrials
        
        self.blLaserTrialsParams = {}
        for ind, i in enumerate(self.blLaserStackTimes):
            self.blLaserTrialsParams[ind] = {}
            self.blLaserTrialsParams[ind]['pCycle'] = self.blLaserPCycle[ind]
            self.blLaserTrialsParams[ind]['pWidth'] = self.blLaserPWidth[ind]
            self.blLaserTrialsParams[ind]['pNumber'] = self.blLaserPNumber[ind]
            self.blLaserTrialsParams[ind]['stimDuration'] = self.blLaserStimDuration[ind]
            self.blLaserTrialsParams[ind]['trial'] = self.blLasertrials[ind]
            self.blLaserTrialsParams[ind]['stackTime'] = self.blLaserStackTimes[ind]
            self.blLaserTrialsParams[ind]['epTime'] = self.blLaserEpTimes[ind]
            self.blLaserTrialsParams[ind]['nir0Time'] = self.blLaserNir0Times[ind]
            self.blLaserTrialsParams[ind]['nir1Time'] = self.blLaserNir1Times[ind]
        self.ep['blLaserTrialsParams'] = self.blLaserTrialsParams

            
            

    def get_femtoEpTimes(self,):
        print('\nprocessing femtoJet channel')
        meta = self.meta
        meta['ch_femtoOn']
        meta['ch_femtoDuration']
        meta['ch_femtoPressure']
        th = 0.1
        
        th = [0.1, 100]
        self.femtoEpTimes = ep.estimate_onset(self.data[meta['ch_femtoOn']], th=th, duration=2)
        
#         self.femtoEpTimes = np.argwhere(self.data[meta['ch_femtoOn']]>th).squeeze()
        self.femtoN = len(self.femtoEpTimes)
        
        self.femtoDurations = np.array([self.data[meta['ch_femtoDuration'], onset] for onset in self.femtoEpTimes])
        self.femtoPressures = np.array([self.data[meta['ch_femtoPressure'], onset] for onset in self.femtoEpTimes])

        self.ep['femtoEpTimes'] = self.femtoEpTimes
        self.ep['femtoN'] = self.femtoN
        self.ep['femtoDurations'] = self.femtoDurations
        self.ep['femtoPressures'] = self.femtoPressures
        print('# femto pulses: %d' % self.femtoN) 
        
    def get_femtoStackTimes(self,):
        self.femtoStackTimes = ep.find_frame_by_LS(self.ls0EpTimes, self.femtoEpTimes)
        self.ep['femtoStackTimes'] = self.femtoStackTimes
        
    def get_femtoNirTimes(self,):
        self.femtoNirTimes = ep.find_frame_by_LS(self.nir0EpTimes, self.femtoEpTimes)
        self.ep['femtoNirTimes'] = self.femtoNirTimes
        
    def get_ledEpTimes(self,):
        print('\nprocessing LED0')
        meta = self.meta
        th = [.5,100]
        self.led0EpTimes = ep.find_onset_offset_timeseries(self.data, meta['ch_LED0'], th = th, duration=40)
        self.led0N = len(self.led0EpTimes)
        print('# LED0 pulses: %d' % self.led0N)
        self.ep['led0N'] = self.led0N
        self.ep['led0EpTimes'] = self.led0EpTimes
        
        if len(self.led0EpTimes)>1:
            self.led0EpDurations = ep.get_durations(np.array(self.led0EpTimes))
            self.led0EpIntervals = ep.get_intervals(np.array(self.led0EpTimes))
            print('duration of LED pulses: {0}'.format(self.led0EpDurations))
            
            self.ep['led0EpDurations'] = self.led0EpDurations
            
        
        
        print('\nprocessing LED1')
        self.led1EpTimes = ep.find_onset_offset_timeseries(self.data, meta['ch_LED1'], th = th, duration=40)
        self.led1N = len(self.led1EpTimes)
        print('# LED1 pulses: %d' % self.led1N) 

        self.ep['led1EpTimes'] = self.led1EpTimes
        self.ep['led1N'] = self.led1N
        
        if len(self.led1EpTimes)>1:
            self.led1EpDurations = ep.get_durations(np.array(self.led1EpTimes))
            self.LED1_epintervals = ep.get_intervals(np.array(self.led1EpTimes))
            print('duration of LED pulses: {0}'.format(self.led1EpDurations))
            self.ep['led1EpDurations'] = self.led1EpDurations
        

    def get_ledStackTimes(self,):
        self.led0StackTimes = ep.find_start_end_frame(self.ls0EpTimes, self.led0EpTimes, var = 2)
        self.led1StackTimes = ep.find_start_end_frame(self.ls0EpTimes, self.led1EpTimes, var = 2)
        self.led0StackDurations = ep.get_durations(np.array(self.led0StackTimes))
        self.led0StackIntervals = ep.get_intervals(np.array(self.led0StackTimes))
        
        self.led1StackDurations = ep.get_durations(np.array(self.led1StackTimes))
        self.led1StackIntervals = ep.get_intervals(np.array(self.led1StackTimes))
        
        self.ep['led0StackTimes'] = self.led0StackTimes
        self.ep['led1StackTimes'] = self.led1StackTimes
        self.ep['led0StackDurations'] = self.led0StackDurations
        self.ep['led1StackDurations'] = self.led1StackDurations
        self.ep['led0StackIntervals'] = self.led0StackIntervals
        self.ep['led1StackIntervals'] = self.led1StackIntervals
            
        
    def get_ledNirTimes(self,):
        self.led0Nir0Times = ep.find_start_end_frame(self.nir0EpTimes, self.led0EpTimes, var = 2)
        self.led1Nir0Times = ep.find_start_end_frame(self.nir0EpTimes, self.led1EpTimes, var = 2)
        
        self.led0Nir1Times = ep.find_start_end_frame(self.nir1EpTimes, self.led0EpTimes, var = 2)
        self.led1Nir1Times = ep.find_start_end_frame(self.nir1EpTimes, self.led1EpTimes, var = 2)
        
        self.ep['led0Nir0Times'] = self.led0Nir0Times
        self.ep['led1Nir0Times'] = self.led1Nir0Times
        self.ep['led0Nir1Times'] = self.led0Nir1Times
        self.ep['led1Nir1Times'] = self.led1Nir1Times
       

        
    def savefig(self, title):
        pl.savefig(self.fig_path + title + '.png', transparent = True)
        
  
  
        
def plot_all(data, meta, t = None, save = True, save_path = None, ds = 1, in_sec = False):
    sr = meta['sr']
    xax = np.arange(data.shape[1])
    x_label = 'time (frames)'
    if in_sec:
        xax = xax/sr
        x_label = 'time (seconds)'
        
    ch_num = data.shape[0]

    
    if t is None:
        t = [0, data.shape[1]]
    sl = slice(int(t[0]*sr), int(t[1]*sr))
    fig, axs = pl.subplots(figsize = (21, 2*ch_num), nrows = ch_num, sharex = True)
    title = 'all channels'
    fig.suptitle(title, y = 1.02)
    for ind in range(ch_num):
        axs[ind].set_title(meta['ch_names'][ind], y = 1.1)
        axs[ind].plot(xax[sl][::ds], data[ind, sl][::ds])

    axs[ind].set_xlabel(x_label)

    pl.tight_layout() 
    if save:
        title = 'all_channels_t' + str(t[0]) + 's_' + str(t[1]) + 's'
        if in_sec:
            title = title + ' inSec'
        pl.savefig(save_path + title + '.png')

def LS_plot(data, meta, ls0EpTimes, t = None, save = True, save_path = None, ds = 1):
    if t is None:
        t = [0, data.shape[1]]
    sr = meta['sr']
    ch_num = data.shape[0]
    xax = np.arange(data.shape[1])/sr
    sl = slice(int(t[0]*sr), int(t[1]*sr))
    LS_ch = ['ch_LS0', 'ch_LS1']

    fig, axs = pl.subplots(figsize = (21, 6), nrows = 2, sharex = True)
    title = 'LS channels'
    fig.suptitle(title, y = 1.1)
    for ind, i in enumerate(LS_ch):
        print(ind, i)
        axs[ind].set_title(i, y = 1.1)
        axs[ind].plot(xax[sl][::ds], data[meta[i], sl][::ds])
    axs[ind].set_xlabel('time (secs)')

    vals = np.array(list(filter(lambda x: x < t[1]*sr and x > t[0]*sr, ls0EpTimes)))
    axs[0].vlines(x = vals/sr, ymin = -1, ymax = 6)
    axs[1].vlines(x = vals/sr, ymin = -1, ymax = 6)

    pl.tight_layout() 
    if save:
        title = 'LS_channels_t' + str(t[0]) + 's_' + str(t[1]) + 's'
        pl.savefig(save_path + title + '.png')
        
        

def NIR_plot(data, meta, nir0EpTimes, t = None, save = True, save_path = None, ds = 1):
    if t is None:
        t = [0, data.shape[1]]
    sr = meta['sr']
    ch_num = data.shape[0]
    xax = np.arange(data.shape[1])/sr
    sl = slice(int(t[0]*sr), int(t[1]*sr))
    NIR_ch = ['ch_NIR0', 'ch_NIR1']

    fig, axs = pl.subplots(figsize = (21, 6), nrows = 2, sharex = True)
    title = 'NIR channels'
    fig.suptitle(title, y = 1.1)
    for ind, i in enumerate(NIR_ch):
        axs[ind].set_title(i, y = 1.1)
        axs[ind].plot(xax[sl][::ds], data[meta[i], sl][::ds])
    axs[ind].set_xlabel('time (secs)')

    args = np.where(np.logical_and(np.array(nir0EpTimes)>=(t[0]*sr), np.array(nir0EpTimes)<(t[1]*sr)))[0]
    axs[0].vlines(x = np.array(nir0EpTimes)[args]/sr, ymin = min(abs(data[meta[NIR_ch[0]]][sl]))-1, ymax = 2+max(abs(data[meta[NIR_ch[0]]][sl])))

    pl.tight_layout() 
    if save:
        title = 'NIR_channels_t' + str(t[0]) + 's_' + str(t[1]) + 's'
        pl.savefig(save_path + title + '.png')

        
def UV_plot(data, meta, nir0EpTimes, t = None, save = True, save_path = None, ds = 1,in_sec = True):
    if t is None:
        t = [0, data.shape[1]]
    sr = meta['sr']
    ch_num = data.shape[0]
    xax = np.arange(data.shape[1])/sr
    sl = slice(int(t[0]*sr), int(t[1]*sr))
    NIR_ch = ['ch_UV','ch_UVlaserfeedback']

    fig, axs = pl.subplots(figsize = (21, len(NIR_ch)*3), nrows = len(NIR_ch), sharex = True)
    title = 'UV channels'
    fig.suptitle(title, y = 1.1)
    for ind, i in enumerate(NIR_ch):
        axs[ind].set_title(i, y = 1.1)
        axs[ind].plot(xax[sl][::ds], data[meta[i], sl][::ds])
    axs[ind].set_xlabel('time (secs)')

    args = np.where(np.logical_and(np.array(nir0EpTimes)>=(t[0]*sr), np.array(nir0EpTimes)<(t[1]*sr)))[0]
    axs[0].vlines(x = np.array(nir0EpTimes)[args]/sr, ymin = min(abs(data[meta[NIR_ch[0]]][sl]))-1, ymax = 2+max(abs(data[meta[NIR_ch[0]]][sl])))

    pl.tight_layout() 
    if save:
        title = 'NIR_channels_t' + str(t[0]) + 's_' + str(t[1]) + 's'
        pl.savefig(save_path + title + '.png')
  

def protocol_plot(data, meta, ch_eptimes, trials, ds = 50, save = True, save_path = None, in_sec = False):
    
    sr = meta['sr']
    xax = np.arange(data.shape[1])
    x_label = 'time (frames)'
    if in_sec:
        xax = xax/sr
        x_label = 'time (seconds)'
    
    xax = xax[::ds]
    data = data[:,::ds]
    
    fig, axs = pl.subplots(figsize = (21, 12), nrows = 4, sharex = True)
    title = 'Protocol'
    fig.suptitle(title, y = 1.1)
    ch_name = 'ch_UV'

    
    axs[0].set_title(ch_name, y = 1.1)
    axs[0].plot(xax, data[meta[ch_name]])
    axs[0].vlines(x = np.array(ch_eptimes)/sr, ymin = 0, ymax = 1.5)
    
    axs[1].set_title('trial', y = 1.1)
    axs[1].plot(xax, data[meta['ch_trial']], label = 'raw data')
    
    [axs[1].vlines(x = i[0]/sr, ymin = trials[ind]+.5, ymax = trials[ind]+1) for ind, i in enumerate(ch_eptimes)]
    axs[1].legend(loc = 1)
    
    axs[2].set_title('trial structure', y = 1.1)
    [axs[2].plot(i[0]/sr, trials[ind]+1,'ok') for ind, i in enumerate(ch_eptimes)]
    axs[2].set_ylim([-.5,3])
    
    
    axs[3].set_title('galvos', y = 1.1)
    axs[3].plot(xax, data[meta['ch_galvoX']], label = 'X')
    axs[3].plot(xax, data[meta['ch_galvoY']], label = 'Y')
    axs[3].legend(loc = 1)
    

    axs[-1].set_xlabel(x_label)
    
    pl.tight_layout() 
    if save:
        title = 'protocol'
        if in_sec:
            title = title + ' inSec'
        pl.savefig(save_path + title + '.png')
    
    
# def edit_meta(dd):
#     d0 = {}
#     for key in dd.keys():
#         if ''.join(list(key)[:2]) == 'ch':
#             if key != 'ch_names':
#                 d0[key] = int(dd[key])
#             else:
#                 d0[key] = dd[key]
#         else:
#             d0[key] = dd[key]
#     return d0