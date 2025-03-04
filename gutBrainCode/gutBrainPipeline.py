# -*- coding: utf-8 -*-
"""
Main Code for the gut-brain analysis pipeline
"""
import h5py
import numpy as np
import scipy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def windowed_variance(signal, kern_mean=None, kern_var=None, fs=6000):
    """
    Estimate smoothed sliding variance of the input signal

    signal : numpy array

    kern_mean : numpy array
        kernel to use for estimating baseline

    kern_var : numpy array
        kernel to use for estimating variance

    fs : int
        sampling rate of the data
    """
    from scipy.signal import gaussian, fftconvolve

    # set the width of the kernels to use for smoothing
    kw = int(0.04 * fs)

    if kern_mean is None:
        kern_mean = gaussian(kw, kw // 10)
        kern_mean /= kern_mean.sum()

    if kern_var is None:
        kern_var = gaussian(kw, kw // 10)
        kern_var /= kern_var.sum()

    mean_estimate = fftconvolve(signal, kern_mean, "same")
    var_estimate = (signal - mean_estimate) ** 2
    fltch = fftconvolve(var_estimate, kern_var, "same")

    return fltch, var_estimate, mean_estimate

def load(in_file, num_channels=10, memmap=False):
    """Load multichannel binary data from disk, return as a [channels,samples] sized numpy array
    """
    from numpy import fromfile, float32

    if memmap:
        from numpy import memmap

        data = memmap(in_file, dtype=float32)
    else:
        with open(in_file, "rb") as fd:
            data = fromfile(file=fd, dtype=float32)
    trim = data.size % num_channels
    # transpose to make dimensions [channels, time]
    data = data[: (data.size - trim)].reshape(data.size // num_channels, num_channels).T
    if trim > 0:
        print("Data needed to be truncated!")

    return data


def loadGutBrainData(baseDir:str):
    """
    Loads gut brain data from a defined path baseDir
    INPUTS:
        baseDir : path to data
    OUTPUTS:
        dffTrace, brainMap, VMask, V,W,X,Y,Z,nT
        """
        
        
    #Gather some naming stuff
    #baseDir = '/nrs/ahrens/Weiyu_nrs/20220918_6mMacid_5mMglutamate_f17/exp2' # good
    cellPath = baseDir + "/mika/cells0_clean.hdf5"
    metaPath = baseDir +"/ephys/channel_meta.npy"
    vName = baseDir + "/mika/volume0.hdf5"
    ephysPath = baseDir+'/ephys/eP.26chFlt-v10'

    
    # Load cell time series and ephys data
    f = h5py.File(cellPath,'r')
    a = h5py.File(vName,'r')
    ephysFile = load(ephysPath,26)
    ep = np.load(metaPath, allow_pickle = True).item()
    
    #Read data components
    F=f['cell_timeseries']
    base_f=f['cell_baseline']
    X=f['cell_x']
    Y=f['cell_y']
    Z=f['cell_z']
    brainMap=a['volume_mean'][:,:,:].T
    VMask=a['volume_mask'][:,:,:].T
    V=f['volume_weight']
    W=f['cell_weights'][()]
    
    # Pull out some basic aspects of the data
    nCells = X.shape[0]
    nT = F.shape[1]

    
    # Compute df/f using response time series and baseline
    dffTrace=(F[:,:]-base_f[:,:])/(base_f[:,:]-100) #100 is camera backgroud in spim2
    
    
    return dffTrace, ep, ephysFile,brainMap, VMask, V,W,X,Y,Z,nT,nCells,F



def pullTimes(ephysFile, lsChannel, uvChannel,galvoChannel,thresh):
    """Pulls the timing of imaging frames, uv pulses, and galvo positions. Also finds which stacks occur concurrently with the UV
    INPUTS:
        ephysFile : ch janelia form ephys file
        lsChannel : lightsheet channel in ephys file
        uvChannel : uv chennel in ephys file
        galvoChannel : galvo channel in ephys file
        thresh : thresh for detecting changes
    OUTPUTS:
        stackTimes : stackTimes (in ephys freq)
        uvTimes : uv times 
        badStacks : which stacks occur with uv
        endGalvo: where the galvo ended up during the imaging stack"""
    lsAbove = np.where(np.append(ephysFile[lsChannel],[thresh+1])>thresh)[0]
    stackTimes = lsAbove[np.where(np.diff(lsAbove)>1)[0]]
    uvAbove = np.where(np.append(ephysFile[uvChannel],[thresh+1])>thresh)[0]
    uvTimes = uvAbove[np.where(np.diff(uvAbove)>1)[0]]
    galvoVals = ephysFile[galvoChannel][uvTimes]
    whichStack = np.zeros(len(uvTimes))
    galvoPos = np.zeros(len(uvTimes))
    #if (np.sum(galvoPos)==0):## issue with alignment of uv and lightsheet
    #    galvoPoses = np.where(ephysFile[galvoChannel]>0)[0]
    #    galvoVals2 = ephysFile[galvoChannel][galvoPoses]
    #    for i in range(0,len(uvTimes)):
    #        galvoVals[i] = galvoVals2[np.where(galvoPoses<uvTimes[i])][-1]
    for i in range(0,len(uvTimes)):
        if uvTimes[i] < stackTimes[-1]:
            whichStack[i] = np.where(stackTimes>uvTimes[i])[0][0]-1
            galvoPos[i] = galvoVals[i]
        #whichStack[i] = np.where((stacktimes > 2) & (A < 8))
    [badStacks, idx] = np.unique(whichStack,return_index=True)
    badStacks = badStacks.astype(int)
    endGalvo = galvoPos[idx]
    return stackTimes, uvTimes, badStacks,endGalvo


def getTrainsTwoTypes(ephysFile,ep,lsChannel,uvChannel,thresh,gutPos,ctrlPos,tailPos,nT):
    """Pulls the stim times of useable trials for gut and ctrl
    INPUTS: 
        ephysFile : ch janelia form ephys file
        ep : ephys meta file
        lsChannel : lightsheet channel in ephys file
        uvChannel : uv chennel in ephys file
        thresh : threshold for detecting ls imaging frames
        gutPos : which integer UV position corresponds to gut pulses
        ctrlPos : same as above, but for ctrl
        nT : length of time series data

    OUTPUTS:
        uvOnGut:times of uv gut pulses
        uvOffGut: times of uv ctrl pulses
        onTrain: times series (nT vector) of uv gut pulses
        offTrain: time series (nT vector) of ctrl pulses
        sp: swim power
        visClosed: closed loop vis
        visOpen: open loop vis
        visISI: stationary grating
        visOpenGut: gut vis
        visISIGut: gut isi
        visOpenCtrl: ctrl vis
        visISICtrl: ctrl isi
        """
    stackTimes, uvTimes,badStacks,galvo = pullTimes(ephysFile,lsChannel,uvChannel,ep['ch_gpos'],thresh)
    uvStackTimes = badStacks[np.where(np.diff(badStacks)>1)[0]]
    galvoPos = galvo[np.where(np.diff(badStacks)>1)[0]]
    uvOnGut = uvStackTimes[np.where(galvoPos==gutPos)[0]]
    uvOffGut =  uvStackTimes[np.where(galvoPos==ctrlPos)[0]]
    uvOnTail= uvStackTimes[np.where(galvoPos==tailPos)[0]]
    if uvOnGut[-1]+50<nT:
        uvOnGut=uvOnGut[:-1]
    onTrain = np.zeros(nT)
    offTrain = np.zeros(nT)
    tailTrain = np.zeros(nT)
    for i in range(0,len(uvOnGut)):
        onTrain[uvOnGut[i]] = 1
    for i in range(0,len(uvOffGut)):
        offTrain[uvOffGut[i]]=1
    for i in range(0,len(uvOnTail)):
        tailTrain[uvOnTail[i]]=1
    swim1 = ephysFile[0]
    swim2 = ephysFile[1]
    swimPower1 = windowed_variance(swim1)[0]
    swimPower2 = windowed_variance(swim2)[0]
    sp = (swimPower1[stackTimes]+swimPower2[stackTimes])/2
    vsG = ephysFile[15]
    vsV = ephysFile[13]
    visISI = np.zeros(len(vsV))
    visOpen = np.zeros(len(vsV))
    visClosed = np.zeros(len(vsV))
    visClosed[np.where((vsG>0) &(vsV>0))[0]]=1 # Closed Loop
    visOpen[np.where((vsG==0) &(vsV>0))[0]]=1 # Open Loop
    visISI[np.where(vsV==0)[0]]=1 # ISI
    visOpen = visOpen[stackTimes]
    visClosed = visClosed[stackTimes]
    visISI = visISI[stackTimes]
 
    visOpenGut =visClosed.copy()
    visISIGut = visISI.copy()
    visOpenCtrl = visClosed.copy()
    visISICtrl = visISI.copy()
    
    startGutInd = np.where(onTrain>0)[0][0]
    endGutInd = np.where(onTrain>0)[0][-1]
    visOpenCtrl[startGutInd:endGutInd+1] = 0
    visISIGut[:startGutInd-1] = 0
    visISIGut[endGutInd+1:] = 0
    visOpenGut[:startGutInd-1] = 0
    visOpenGut[endGutInd+1:] = 0
    visISICtrl[startGutInd:endGutInd+1]=0
    isGutPeriod = np.zeros(len(visOpen))
    isCtrlPeriod = np.zeros(len(visOpen))
    if uvOffGut[0]<uvOnGut[0]:
        onFirst = 0
        isCtrlPeriod[:uvOnGut[0]]=1
        isGutPeriod[uvOnGut[0]:]=1
    
    visGut = visOpen.copy()
    visGut*=isGutPeriod
    visISIGut = visISI.copy()
    visISIGut *=isGutPeriod
    visISICtrl = visISI.copy()*isCtrlPeriod
    visCtrl = visOpen.copy()*isCtrlPeriod

    
    return uvOnGut,uvOffGut,uvOnTail, onTrain, offTrain,tailTrain,sp,visClosed, visISI,visGut,visISIGut,visISICtrl,visCtrl,visOpen



def getTrains(ephysFile,ep,lsChannel,uvChannel,thresh,gutPos,ctrlPos,nT):
    """Pulls the stim times of useable trials for gut and ctrl
    INPUTS: 
        ephysFile : ch janelia form ephys file
        ep : ephys meta file
        lsChannel : lightsheet channel in ephys file
        uvChannel : uv chennel in ephys file
        thresh : threshold for detecting ls imaging frames
        gutPos : which integer UV position corresponds to gut pulses
        ctrlPos : same as above, but for ctrl
        nT : length of time series data

    OUTPUTS:
        uvOnGut:times of uv gut pulses
        uvOffGut: times of uv ctrl pulses
        onTrain: times series (nT vector) of uv gut pulses
        offTrain: time series (nT vector) of ctrl pulses
        sp: swim power
        visClosed: closed loop vis
        visOpen: open loop vis
        visISI: stationary grating
        visOpenGut: gut vis
        visISIGut: gut isi
        visOpenCtrl: ctrl vis
        visISICtrl: ctrl isi
        """
    stackTimes, uvTimes,badStacks,galvo = pullTimes(ephysFile,lsChannel,uvChannel,ep['ch_gpos'],thresh)
    uvStackTimes = badStacks[np.where(np.diff(badStacks)>1)[0]]
    galvoPos = galvo[np.where(np.diff(badStacks)>1)[0]]
    uvOnGut = uvStackTimes[np.where(galvoPos!=ctrlPos)[0]]
    uvOffGut =  uvStackTimes[np.where(galvoPos==ctrlPos)[0]]
    if uvOnGut[-1]-50>nT:
        uvOnGut=uvOnGut[:-1]
    onTrain = np.zeros(nT)
    offTrain = np.zeros(nT)
    for i in range(0,len(uvOnGut)):
        onTrain[uvOnGut[i]] = 1
    for i in range(0,len(uvOffGut)):
        offTrain[uvOffGut[i]]=1
    swim1 = ephysFile[0]
    swim2 = ephysFile[1]
    swimPower1 = windowed_variance(swim1)[0]
    swimPower2 = windowed_variance(swim2)[0]
    sp = (swimPower1[stackTimes]+swimPower2[stackTimes])/2
    vsG = ephysFile[15]
    vsV = ephysFile[13]
    visISI = np.zeros(len(vsV))
    visOpen = np.zeros(len(vsV))
    visClosed = np.zeros(len(vsV))
    visClosed[np.where((vsG>0) &(vsV>0))[0]]=1 # Closed Loop
    visOpen[np.where((vsG==0) &(vsV>0))[0]]=1 # Open Loop
    visISI[np.where(vsV==0)[0]]=1 # ISI
    visOpen = visOpen[stackTimes]
    visClosed = visClosed[stackTimes]
    visISI = visISI[stackTimes]
 
    visOpenGut =visClosed.copy()
    visISIGut = visISI.copy()
    visOpenCtrl = visClosed.copy()
    visISICtrl = visISI.copy()
    
    startGutInd = np.where(onTrain>0)[0][0]
    endGutInd = np.where(onTrain>0)[0][-1]
    visOpenCtrl[startGutInd:endGutInd+1] = 0
    visISIGut[:startGutInd-1] = 0
    visISIGut[endGutInd+1:] = 0
    visOpenGut[:startGutInd-1] = 0
    visOpenGut[endGutInd+1:] = 0
    visISICtrl[startGutInd:endGutInd+1]=0
    isGutPeriod = np.zeros(len(visOpen))
    isCtrlPeriod = np.zeros(len(visOpen))
   
    
    visGut = visOpen.copy()
    visGut*=isGutPeriod
    visISIGut = visISI.copy()
    visISIGut *=isGutPeriod
    visISICtrl = visISI.copy()*isCtrlPeriod
    visCtrl = visOpen.copy()*isCtrlPeriod

    
    return uvOnGut,uvOffGut, onTrain, offTrain,sp,visClosed, visISI,visGut,visISIGut,visISICtrl,visCtrl,visOpen

def shift(xs, n):
    """Helper fucntion to construct shifted time series. pads with zeros.
    Inputs: 
        xs : time series
        n : shift degree
    Outputs: 
        e: shifted time series"""
    
    e = np.empty_like(xs)
    if n >= 0:
        e[:n] = 0
        e[n:] = xs[:-n]
    else:
        e[n:] = 0
        e[:n] = xs[-n:]
    return e





class fusedLoocv:
    """Runs 'Fused' regression with LOOCV.
    INPUTS (init):
        order : length of longest lag period (for gut and ctrl UV)
        lambdaRidge : ridge penalty
        lambdaSmooth : smoothing penalty
        
    INPUTS (running models):
        cellData: nCell x nT matrix of cell data
        fullReg: nT x nDim matrix of regression data for the full model 
        fullType: nDim vector of lag orders for the full model
        partReg: nT x nDim matrix of regression data for the part model
        partType: nDim vector of lag orders for the part model
        trialTimes: gut pulse times"""
    
    def __init__(self,order,lambdaRidge,lambdaSmooth):
        "Init"
        self.order = order
        self.lambdaRidge = lambdaRidge
        self.lambdaSmooth = lambdaSmooth
        
    
    
    def constructLag(self,toReg,n):
        """Constructs lag matrix for regression for a single dimension
        INPUTS: 
            toReg : vector of data to shift
            n : shift order
        OTUPUTS:
            xMat : lag matrix for a single dimension"""
        xMat = np.zeros((len(toReg),n))
        xMat[:,0] = toReg
        for i in range(1,n):
            xMat[:,i] = shift(toReg,i)
        return xMat

    def makeInputMats(self,testReg,trainReg,typeVec):
        """Constructs the lag regression matrix from individual regression vectors
        INPUTS:
            testReg : regression matrix for testing
            trainReg : regression matrix for training
            typeVec : vector of lag orders for testReg and trainReg
        OUTPUTS:
            trainXMat : final regression matrix for training of size (nT x sum(typeVec)+1)
            testXMat : same as above for testing"""
        trainXMat = []
        testXMat = []
        if trainReg.ndim>1:
            for i in range(0,len(trainReg)):
                if typeVec[i]>1:
                    trainXMat.append(self.constructLag(trainReg[i,:],typeVec[i]))
                    testXMat.append(self.constructLag(testReg[i,:],typeVec[i]))
                else:
                    temp = np.zeros((trainReg.shape[1],1))
                    temp2 = np.zeros((testReg.shape[1],1))
                    temp[:,0] = trainReg[i,:]
                    temp2[:,0] = testReg[i,:]
                    trainXMat.append(temp)
                    testXMat.append(temp2)
        else:
            trainXMat.append(self.constructLag(trainReg,typeVec[0]))
            testXMat.append(self.constructLag(testReg,typeVec[0]))
        trainXMat = np.hstack(trainXMat)
        testXMat = np.hstack(testXMat)

            

        return trainXMat, testXMat
    

    def makePsiMats(self,fullType,partType):
        """Construct Psi matrices (difference matrices + ridge matrices)
        INPUTS:
            fullType: vector of lag orders for the full model
            partType : vector of lag orders for the part model"""
        fullD = np.zeros((np.sum(fullType),np.sum(fullType)))
        partD = np.zeros((np.sum(partType),np.sum(partType)))
        testD = np.zeros((self.order,self.order))
        for i in range(1,self.order-1):
            testD[i,i] = 2
            testD[i-1,i]=-1
            testD[i+1,i] = -1
        testD[0,0] = 1
        testD[1,0] = 1
        testD[self.order-1,self.order-1]=1
        testD[self.order-2,self.order-1]=-1
        fullD[:self.order,:self.order] = testD
        fullD[self.order:2*self.order,self.order:2*self.order] = testD
        partD[:self.order,:self.order] = testD
        fullD = fullD*self.lambdaSmooth+ np.eye(np.sum(fullType))*self.lambdaRidge
        partD=partD*self.lambdaSmooth + np.eye(np.sum(partType))*self.lambdaRidge
        return fullD,partD
    
    
    
    def runBothModels(self,cellData,fullReg,fullType,partReg,partType,trialTimes):
        """Run the model
        INPUTS:
            cellData : nCell x nT matrix of cell data
            fullReg : raw regression data for the full model (nT x nDim)
            partReg : same, but for the part model
            fullType: vector of lag orders for hte full model
            partType: same, but for part model
            trialTimes : times for UV gut pulses for LOOCV
        OUTPUTS:
            betaHats : full coefficient matrix of model for full model (nDim x nTrials x nCell)
            betaHats2 : same as above, but for the part model
            yHats: yHat data for full model (nT x nTrials x nCells)
            yHats2 : same as above for part model 
            allY : true data (nT x nTrials x nCells)
            rmse : rmse matrix for model 1
            rmse2 : rmse for model2
            corr : corr for model 1
            corr2 : corr for model2
            data : data used in regression
            reg1 : reg mat for 1
            reg2 : 
            type1 : lag orders for 1
            type2 : lag order for 2
            """
        
        self.reg1 = fullReg
        self.reg2 = partReg
        self.typ1 = fullType
        self.type2 = partType
        self.trialTimes= trialTimes
        nCells = len(cellData)
        fullRMSE = np.zeros((len(trialTimes),len(cellData)))
        partRMSE = np.zeros((len(trialTimes),len(cellData)))
        fullCorr = np.zeros((len(trialTimes),len(cellData)))
        partCorr = np.zeros((len(trialTimes),len(cellData)))
        allFullBetaHats = np.zeros((np.sum(fullType),len(trialTimes),nCells))
        allPartBetaHats = np.zeros((np.sum(partType),len(trialTimes),nCells))
        allFullYHats = np.zeros((self.order+10,len(trialTimes),nCells))
        allPartYHats = np.zeros((self.order+10,len(trialTimes),nCells))
        allY = np.zeros((self.order+10,len(trialTimes),nCells))
        ## Get Psi
        fullPsi, partPsi = self.makePsiMats(fullType,partType)
        for i in range(0,len(trialTimes)): # Loop over trials
            print('Running trial '+ str(i+1) +' of '+ str(len(trialTimes)+1))
            toDel = np.arange(trialTimes[i]-10,trialTimes[i]+self.order,1)
            trainData = np.delete(cellData.copy(),toDel,axis=1)
            testData = cellData[:,trialTimes[i]-10:trialTimes[i]+self.order]
            fullTrainReg = np.delete(fullReg.copy(),toDel,axis=1)
            fullTestReg = fullReg[:,trialTimes[i]-10:trialTimes[i]+self.order]
            toDelPart = np.arange(trialTimes[i]-10,trialTimes[i]+self.order,1)
            trainDataPart = np.delete(cellData.copy(),toDelPart,axis=1)
            if partReg.ndim >1:
                partTrainReg = np.delete(partReg.copy(),toDelPart,axis=1)
                partTestReg = partReg[:,trialTimes[i]-10:trialTimes[i]+self.order]
            else:
                partTrainReg =np.delete(partReg.copy(),toDelPart)
                partTestReg = partReg[trialTimes[i]-10:trialTimes[i]+self.order]
            
            fullTrainX, fullTestX = self.makeInputMats(fullTestReg,fullTrainReg,fullType)
            partTrainX, partTestX = self.makeInputMats(partTestReg,partTrainReg,partType)

            fullBetaHat = np.dot(np.linalg.inv(np.dot(fullTrainX.T,fullTrainX)+fullPsi),np.dot(fullTrainX.T,trainData.T))
            partBetaHat = np.dot(np.linalg.inv(np.dot(partTrainX.T,partTrainX)+partPsi),np.dot(partTrainX.T,trainDataPart.T))
            fullYHat = np.dot(fullTestX,fullBetaHat)
            partYHat = np.dot(partTestX,partBetaHat)
            
            allFullBetaHats[:,i,:] = fullBetaHat
            allPartBetaHats[:,i,:] = partBetaHat
            allFullYHats[:,i,:] = fullYHat
            allPartYHats[:,i,:] = partYHat
            allY[:,i,:] = testData.T
            
            fullRMSE[i,:] = np.mean(np.sqrt((fullYHat-testData.T)**2),axis=0)
            partRMSE[i,:] = np.mean(np.sqrt((partYHat-testData.T)**2),axis=0)

            fullCorr[i,:] = corr(fullYHat,testData.T)#c(rankY, rankFull)
            partCorr[i,:] = corr(partYHat,testData.T)#c(rankY,rankPart)
            
            self.betaHats = allFullBetaHats
            self.betaHats2 = allPartBetaHats
            self.yHats = allFullYHats
            self.yHats2 = allPartYHats
            self.allY = allY
            self.rmse = fullRMSE
            self.rmse2 = partRMSE
            self.corr = fullCorr
            self.corr2 = partCorr
            self.data = cellData
            
            
            
def corr(a,b):
    """Fast row-wise correlation between two matrices of same size"""
    p1 = a-np.mean(a,axis=0)
    p2 = b-np.mean(b,axis=0)
    num = np.mean(p1*p2,axis=0)
    p3 = np.std(a,axis=0)
    p4 = np.std(b,axis=0)
    den = p3*p4
    c = num/den
    return c


def pullCells(mod,thresh:float,pThresh: float,prePulseInds: int, postPulseInds: int,uvOnGut,uvOffGut):
    """Pulls gut-related cells
    INPUTS: 
        mod : the regression class
        thresh : correlation threshold (full) for pulling cells
        pThresh : alpha value for comparing parts and full models
    OUTPUTS:
        firstPass : cells with corr > thresh
        secondPass : firstPass AND better corr for full model than parts
        thridPass : secondPass AND better rmse for full model than parts"""
    # Pull cells with mean corr > thresh
    mnCorr = np.mean(mod.corr,axis=0)
    firstPass = np.where(mnCorr>thresh)[0]
    
    #Pull cells with statistically better corr to full than parts
    secondPassP = np.zeros(len(firstPass))
    firstPassFull = mod.corr[:,firstPass]
    firstPassParts = mod.corr2[:,firstPass]
    for i in range(0,len(firstPass)):
        d= firstPassFull[:,i]-firstPassParts[:,i]
        secondPassP[i] = scipy.stats.wilcoxon(d,alternative = 'greater')[1]
    secondPass = firstPass[np.where(secondPassP<pThresh)[0]]
    pValues = secondPassP[np.where(secondPassP<pThresh)[0]]

    
    testDFF = mod.data[secondPass,:]
    onPulseResponse = np.zeros((postPulseInds+prePulseInds,len(secondPass),len(uvOnGut)))
    offPulseResponse = np.zeros((postPulseInds+prePulseInds,len(secondPass),len(uvOffGut)))
    for i in range(0,len(uvOnGut)):
        onPulseResponse[:,:,i] = testDFF[:,uvOnGut[i]-prePulseInds:uvOnGut[i]+postPulseInds].T
    for i in range(0,len(uvOffGut)):
        offPulseResponse[:,:,i] = testDFF[:,uvOffGut[i]-prePulseInds:uvOffGut[i]+postPulseInds].T
    rankP = np.zeros(len(secondPass))
    for i in range(0,len(secondPass)):
        rankP[i] = scipy.stats.ranksums(np.max(onPulseResponse[:18,i,:],axis=0),np.max(offPulseResponse[:18,i,:],axis=0),alternative='greater')[1]
    thirdPass = secondPass[np.where(rankP<pThresh)[0]]
    rankP = rankP[np.where(rankP<pThresh)[0]]
    return firstPass, secondPass,thirdPass, pValues, rankP


def pullCellMasks(W,V,X,Y,Z,whichCells):
    """Pulls masks for making brain maps
    INPUTS:
        W,V,W,Y,Z : outputs from Mika's pipeline
        whichCells : which cells to mask (gut-sensitive)
    OUTPUTS:
        maskAll : masks for brain map"""
    maskW = np.zeros(V.shape)
    ww = W[whichCells]
    for i in (range(0,len(whichCells))):
        ww = W[whichCells[i]]
        xx,yy,zz = X[whichCells[i]],Y[whichCells[i]],Z[whichCells[i]]
        maskW[xx,yy,zz] = np.maximum(maskW[xx,yy,zz],ww)
    maskW =np.ma.masked_where((maskW ==0), maskW)
    return maskW 

def makeBrainMap(brainMap,maskAll):
    """Makes a quick brain map from selected cells
    INPUTS:
        brainMap : map from Mika's pipeline
        maskAll : selected cell Mask
    OUTPUTS:
        NA"""
    plt.figure(figsize=(10, 10))
    fig, axs = plt.subplots(nrows=1, ncols=2, sharex='col',figsize=(18, 10 ))
    ax_im1 = axs[0]
    ax_im2 = axs[1]
    im1_m=ax_im1.imshow(brainMap[:,:,:-1].max(axis=2).T, vmin=100,vmax=np.percentile(brainMap[:].squeeze(), 99.98),cmap='gray')
    im1_0 = ax_im1.imshow(maskAll[:,:,:-1].max(axis=2).T, cmap='magma',alpha=0.5,vmin=0,vmax=0.0025)
    im2_m=ax_im2.imshow(np.flip(brainMap[:,:,:-1].max(axis=1).T,axis=0), cmap='gray',vmin =100, vmax=np.percentile(brainMap[:].squeeze(), 99.98),aspect=6,interpolation='nearest')
    im2=ax_im2.imshow(np.flip(maskAll[:,:,:-1].max(axis=1).T,axis=0), vmin = 0,vmax=0.005,cmap='magma',alpha=0.5,aspect=6,interpolation='nearest')





def fusedRidge(y,toReg,orderVec,lambdaRidge,lambdaSmooth):
    """Runs the fused ridge regression
    INPUTS:
        y : matrix (nCells x nT) of neural data
        toReg : matrix (nRegressors x nT) of regressor data
        orderVec : vector (of size nRegressors) of lag orders
        lambdaRidge : ridge regularization strength
        lambdaSmooth : smooth regularization strength
    OUTPUTS:
        betaHat : matrix (sum(orderVec) x nCells)
        yHat : matrix (nT x nCells) of predicted data"""
    psiMat = makePsiMat(orderVec,2,10) # construct psi matrix
    xMat = makeInputMat(toReg,orderVec) #
    betaHat = np.dot(np.linalg.inv(np.dot(xMat.T,xMat)+psiMat),np.dot(xMat.T,y.T))
    yHat = np.dot(xMat,betaHat) 
    return betaHat, yHat,xMat


############################ Fused Ridge Helper Functions #################

def constructLag(toReg,n):
    """Constructs lag matrix for regression for a single dimension
    INPUTS: 
        toReg : vector of data to shift
        n : shift order
    OTUPUTS:
        xMat : lag matrix for a single dimension"""
    xMat = np.zeros((len(toReg),n))
    xMat[:,0] = toReg
    for i in range(1,n):
        xMat[:,i] = shift(toReg,i)
    return xMat

def makePsiMat(orderVec,lambdaRidge,lambdaSmooth):
    """ Constructs Psi Matrix (difference matrix + ridge matrix)
    INPUTS:
        orderVec : vector of lag orders
    OUTPUTS:
        psiMat : psi matrix"""
    psiMat = np.zeros((np.sum(orderVec),np.sum(orderVec)))
    psiMat[:orderVec[0],:orderVec[0]] = makeSinglePsiMat(orderVec[0])
    if len(orderVec)>1:
        for i in range(1,len(orderVec)-1):
            psiMat[np.sum(orderVec[:i]):np.sum(orderVec[:i+1]),np.sum(orderVec[:i]):np.sum(orderVec[:i+1])]= makeSinglePsiMat(orderVec[i])
        psiMat[np.sum(orderVec[:-1]):np.sum(orderVec),np.sum(orderVec[:-1]):np.sum(orderVec)]= makeSinglePsiMat(orderVec[-1])
    psiMat = psiMat * lambdaSmooth + np.eye(np.sum(orderVec))*lambdaRidge
    return psiMat
    
def makeSinglePsiMat(orderScal):
    """Constructs a single D difference matrix
    INPUTS: 
        orderScal : scalar of the lag order
    OUTPUTS:
        partPsi : single D matrix"""
    partPsi = np.zeros((orderScal,orderScal))
    for i in range(1,orderScal-1):
        partPsi[i,i] = 2
        partPsi[i-1,i] = -1
        partPsi[i+1,i] = -1
        partPsi[0,0] = 1
        partPsi[1,0] = -1
        partPsi[orderScal-1][orderScal-1]=1
        partPsi[orderScal-2][orderScal-1]=-1
    return partPsi

def makeInputMat(toReg,orderVec):
    """Constructs the lag regression matrix from individual regression vectors
    INPUTS:
        toReg : regression matrix for training
        orderVec : vector of lag orders for testReg and trainReg
    OUTPUTS:
        xMat : final regression matrix for training of size (nT x sum(typeVec)+1)"""
    xMat =[]
    if toReg.ndim>1: #there are more than one regression time series
        for i in range(0,len(toReg)):
            if orderVec[i]>1:
                xMat.append(constructLag(toReg[i,:],orderVec[i]))
            else:
                temp = np.zeros((len(toReg[i,:]),1))
                temp[:,0] = toReg[i,:]
                xMat.append(temp)
    else:
        xMat.append(constructLag(toReg,orderVec[0]))
    return np.hstack(xMat)




############################ Fused Ridge Output Analysis Functions #################
def gaus(x,a,x0,sigma):
    """Normal distribution function, used for curve fitting"""
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def halfGaus(x,x0,sigma):
    """Normal distribution function, used for curve fitting"""
    return np.sqrt(2)/(np.sqrt(np.pi)*sigma)* np.exp(-(x-x0)**2/(2*sigma**2))


def fitLeftGetThresh(corrs,stdThresh):
    """Estimates a correlation threshold for significantly correlated data. Relies upon
        fitting the left hand side of a normal distribution (relative to mode) for an estimate
        of noise variance
    INPUTS:
        corrs : vector of corr coefs
        stdThresh : threshold on STD for estimate (probabilities related to z-score"""
    counts, bins,bars= plt.hist(corrs,density=True,bins=100,alpha = .6,label = "Full")
    mode = bins[np.where(counts==np.max(counts))]+ (bins[1]-bins[0])/2
    leftDat = corrs[np.where(corrs<mode)[0]]
    countsL,binsL,barsL = plt.hist(leftDat,density=True,bins=100,alpha = .6,label = "Full")
    popt,pcov = curve_fit(halfGaus,binsL[:-1],countsL,p0=None, sigma=None) 
    plt.plot(bins,halfGaus(bins,popt[0],popt[1]))
    thresh = popt[0]+(np.abs(popt[1])*stdThresh)
    plt.plot([thresh,thresh],[0,1])
    return thresh, popt[0],popt[1]



def pullAverages(dffTrace,uvOnGut,uvOffGut,preTime,postTime):
    """Pulls average and se gut and ctrl pulses"""
    gutResponses = np.zeros((len(dffTrace),postTime+preTime,len(uvOnGut)))
    ctrlResponses = np.zeros((len(dffTrace),postTime+preTime,len(uvOffGut)))
    for i in range(0,len(uvOnGut)):
        gutResponses[:,:,i] = dffTrace[:,int(uvOnGut[i]-preTime):int(uvOnGut[i]+postTime)]
    for i in range(0,len(uvOffGut)):
        ctrlResponses[:,:,i] = dffTrace[:,int(uvOffGut[i]-preTime):int(uvOffGut[i]+postTime)]
    aveGutResponse = np.mean(gutResponses,axis=2)
    seGutResponse = np.std(gutResponses,axis=2)/np.sqrt(len(uvOnGut))
    aveCtrlResponse = np.mean(ctrlResponses,axis=2)
    seCtrlResponse = np.std(ctrlResponses,axis=2)/np.sqrt(len(uvOffGut))
    return aveGutResponse, seGutResponse,aveCtrlResponse,seCtrlResponse



def meanAndSE(mat,trigs,preTime,postTime):
    trialResponses = np.zeros((len(mat),postTime+preTime,len(trigs)))
    for i in range(0,len(trigs)):
        trialResponses[:,:,i] = mat[:,trigs[i]-preTime:trigs[i]+postTime]
    trialMeans = np.mean(trialResponses,axis=2)
    trialSEs = np.std(trialResponses,axis=2)/np.sqrt(len(trigs))
    return trialResponses, trialMeans, trialSEs