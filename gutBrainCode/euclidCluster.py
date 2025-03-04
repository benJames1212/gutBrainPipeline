# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 13:28:24 2023

@author: jamesb
"""

from sklearn.decomposition import PCA
from scipy.stats import invwishart, invgamma, dirichlet,gamma,expon, norm, multivariate_normal, invgamma
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt



class euclidGibbs:
    """Ben James 2021
    
    Runs a 'Euclidean distance penalized' Gaussian Mixture Model. 
    
    Basic algorithm works as follows:
        Iterate through:
        1) Sample vector mu_k ~ Normal(muN,kappaN) for k in {0,...,nK}
        2) Sample matrix sigma_k ~ Inverse-Wishart(nuN, psiN) for k in {0,...,nK}
        3) Sample pi_k ~ Dirichlet(alphaN) for k in {0,...,nK}
        4) Sample lambda_k ~ Gamma(alphaN, betaN)
        5) Sample z_i with p(z_i=k) = N(data_i;mu_k,sigma_k)Exp(pairwise_distance;lambda_k)Cat(pi_k) for i in {0,...,nData}
    
    INPUTS: 
        nK : number of clusters for a single run (absorbed through multi-runs)
        nIts : number of iterations
        lag : time between recording samples
        burnIn: time before recording samples
        respondeData : nCells x nDims array of PCA data
        euclidMat : nCells x 3 array of cell locations
    OUTPUTS:
        allZ : nCells x nIts list of recorded iterations
        logLikelihood : trace of l(R|d) for recorded iterations
        isSame : matrix of p(cells in same cluster)
        
        
    Hyperparameters (Normal prior on Normal Mean): 
        mu0 : prior belief over mean
        kappa0 : prior belief over variance about the mean (how much we expect our prior to be true)
    Hyperparameters (Inverse Wishart prior on Normal covariance)
        psi0 : prior belief about covariance
        nuN : : prior belief about divergence of covariance from our prior
    Hyperparameters (Dirichlet on Categorical):
        alpha0 : 'prior occupancies of each cluster'. Uniform means equally likely, with increasing value increasing belief
    HyperParameters (Gamma on Exponential)
        alpha0 : 
        beta0 : """
    
    def __init__(self,nK:int, nIts:int, lag:int, burnIn:int):
        # Initialize base model parameters
        self.nK = nK
        self.nIts = nIts
        self.lag = lag
        self.burnIn = burnIn
        
        
        
    def initializeOthers(self,responseData, euclidData):
        # Initialize other things (based on responseData and euclidData)
        [nDats,nDims] = responseData.shape
        
        ## Init hyperparameters
        # For Normal/normal-inverse-wishart
        self.muNaught = 0
        self.kappaNaught = 1
        self.nuNaught = 1
        self.psiNaught = np.cov(responseData, rowvar = False)
        self.muN = np.zeros((self.nK,nDims))
        self.nuN = np.zeros(self.nK)
        self.kappaN = np.zeros(self.nK)
        self.psiN = np.zeros((self.nK,nDims,nDims))
        self.nIn = np.zeros(self.nK)
        self.s = np.zeros((self.nK,nDims,nDims))
        # For Dirichlet - 
        self.dirAlpha = 10 #,50
        # For exponential
        self.gammaAlpha = 1#1.5, 2
        self.gammaBeta =50 #4, 2
        self.alphaN = np.zeros(self.nK)
        self.betaN = np.zeros(self.nK)
        self.llambda = np.zeros(self.nK)
  
        # Compute euclidean matrix data
        self.euclidMat = euclidean_distances(euclidData,euclidData)

        # Initialize clusters, means, pi, covariances
        self.z = np.random.choice(self.nK,responseData.shape[0])
        self.pi = np.ones(self.nK)/self.nK
        self.means = np.zeros((self.nK,nDims))
        self.covars = np.zeros((self.nK,nDims,nDims))
        
        return self
    
    def sampleGammaDirichlet(self):
        # Sample lambda ~ gamma(alphaN, betaN) and pi ~ Dirichlet(alphaN)
        for k in range(self.nK): # Loop over clusters, pull stats
            # Pull alphaN, betaN for dirichelt and gamma
            isInCluster = np.where(self.z==k)[0]
            self.nIn[k] = len(isInCluster)
            nDistMat = self.euclidMat[np.ix_(isInCluster,isInCluster)]
            distVec = nDistMat[np.triu_indices(len(nDistMat),1)]
            self.alphaN[k] =  self.gammaAlpha+self.nIn[k]
            self.betaN[k] = self.gammaBeta + self.nIn[k]*np.mean(distVec)
            # Sample lambda
            self.llambda[k] = gamma.rvs(self.alphaN[k],scale=1/self.betaN[k],size=1)
        # Sample pi
        self.pi= dirichlet.rvs(self.nIn+self.dirAlpha,size=1)[0]
        
        return self
            
    
    def sampleNormalInverseWishart(self,responseData):
        # Sample mu ~ Normal(muN,kappaN) and sigma ~ Inverse-Wishart(psiN,nuN)
        for k in range(self.nK): # Loop over clusters, pull stats
            # Compute muN, kappaN, psiN, and nuN
            isInCluster = np.where(self.z==k)[0]
            self.nIn[k] = len(isInCluster)
            self.means[k,:] = np.mean(responseData[isInCluster,:])
            resid = (self.means[k,:]-responseData[isInCluster,:]).T
            self.s[k,:,:] = np.dot(resid, resid.T)
            self.muN[k,:] = (self.kappaNaught * self.muNaught + self.nIn[k]*self.means[k,:])/(self.kappaNaught+self.nIn[k])
            self.kappaN[k] = self.kappaNaught+self.nIn[k]
            self.nuN[k] = self.nuNaught+self.nIn[k]
            grandResid = (self.means[k,:]-self.muNaught).T
            self.psiN[k,:,:] = self.psiNaught + self.s[k,:,:]+ (self.kappaNaught*self.nIn[k])/(self.kappaNaught*self.nIn[k])*grandResid
            ## sample covariance matrix
            self.covars[k,:,:] = invwishart.rvs(self.nuN[k],self.psiN[k,:,:],size=1)
            # Sample mean vector
            self.means[k,:] = multivariate_normal.rvs(self.muN[k,:],self.covars[k,:,:]/self.kappaN[k],size=1)
        return self
    
    
    def updateClustering(self,responseData):
        # Sample occupancy vector for each datum
        [nDats,nDims] = responseData.shape
        
        for i in range(nDats):
            pVec = np.zeros(self.nK)
            exponP = np.zeros(self.nK)
            newZ = self.z.copy()
            newZ[i]=-1
            for k in range(0,self.nK):
                isInCluster = np.where(newZ==k)[0]
                self.nIn[k] = len(isInCluster)
                distVec = self.euclidMat[i,isInCluster]
                exponP[k] =  np.sum(np.log(expon(scale=1/self.llambda[k]).pdf((distVec))))
                pVec[k]= np.log(self.pi[k]) +exponP[k]+ np.log(multivariate_normal(self.muN[k,:],self.covars[k,:,:]).pdf(responseData[i,:]))
            # Log sum exp trick
            pVec-=np.max(pVec)
            pVec = np.exp(pVec)
            pVec/=np.sum(pVec)
            # Sample z
            self.z[i] = np.random.choice(range(0,self.nK),p=pVec)
        
        return self
    
    def logLike(self,responseData):
        # Compute log likelihood of the model
        logLik = 0
        [nDats,nDims] = responseData.shape
        exponP = np.zeros(self.nK)
        for k in range(0, self.nK):
            isInCluster = np.where(self.z==k)[0]
            self.nIn[k] = len(isInCluster)
            nDistMat = self.euclidMat[np.ix_(isInCluster,isInCluster)]
            distVec = nDistMat[np.triu_indices(len(nDistMat),1)]
            exponP[k] =  np.sum(np.log(expon(scale=1/self.llambda[k]).pdf((distVec))))
        for i in range(nDats):
            logLik+=np.log(self.pi[self.z[i]]) + np.log(multivariate_normal(self.muN[self.z[i],:],self.covars[self.z[i],:,:]).pdf(responseData[i,:]))
        logLik += np.sum(exponP)
        return logLik
    
    def makeSimMat(self):
        # Constructs 'similarity matrix' : p(cells in same cluster)
        [nDats,nRuns] = self.allZ.shape
        self.isSame = np.zeros((nDats,nDats))

        for i in range(0,nRuns):
            for j in range(0,self.nK):
                isIn = np.where(self.allZ[:,i]==j)[0]
                self.isSame[np.ix_(isIn,isIn)]+=1
        self.isSame/=nRuns
        return self
    
    
    def runModel(self,responseData,euclidData):
        # Runs main model
        # Initailize some other stuff
        self.initializeOthers(responseData,euclidData)
        # for recording
        self.logLikelihood = []
        nSamps = int((self.nIts-self.burnIn)/self.lag)+1
        self.allZ = np.zeros((len(responseData),nSamps))
        onWhich = 0
        # Loop over iterations
        for i in range(self.nIts):
            print(f"On iteration {i+1} of {self.nIts}")
            ####Sample means, covs
            self.sampleNormalInverseWishart(responseData)
            # sample pi, lambda
            self.sampleGammaDirichlet()
            #Update clusters
            self.updateClustering(responseData)
            logLik = self.logLike(responseData)
            self.logLikelihood.append(logLik)
            if ((i>=self.burnIn) and ((i-self.burnIn) % self.lag==0)):
                dd = self.z.copy()
                self.allZ[:,onWhich] = dd
                onWhich+=1
        self.makeSimMat()
        
        return self


def pullCenters(X,Y,Z):
    """Pulls the mean X, Y, and Z coordinates for a set of cells
    INPUTS:
        X,Y,Z : mats of loc data
    outPuts:
        cellCenters : mean X,Y, and Z locations for each cell in X,Y,Z"""
    cellCenters = np.zeros((len(X),3))
    for i in range(0,len(X)):
        cellCenters[i,:] = pullLocs(X[i],Y[i],Z[i])
    return cellCenters

def pullLocs(X,Y,Z):
    """Computes mean locs from data
    INPUTS:
        X,Y,Z : mats of loc data
    outPuts:
        xG,yG,zG : mean locs of cells
    """
    xx = X
    yy = Y
    zz = Z
    xG = np.mean(xx[np.where(xx>=0)[0]])
    yG = np.mean(yy[np.where(yy>=0)[0]])
    zG = np.mean(zz[np.where(zz>=0)[0]])
    return xG, yG,zG
    
    
def gatherForCluster(X,Y,Z,whichCells,dffTrace,pcaComps = 10):
    """Reformats data to use for euclid gibbs clustering
    INPUTS:
        X,Y,Z : coords of each cell from Mika's pipeline
        whichCells: which cells to cluster
        dffTrace : dffTrace for all cells
        nComps : number of temporal components to reduce dffTrace to. Default is 10"""
    centers=pullCenters(X[whichCells],Y[whichCells],Z[whichCells])
    cellCentersRefl = centers.copy()
    for j in range(0,len(centers)):
        cellCentersRefl[j,1] = np.abs(250-centers[j,1])
    cellCentersRefl[:,0] *=0.40625
    cellCentersRefl[:,1] *=0.40625
    cellCentersRefl[:,2] *=6
    cellDFF=dffTrace[whichCells,:]
    pca = PCA(n_components=10)
    pcaTraces = pca.fit_transform(cellDFF)
    return centers, cellCentersRefl,pcaTraces


def plotStuffFromCluster(cellCenters,brain_map,labels):
    """Plots identified brain clusters and responses (from pullForPlots)
    INPUTS:
        labels : cluster outputs from GMM
        mnResp : 50 x nComp mean response to each stim for all cells in cluster
        sdResp: same as above, but for sd
        cellCenters : mean locations of each cell
        brain_map : brain map (from Mika's pipeline)
    OUTPUTS:
        N/A
    """
    #ucL = np.unique(labels)
    # Make fig 
    last_plane=-1
    fig, (ax_im1,ax_im2) = plt.subplots(nrows=1, ncols=2,sharex='col',figsize=(20, 20 ))
    #First fig
    im1_m=ax_im1.imshow(brain_map[:,:,:last_plane].max(axis=2).T, vmax=np.percentile(brain_map[:].squeeze(), 100),cmap='gray')
    scatter = ax_im1.scatter(cellCenters[:, 0],cellCenters[:, 1], c=labels, s=10);
    # produce a legend with the unique colors from the scatter
    #legend1 = ax_im1.legend(*scatter.legend_elements(),loc="lower left", title="Classes")
    #ax_im1.add_artist(legend1)
    #Second Fig
    ax_im2.imshow(brain_map[:,:,:].max(axis=1).T, cmap='gray',vmax=np.percentile(brain_map[:].squeeze(), 99),aspect='auto')
    ax_im2.scatter(cellCenters[:, 0],cellCenters[:, 2], c=labels, s=10)
    ax_im2.set_aspect(5)
    plt.show()
    xx = range(-10,40)