import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import copy
import sys
from scipy.stats import norm
import time
import seaborn as sns
import pandas as pd

class BaseClass:
    """Class for different sampling algorithms"""

    def __init__(self,model,qstart,qlims,nsamples,output_file="samples.txt",
                        printoutput=True,plotfigures=True,saveoutputfile=True):
        """
        Construct an object from sampling method

        qstart: Initial value of sampling
        qlims: Limits of the variables
        nsamples: Number of samples
        Vstart: Variance matrix
        method: Choice of method for sampling
        output_file: File name to save the samples
        """

        self.model=model
        self.qstart=np.reshape(np.asarray(qstart),(-1,1))
        self.qlims=qlims
        self.nsamples=nsamples
        self.printoutput=printoutput
        self.plotfigures=plotfigures
        self.saveoutputfile=saveoutputfile
        self.sp=2.38**2/self.qstart.shape[0]

        os.chdir("..")
        outputpath= 'results'
        if os.path.isdir(outputpath)==False:
            print(outputpath ,"directory doesnot exist, creating one to save the output")
            os.mkdir(outputpath)
        os.chdir("code")
        self.outputpath="../"+outputpath+"/"
        self.output_file=self.outputpath+output_file
    
    def mindist(self,qsamples):
        from scipy.spatial.distance import cdist

        qsamples_set=[]
        for i in range(0,qsamples.shape[0]):
            qsamples_set.append(tuple(qsamples[i,:]))
        
        dum_mindist=cdist(qsamples_set,qsamples_set)
        dum_mindist=np.min(np.sort(dum_mindist,axis=1)[:,1])

        return dum_mindist

    def maxdist(self,qsamples):
        from scipy.spatial.distance import cdist

        qsamples_set=[]
        for i in range(0,qsamples.shape[0]):
            qsamples_set.append(tuple(qsamples[i,:]))
        
        dum_maxdist=cdist(qsamples_set,qsamples_set)
        dum_maxdist=np.max(np.sort(dum_maxdist,axis=1)[:,-1])

        return dum_maxdist

    def printoutput(self,qsamples,ntot,neff,accpt_ratio,size=1,comp_time=1):
        
        """
        Function to call the sampling method of choice and plot results
        """

        qsamples=np.unique(qsamples,axis=0)

        # dum_mindist=self.mindist(qsamples)
        # dum_maxdist=self.maxdist(qsamples)
        # comp_array=np.array([[size,comp_time,dum_mindist,dum_maxdist]])
        comp_array=np.array([[size,comp_time]])
        np.savetxt("../results/comp_time.log",comp_array)

        if self.printoutput:
            print("Total generated samples:",ntot)
            print("Total effective samples:",int(neff))
            print('Acceptance ratio:',accpt_ratio)
            print("Computational time:",comp_time)
            # print("Min. distance:",dum_mindist)
            # print("Max. distance:",dum_maxdist)

        self.testsamples(qsamples)

    def testsamples(self,qsamples):

        testinds=np.random.randint(qsamples.shape[0],size=100)
        print("---------Testing the generated samples randomly for",len(testinds),"samples---------")
        totfailed=0
        for ind in testinds:
            testval=self.model.apply(list(qsamples[ind,:]))
            testfunval=self.model.apply_eval(list(qsamples[ind,:]))
            if np.min(testfunval)<-1e-6:
                print("Sampling failed for sample number:",ind)
                print("Constrain values")
                print(self.model.apply_eval(list(qsamples[ind,:])))
                totfailed=totfailed+1
        print("Total failed:",totfailed)
        return

    def plotsamples(self,qsamples,method):
        """
        Function for plotting the samples
        """
        if self.plotfigures:
            filname=self.output_file[:-4]+".png"
            print("Plots saved to the folder:",filname)
            fig=sns.pairplot(pd.DataFrame(qsamples), markers='o')
            fig.savefig(filname)
        else:
            print("Plots are not generated")
        return

    def savesamples(self,qsamples,method):
        """
        Function for saving the samples
        """
        if self.saveoutputfile:
            filname=self.output_file[:-4]+".txt"
            print("Saving samples to the folder:",filname)
            np.savetxt(filname,qsamples)
        else:
            print("Results are not saved")
        return
