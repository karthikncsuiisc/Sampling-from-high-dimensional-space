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
from baseclass import BaseClass

class Metropolis(BaseClass):
    """Class for different sampling algorithms"""

    def __init__(self,model,qstart,qlims,nsamples,Vstart=None,output_file="samples.txt",
                     method="Metropolis",adapt_interval=100,printoutput=True,plotfigures=True,
                     saveoutputfile=True):
        """
        Inhereted from BaseClass
        """
        BaseClass.__init__(self,model=model,
                                qstart=qstart,
                                qlims=qlims,
                                nsamples=nsamples,
                                output_file=output_file,
                                printoutput=printoutput,
                                plotfigures=plotfigures,
                                saveoutputfile=saveoutputfile)
        if Vstart==None:
            self.Vstart=np.eye(len(qstart))*1.0/len(qstart)
        else:
            self.Vstart=Vstart
        self.adapt_interval=adapt_interval
        self.method=method

    def sample(self):
        """
        Function to call the sampling method of choice and plot results
        """
        if self.method=="Metropolis":
            print("Sampling using random walk metropolis")
            qsamples,ntot,naccept=self.RandomWalk()
        elif self.method=="AdaptiveMetropolis":
            print("Sampling using adaptive metropolis")
            qsamples,ntot,naccept=self.AdaptiveMetropolis()
        else:
            print("Wrong choice of method for sampler algorithm")
            sys.exit(1)
        
        super(Metropolis, self).printoutput(ntot,naccept,naccept/ntot)
        super(Metropolis, self).plotsamples(qsamples,self.method)
        super(Metropolis, self).savesamples(qsamples,self.method)
        return

    def RandomWalk(self):        
        """
        Function for sampling using random walk Metropolis algorithm

        Return:
            Q_MCMC: Accepted samples
        """
        R = np.linalg.cholesky(self.Vstart)
        q_old =self.qstart
        iaccept=0
        ireject=0
        qsamples=[list(q_old[:,0])]

        pbar = tqdm(total = self.nsamples)
        while iaccept<self.nsamples:
            z = np.random.randn(q_old.shape[0],1)
            q_new = q_old + np.dot(R.T,z)
            output=self.objF(q_new[:,0])
            if output:
                q_old=copy.deepcopy(q_new)
                qsamples.append(list(q_old[:,0]))
                iaccept=iaccept+1
                pbar.update(1)
            else:
                ireject=ireject+1
        pbar.close()
        qsamples=np.asarray(qsamples)
        return qsamples,iaccept+ireject,iaccept

    def AdaptiveMetropolis(self):        
        """
        Function for sampling using adaptive Metropolis algorithm

        Return:
            Q_MCMC: Accepted samples
        """
        print("adaption interval:",self.adapt_interval)
        R = np.linalg.cholesky(self.Vstart);
        Vold=copy.deepcopy(self.Vstart)
        q_old =self.qstart;
        qsamples=[list(q_old[:,0])];
        iaccept=0
        ireject=0

        #Adaptive metropoils parameters
        qmean=copy.deepcopy(q_old)
        qmean_old=copy.deepcopy(q_old)
        adptintnew=1

        pbar = tqdm(total = self.nsamples)        
        while iaccept<self.nsamples:

            z = np.random.randn(q_old.shape[0],1);
            q_new = q_old + np.dot(R.T,z);
            output=self.objF(q_new[:,0])

            if output:
                k=copy.deepcopy(iaccept)+1               

                #update variance and qmean
                Vupdate=k*np.dot(qmean_old,qmean_old.T)
                Vupdate=Vupdate-(k+1)*np.dot(qmean,qmean.T)
                Vupdate=Vupdate+np.dot(q_new,q_new.T)
                Vupdate=Vupdate+np.eye(len(q_new))*1e-16
                Vupdate=self.sp/k*Vupdate

                Vold=copy.deepcopy(self.Vstart)
                self.Vstart=(k-1.0)/k*self.Vstart+Vupdate
                qmean_old=copy.deepcopy(qmean)
                qmean=k*qmean/(k+1)+q_old/(k+1)

                q_old=copy.deepcopy(q_new)
                qsamples.append(list(q_old[:,0]))
                iaccept=iaccept+1
                pbar.update(1)
            else:
                ireject=ireject+1
           
            adptintnew=norm.cdf(iaccept/self.adapt_interval,0.5,0.2)*self.adapt_interval
            adptintnew=int(max(1.0,adptintnew))

            if np.mod(iaccept,adptintnew)==0:
                try:
                    R = np.linalg.cholesky(self.Vstart)
                except:
                    self.Vstart=copy.deepcopy(Vold)
        pbar.close()
        return qsamples,iaccept+ireject,iaccept

    def objF(self,qval):
        output=np.all(qval>=self.qlims[:,0]) and np.all(qval<=self.qlims[:,1]) and self.model.apply(list(qval))
        return output

