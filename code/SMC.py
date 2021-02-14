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

class SMC(BaseClass):
    """Class for different sampling algorithms"""

    def __init__(self,model,qstart,qlims,nsamples,Vstart=None,output_file="samples.txt",
                     method="SMC",adapt_interval=100,printoutput=True,plotfigures=True,
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

        self.ESSfract=0.5
        self.tauThres=1e6
        self.NSMC_MCMC=5
        self.nsamples=int(self.nsamples/(self.NSMC_MCMC*0.25))

    def sample(self):
        """
        Function to call the sampling method of choice and plot results
        """
        print("Sampling using sequential monte carlo")
        qsamples,ntot,naccept=self.SMCsampler()

        super(SMC, self).printoutput(qsamples,ntot,naccept,naccept/ntot)
        super(SMC, self).plotsamples(qsamples,self.method)
        super(SMC, self).savesamples(qsamples,self.method)
        return naccept,naccept/ntot

    def SMCsampler(self):        
        """
        Function for sampling using sequential monter carlo

        Return:
            Q_MCMC: final samples
        """
        tau=0;
        qprtls=np.random.rand(self.qlims.shape[0],self.nsamples)
        qprtls=np.reshape(self.qlims[:,0],(-1,1))+(np.reshape(self.qlims[:,1],(-1,1))-np.reshape(self.qlims[:,0],(-1,1)))*qprtls[:,:]
        W=np.ones(self.nsamples)*1/self.nsamples

        allaccepts_qs=[]
        tcount=0
        Var=copy.deepcopy(self.Vstart)
        VarR=copy.deepcopy(Var)

        print("-------------------------------------")
        while tau < self.tauThres:
            tcount=tcount+1
            tau,wnt,ESSeff=self.find_tau(qprtls,tauL=tau,tauU=self.tauThres)
            W=W*wnt
            W=W/np.sum(W)

            qprtls,W=self.resampling(qprtls,W)
            Var,VarR=self.calSMCvar(qprtls,W,Var,VarR)
            EFF_prtls=np.unique(qprtls,axis=1).shape[1]

            iacceptall=0
            if np.abs(tau-self.tauThres)/self.tauThres<1e-3:
                for i in range(0,self.nsamples):
                    # qprtls[:,i],iaccept,allaccepts_dum=self.SMCRW(tau,qprtls[:,i],Var,VarR)
                    qprtls[:,i],iaccept,allaccepts_dum=self.GibbsSampler(tau,qprtls[:,i],Var,VarR)
                    iacceptall=iacceptall+iaccept
                    allaccepts_qs=allaccepts_qs+allaccepts_dum
                allaccepts_qs=np.asarray(allaccepts_qs)
                EFF_prtls=np.unique(allaccepts_qs,axis=0).shape[0]
                print("tau:",tau,"Accept. Ratio:",iacceptall/(self.nsamples*self.NSMC_MCMC),"ESS:",ESSeff,"Eff. particles:",EFF_prtls)
            else:
                for i in range(0,self.nsamples):
                    # qprtls[:,i],iaccept,allaccepts_dum=self.SMCRW(tau,qprtls[:,i],Var,VarR)
                    qprtls[:,i],iaccept,_=self.GibbsSampler(tau,qprtls[:,i],Var,VarR)
                    iacceptall=iacceptall+iaccept
                print("tau:",tau,"Accept. Ratio:",iacceptall/(self.nsamples*self.NSMC_MCMC),"ESS:",ESSeff,"Eff. particles:",EFF_prtls)
        print("-------------------------------------")

        ntot=(self.nsamples*self.NSMC_MCMC)*tcount

        return allaccepts_qs,ntot,EFF_prtls

    def find_tau(self,qprtls,tauL,tauU,deltatau=1.0,Niter=100):        
        """
        Function to find tau using ESS

        Return:
            tau: returns
        """
        pitallminus=self.pitallfun(tauL,qprtls)

        funL=1.0-self.ESSfract
        ESS,wnt=self.ESSfun(tauU,qprtls,pitallminus)
        funU=ESS/qprtls.shape[1]-self.ESSfract
        fact=1.0
        if funU>=0.0:
            return tauU,wnt,ESS/qprtls.shape[1]

        for iter in np.arange(0,Niter):

            taunew=tauL+fact*deltatau
            ESS,wnt=self.ESSfun(taunew,qprtls,pitallminus)
            funnew=ESS/qprtls.shape[1]-self.ESSfract

            if funnew<0:
                fact=fact/2.0
                tauU=copy.deepcopy(taunew)
                funU=copy.deepcopy(funnew)
            elif funnew>0:
                fact=fact*2.0
                tauL=copy.deepcopy(taunew)
                funL=copy.deepcopy(funnew)
            
            if np.abs(funnew)<1e-3:
                break
        
        return taunew,wnt,ESS/qprtls.shape[1]
    
    def ESSfun(self,tau,qprtls,pitallminus):
        """
        Calcualtes effective samples

        Return:
            ESS: returns
        """
        pitall=self.pitallfun(tau,qprtls)
        wnt=pitall/(pitallminus+1e-16)
        ESS=(np.sum(wnt))**2/(np.sum(wnt**2)+1e-16)
        return ESS,wnt

    def pitallfun(self,tau,qprtls):        
        """
        Function for evaluatinng the target distribution for sequential monte carlo
        𝜋(𝑥) = 𝜙(𝜏𝐶1(𝑥)) * 𝜙(𝜏𝐶2(𝑥)) * .... * 𝜙(𝜏𝐶k(𝑥)) 

        Return:
            pit: returns
        """
        pitall=[]
        for i in range(0,qprtls.shape[1]):
            pitall.append(self.pitfun(tau,qprtls[:,i]))
        pitall=np.asarray(pitall)

        return pitall 

    def pitfun(self,tau,qval):        
        """
        Function for evaluatinng the target distribution for sequential monte carlo
        𝜋(𝑥) = 𝜙(𝜏𝐶1(𝑥)) * 𝜙(𝜏𝐶2(𝑥)) * .... * 𝜙(𝜏𝐶k(𝑥)) 

        Return:
            pit: returns
        """
        Call= self.model.apply_eval(list(qval))       
        Call=Call+list(qval-self.qlims[:,0])
        Call=Call+list(-qval+self.qlims[:,1])


        pit=tau*np.asarray(Call)
        pit=norm.cdf(pit)
        pit=np.prod(pit)

        return pit

    def resampling(self,qprtls,W):    
        """
        Resampling using systematic resampling 

        Return:
            pit: returns
        """
        qprtlsnew=copy.deepcopy(qprtls)

        j=0
        isavejend=0
        sumW=W[j]
        u=np.random.rand()/self.nsamples
        for i in range(0,self.nsamples):
            while sumW<u:
                j=j+1
                isavejend=copy.deepcopy(i)
                sumW=sumW+W[j]
            qprtlsnew[:,i]=qprtls[:,j]
            u=u+1.0/self.nsamples
        W=np.ones(self.nsamples)/self.nsamples
        return qprtlsnew,W

    def calSMCvar(self,qprtls,W,Varold,VarRold):

        qmean=np.reshape(np.mean(qprtls,axis=1),(-1,1))
        Var=(qprtls-qmean)   
        Var=np.dot(Var,Var.T)/qprtls.shape[1]
        try:
            VarR = np.linalg.cholesky(Var)
        except:
            VarR=copy.deepcopy(VarRold)
            Var=copy.deepcopy(Varold)

        return Var,VarR

    def SMCRW(self,tau,qval,Var,VarR):        
        """
        Function for sampling using random walk Metropolis algorithm for SMC

        Return:
            Q_MCMC: Accepted samples
        """
        qval=np.reshape(np.asarray(qval),(-1,1))
        qmean=copy.deepcopy(qval)
        qmean_old=copy.deepcopy(qval)
        Var=copy.deepcopy(Var)
        Varold=copy.deepcopy(Var)
        
        iaccept=0
        output0=self.pitfun(tau,qval[:,0])+1e-16
        accepted_qs=[qval[:,0]]
        for isamples in range(0,self.NSMC_MCMC):

            z = np.random.randn(qval.shape[0],1)
            qnew = qval + np.dot(VarR.T,z)

            output=self.pitfun(tau,qnew[:,0])
            ratio=output/output0

            if ratio>np.random.rand():
                qval=copy.deepcopy(qnew)
                output0=copy.deepcopy(output0)
                iaccept=iaccept+1
                accepted_qs.append(list(qval[:,0]))

        return qval[:,0],iaccept,accepted_qs

    def GibbsSampler(self,tau,qval,Var,VarR):        
        """
        Function for sampling using gibs sampler for SMC

        Return:
            qval: Accepted samples
        """        
        qval=np.reshape(np.asarray(qval),(-1,1))
        
        iaccept=0
        output0=self.pitfun(tau,qval[:,0])+1e-16
        accepted_qs=[qval[:,0]]

        qindG=np.random.randint(qval.shape[0],size=self.NSMC_MCMC)
        for ind in qindG:
            qnew=copy.deepcopy(qval)
            qnew[ind,0]=qval[ind,0]+np.sqrt(Var[ind,ind])*np.random.randn()

            output=self.pitfun(tau,qnew[:,0])
            ratio=output/output0

            if ratio>np.random.rand():
                qval=copy.deepcopy(qnew)
                output0=copy.deepcopy(output0)
                iaccept=iaccept+1
                accepted_qs.append(list(qval[:,0]))

        return qval[:,0],iaccept,accepted_qs
