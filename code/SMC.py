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
from mpi4py import MPI

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
        self.NSMC_MCMC=3
        # self.nsamples=self.nsamples

        self.comm = MPI.COMM_WORLD
        self.myrank = self.comm.Get_rank()
        self.size=self.comm.size

    def sample(self):
        """
        Function to call the sampling method of choice and plot results
        """
        if self.myrank==0:
            print("Sampling using sequential monte carlo")
        
        start_time=time.time()
        qsamples,ntot,naccept=self.SMCsampler()
        comp_time=time.time()-start_time

        if self.myrank==0:
            super(SMC, self).printoutput(qsamples,ntot,naccept,naccept/ntot,self.size,comp_time)
            # super(SMC, self).plotsamples(qsamples,self.method)
            super(SMC, self).savesamples(qsamples,self.method)

        return naccept,naccept/ntot

    def SMCsampler(self):        
        """
        Function for sampling using sequential monter carlo

        Return:
            Q_MCMC: final samples
        """
        tau=0;
        tcount=0
        allaccepts_qs=[]

        if self.myrank==0:
            qprtls=np.random.rand(self.qlims.shape[0],self.nsamples)
            qprtls=np.reshape(self.qlims[:,0],(-1,1))+(np.reshape(self.qlims[:,1],(-1,1))-np.reshape(self.qlims[:,0],(-1,1)))*qprtls[:,:]

            Var=copy.deepcopy(self.Vstart)
            VarR=copy.deepcopy(Var)
            W=np.ones(self.nsamples)*1/self.nsamples

            pbar = tqdm(total = 1000)
            count=0
            countold=0
        else:
            qprtls=None

        start_time=time.time()

        while tau < self.tauThres:

            tcount=tcount+1         
            partitions=self.distributor(qprtls)          
            qprtls_div=self.comm.scatter(partitions,root=0)

            tau,wnt,ESSeff=self.find_tau(qprtls_div,tauL=tau,tauU=self.tauThres)
            wnt_all=self.comm.gather(wnt,root=0)

            if self.myrank==0:
                wnt_all=np.hstack(wnt_all)
                W=W*wnt_all;    W=W/np.sum(W)
                qprtls,W=self.resampling(qprtls,W)
                Var,VarR=self.calSMCvar(qprtls,W,Var,VarR)
            else:
                Var=None; VarR=None

            partitions=self.distributor(qprtls)          
            qprtls_div=self.comm.scatter(partitions,root=0)
            Var=self.comm.bcast(Var,root=0)
            VarR=self.comm.bcast(VarR,root=0)

            iacceptall=0
            if tau>=self.tauThres:
                for i in range(0,qprtls_div.shape[1]):
                    # qprtls[:,i],iaccept,allaccepts_dum=self.SMCRW(tau,qprtls[:,i],Var,VarR)
                    qprtls_div[:,i],iaccept,allaccepts_dum=self.GibbsSampler(tau,qprtls_div[:,i],Var,VarR)
                    iacceptall=iacceptall+iaccept
                    allaccepts_qs=allaccepts_qs+allaccepts_dum
                allaccepts_qs=np.asarray(allaccepts_qs)
                allaccepts_qs=self.comm.gather(allaccepts_qs,root=0)

            else:
                for i in range(0,qprtls_div.shape[1]):
                    # qprtls[:,i],iaccept,allaccepts_dum=self.SMCRW(tau,qprtls[:,i],Var,VarR)
                    qprtls_div[:,i],iaccept,_=self.GibbsSampler(tau,qprtls_div[:,i],Var,VarR)
                    iacceptall=iacceptall+iaccept               

            qprtls=self.comm.gather(qprtls_div,root=0)
            if self.myrank==0:
                qprtls=np.hstack(qprtls)
                count=np.abs(int(np.log10(tau)*1000/6))
                pbar.update(count-countold)
                countold=copy.deepcopy(count)
            
            if tcount>200:
                print("Number of iterations are more than ",tcount)

                for i in range(0,qprtls_div.shape[1]):
                    # qprtls[:,i],iaccept,allaccepts_dum=self.SMCRW(tau,qprtls[:,i],Var,VarR)
                    qprtls_div[:,i],iaccept,allaccepts_dum=self.GibbsSampler(tau,qprtls_div[:,i],Var,VarR)
                    iacceptall=iacceptall+iaccept
                    allaccepts_qs=allaccepts_qs+allaccepts_dum
                allaccepts_qs=np.asarray(allaccepts_qs)
                allaccepts_qs=self.comm.gather(allaccepts_qs,root=0)

                break
        
        if self.myrank==0:
            allaccepts_qs=np.vstack(allaccepts_qs)
            allaccepts_qs=np.unique(allaccepts_qs,axis=0)
            EFF_prtls=np.unique(allaccepts_qs,axis=0).shape[0]
            print("Accepted samples:",allaccepts_qs.shape)
        else:
            EFF_prtls=1


        ntot=(self.nsamples*self.NSMC_MCMC)*tcount
        return allaccepts_qs,ntot,EFF_prtls
    
    def distributor(self,qprtls):

        if self.myrank==0:
            partitions=[]
            for prtls in np.array_split(np.arange(0,self.nsamples),self.size,axis=0):
                partitions.append(qprtls[:,prtls])
        else:
            partitions=[]
        
        return partitions

    def find_tau(self,qprtls,tauL,tauU,deltatau=100.0,Niter=20):        
        """
        Function to find tau using ESS

        Return:
            tau: returns
        """

# #------------------------manual tau
#         pitallminus=self.pitallfun(tauL,qprtls)
#         taunew=10**(np.log10(tauL+1)+0.1/2)
#         wnt=self.ESSfun(taunew,qprtls,pitallminus)
#         wnt_all=self.comm.gather(wnt,root=0)
#         if self.myrank==0:
#             wnt_all=np.hstack(wnt_all)
#             ESS=(np.sum(wnt_all))**2/(np.sum(wnt_all**2)+1e-16)
#         else:
#             ESS=None
#         ESS=self.comm.bcast(ESS,root=0)
#         return taunew,wnt,ESS/self.nsamples

#-------------------------------------------

        pitallminus=self.pitallfun(tauL,qprtls)
        funL=1.0-self.ESSfract
        wnt=self.ESSfun(tauU,qprtls,pitallminus)

        wnt_all=self.comm.gather(wnt,root=0)
        if self.myrank==0:
            wnt_all=np.hstack(wnt_all)
            ESS=(np.sum(wnt_all))**2/(np.sum(wnt_all**2)+1e-16)
        else:
            ESS=None
        ESS=self.comm.bcast(ESS,root=0)

        funU=ESS/self.nsamples-self.ESSfract
        fact=1.0

        if funU>=0.0:
            return tauU,wnt,ESS/self.nsamples

        for iter in np.arange(0,Niter):

            taunew=tauL+fact*deltatau
            wnt=self.ESSfun(taunew,qprtls,pitallminus)
            wnt_all=self.comm.gather(wnt,root=0)
            if self.myrank==0:
                wnt_all=np.hstack(wnt_all)
                ESS=(np.sum(wnt_all))**2/(np.sum(wnt_all**2)+1e-16)
            else:
                ESS=None
            ESS=self.comm.bcast(ESS,root=0)

            funnew=ESS/self.nsamples-self.ESSfract
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

        return taunew,wnt,ESS/self.nsamples
    
    def ESSfun(self,tau,qprtls,pitallminus):
        """
        Calcualtes effective samples

        Return:
            ESS: returns
        """
        pitall=self.pitallfun(tau,qprtls)
        wnt=pitall/(pitallminus+1e-16)
        # ESS=(np.sum(wnt))**2/(np.sum(wnt**2)+1e-16)
        # return ESS,wnt
        return wnt

    def pitallfun(self,tau,qprtls):        
        """
        Function for evaluatinng the target distribution for sequential monte carlo
        𝜋(𝑥) = 𝜙(𝜏𝐶1(𝑥)) * 𝜙(𝜏𝐶2(𝑥)) * .... * 𝜙(𝜏𝐶k(𝑥)) 
        for all particles

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
        for single particle

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
            # print(ind)
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
