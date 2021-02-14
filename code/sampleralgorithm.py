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

class SampAlg:
    """Class for different sampling algorithms"""

    def __init__(self,model,qstart,qlims,nsamples,Vstart=None,output_file="samples.txt",
                     method="RandomWalk",adapt_interval=100,printoutput=True,plotfigures=True,
                     saveoutputfile=True):
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
        if Vstart==None:
            self.Vstart=np.eye(len(qstart))*1.0/len(qstart)
        else:
            self.Vstart=Vstart
        self.method=method
        self.nsamples=nsamples
        self.printoutput=printoutput
        self.plotfigures=plotfigures
        self.saveoutputfile=saveoutputfile

        os.chdir("..")
        outputpath= 'results'
        if os.path.isdir(outputpath)==False:
            print(outputpath ,"directory doesnot exist, creating one to save the output")
            os.mkdir(outputpath)
        os.chdir("code")
        self.outputpath="../"+outputpath+"/"
        self.output_file=self.outputpath+output_file
        self.saveevery=nsamples
        self.adapt_interval=adapt_interval
        self.ESSfract=0.5
        self.tauThres=1e6
        self.NSMC_MCMC=5
        self.sp=2.38**2/self.qstart.shape[0]
        
        return
    def sample(self):
        """
        Function to call the sampling method of choice and plot results
        """
        
        start_time=time.time()
        if self.method=="RandomWalk":
            ntot,accpt_ratio=self.RandomWalk()
        elif self.method=="AdaptiveMetropolis":
            ntot,accpt_ratio=self.AdaptiveMetropolis()
        elif self.method=="SMC":
            qprtls,ntot,neff,accpt_ratio=self.SMC()
        else:
            print("Wrong choice of method for sampler algorithm")
            sys.exit(1)
        
        comp_time=time.time()-start_time

        if self.printoutput:
            print("Total generated samples:",ntot)
            print("Total effective samples:",int(ntot*accpt_ratio))
            print('Acceptance ratio:',accpt_ratio)
            print("Computational time:",comp_time)
            print("Saving the results to ",self.outputpath)
        
        if self.saveoutputfile:
            np.savetxt(self.output_file,qprtls.T)

        if self.plotfigures:
            fig=sns.pairplot(pd.DataFrame(qprtls.T), markers='o')
            fig.savefig(self.output_file[:-4]+"_"+self.method+".png")
            plt.close()
        return
    
    def objF(self,qval):

        output=np.all(qval>=self.qlims[:,0]) and np.all(qval<=self.qlims[:,1]) and self.model.apply(list(qval))
        return output

    def RandomWalk(self):        
        """
        Function for sampling using random walk Metropolis algorithm

        Return:
            Q_MCMC: Accepted samples
        """

        R = np.linalg.cholesky(self.Vstart);
        q_old =self.qstart;
        Q_MCMC=[q_old[:,0]];
        iaccept=0
        ireject=0

        pbar = tqdm(total = self.nsamples)
        while iaccept<self.nsamples:

            z = np.random.randn(q_old.shape[0],1);
            q_new = q_old + np.dot(R.T,z);
            output=self.objF(q_new[:,0])

            if output:
                q_old=copy.deepcopy(q_new)
                Q_MCMC.append(q_old[:,0])
                iaccept=iaccept+1

                ntot=iaccept+ireject
                accpt_ratio=iaccept/ntot
                pbar.update(1)
            else:
                ireject=ireject+1

            if np.mod(iaccept+1,self.saveevery)==0 or iaccept==self.nsamples-1:
                fout=open(self.output_file,"a")
                np.savetxt(fout,np.asarray(Q_MCMC)[:-1,:])
                fout.close()
                Q_MCMC=[q_old[:,0]];
        pbar.close()

        return ntot,accpt_ratio

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
        Q_MCMC=[q_old[:,0]];
        iaccept=0
        ireject=0

        #Adaptive metropoils parameters
        qmean=copy.deepcopy(q_old)
        qmean_old=copy.deepcopy(q_old)
        sp=2.38**2/q_old.shape[0]
        adptintnew=1

        pbar = tqdm(total = self.nsamples)
        
        while iaccept<self.nsamples:

            z = np.random.randn(q_old.shape[0],1);
            q_new = q_old + np.dot(R.T,z);
            output=self.objF(q_new[:,0])
            # print(q_new.T,q_old.T,output)
            # sys.exit()

            if output:
                k=copy.deepcopy(iaccept)+1               

                #update variance and qmean
                Vupdate=k*np.dot(qmean_old,qmean_old.T)
                Vupdate=Vupdate-(k+1)*np.dot(qmean,qmean.T)
                Vupdate=Vupdate+np.dot(q_new,q_new.T)
                Vupdate=Vupdate+np.eye(len(q_new))*1e-16
                Vupdate=sp/k*Vupdate

                Vold=copy.deepcopy(self.Vstart)
                self.Vstart=(k-1.0)/k*self.Vstart+Vupdate
                qmean_old=copy.deepcopy(qmean)
                qmean=k*qmean/(k+1)+q_old/(k+1)

                q_old=copy.deepcopy(q_new)
                Q_MCMC.append(q_old[:,0])                
                iaccept=iaccept+1

                pbar.update(1)

            else:
                ireject=ireject+1
            
            if self.saveoutputfile:            
                if np.mod(iaccept+1,self.saveevery)==0 or iaccept==self.nsamples-1:
                    fout=open(self.output_file,"a")
                    np.savetxt(fout,np.asarray(Q_MCMC)[:-1,:])
                    fout.close()
                    Q_MCMC=[q_old[:,0]]
            
            adptintnew=norm.cdf(iaccept/self.adapt_interval,0.5,0.2)*self.adapt_interval
            adptintnew=int(max(1.0,adptintnew))

            if np.mod(iaccept,adptintnew)==0:

                # if output:
                #     print("Adapting new variance at iteration #:",iaccept)
                   
                try:
                    R = np.linalg.cholesky(self.Vstart)
                except:
                    self.Vstart=copy.deepcopy(Vold)
                    # sys.exit(0)

        pbar.close()

        ntot=iaccept+ireject
        accpt_ratio=iaccept/ntot

        return ntot,accpt_ratio


    def plotsamples(self,Q_MCMC):
        """
        Function for plotting the accepted samples
        """

        fig=sns.pairplot(pd.DataFrame(Q_MCMC), markers='o')
        fig.savefig(self.output_file[:-4]+".png")

    def SMC(self):        
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

        print("-------------------------------------")
        while tau < self.tauThres:
            tcount=tcount+1
            tau,wnt,ESSeff=self.find_tau(qprtls,tauL=tau,tauU=self.tauThres)
            W=W*wnt
            W=W/np.sum(W)

            qprtls,W=self.resampling(qprtls,W)
            Var,VarR=self.calSMCvar(qprtls,W)
            EFF_prtls=np.unique(qprtls,axis=1).shape[1]

            iacceptall=0
            if np.abs(tau-self.tauThres)/self.tauThres<1e-3:
                for i in range(0,self.nsamples):
                    # qprtls[:,i],iaccept,allaccepts_dum=self.SMCRW(tau,qprtls[:,i],Var,VarR)
                    qprtls[:,i],iaccept,allaccepts_dum=self.GibbsSampler(tau,qprtls[:,i],Var,VarR)
                    iacceptall=iacceptall+iaccept
                    allaccepts_qs=allaccepts_qs+allaccepts_dum
                allaccepts_qs=np.asarray(allaccepts_qs)
                allaccepts_qs=np.unique(allaccepts_qs,axis=1).T
                EFF_prtls=allaccepts_qs.shape[1]
                print("tau:",tau,"Accept. Ratio:",iacceptall/(self.nsamples*self.NSMC_MCMC),"ESS:",ESSeff,"Eff. particles:",EFF_prtls)
            else:
                for i in range(0,self.nsamples):
                    qprtls[:,i],iaccept,_=self.GibbsSampler(tau,qprtls[:,i],Var,VarR)
                    iacceptall=iacceptall+iaccept
                print("tau:",tau,"Accept. Ratio:",iacceptall/(self.nsamples*self.NSMC_MCMC),"ESS:",ESSeff,"Eff. particles:",EFF_prtls)

        print("-------------------------------------")
           

        ntot=(self.nsamples*self.NSMC_MCMC)*tcount
        neff=allaccepts_qs.shape[1]
        accpt_ratio=neff/ntot

        # for i in range(0,allaccepts_qs.shape[1]):
        #     print(i,self.model.apply(list(allaccepts_qs[:,i])))

        return allaccepts_qs,ntot,neff,accpt_ratio

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

        # neff_RS=np.unique(qprtlsnew,axis=1).shape[1]
        # neff_BR=np.unique(qprtls,axis=1).shape[1]               
        # print("---",neff_RS," effective samples generated after resampling from ",neff_BR,"---")   
        
        # fig,axs=plt.subplots(1,2,figsize=(6,4))
        # s1=axs[0].scatter(qprtls[0,:],qprtls[1,:],c=W)
        # axs[1].scatter(qprtlsnew[0,:],qprtlsnew[1,:],c=W)
        # fig.colorbar(s1)
        # plt.show()
        # plt.close()

        W=np.ones(self.nsamples)/self.nsamples

        return qprtlsnew,W
    
    def calSMCvar(self,qprtls,W):

        qmean=np.reshape(np.mean(qprtls,axis=1),(-1,1))
        Var=(qprtls-qmean)   
        Var=np.dot(Var,Var.T)/qprtls.shape[1]
        VarR = np.linalg.cholesky(Var)

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
