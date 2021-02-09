import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

class SampAlg:
    """Class for different sampling algorithms"""

    def __init__(self,model,qstart,qlims,nsamples,Vstart=None,method="MCMCRW",output_file="samples.txt"):
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
            self.Vstart=np.eye(len(qstart))
        else:
            self.Vstart=Vstart
        self.method=method
        self.nsamples=nsamples

        os.chdir("..")
        outputpath= 'output'
        if os.path.isdir(outputpath)==False:
            print(outputpath ,"directory doesnot exist, creating one to save the output")
            os.mkdir(outputpath)
        os.chdir("code")
        self.outputpath="../"+outputpath+"/"
        self.output_file=self.outputpath+output_file
        self.saveevery=2
        fout=open(self.output_file,"w")
        fout.close()

        return
    def sample(self):
        """
        Function to call the sampling method of choice and plot results
        """
        
        if self.method=="MCMCRW":
            ntot,accpt_ratio=self.MCMCRW()
            print("Total generated samples:",ntot)
            print('Acceptance ratio:',accpt_ratio)
        else:
            print("Wrong choice of method for sampler algorithm")
        Q_MCMC=np.loadtxt(self.output_file)
        self.plotsamples(Q_MCMC)

        print("Saving the results to ",self.outputpath)

    def MCMCRW(self):        
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
            qinlimit=(q_new[:,0]>=self.qlims[:,0]).sum()+(q_new[:,0]<=self.qlims[:,1]).sum()
            qinlimit=(qinlimit==2*q_new.shape[0])

            output=self.model.apply(list(q_new[:,0]))

            if output and qinlimit:
                q_old=q_new
                Q_MCMC.append(q_old[:,0])
                iaccept=iaccept+1

                ntot=iaccept+ireject
                accpt_ratio=iaccept/ntot
                pbar.update(1)
            else:
                ireject=ireject+1
            
            if np.mod(iaccept,self.saveevery)==0 or iaccept==self.nsamples-1:
                fout=open(self.output_file,"a")
                np.savetxt(fout,np.asarray(Q_MCMC)[:-1,:])
                fout.close()
                Q_MCMC=[q_old[:,0]];
        pbar.close()

        return ntot,accpt_ratio

    def plotsamples(self,Q_MCMC):
        """
        Function for plotting the accepted samples
        """
        fig=plt.figure()
        plt.plot(Q_MCMC[:,0],Q_MCMC[:,1],'o',markersize=4)
        fig.savefig(self.output_file[:-4]+".png")


        






