import numpy as np
import sys
from constraints import *
from metropolis import Metropolis
from SMC import SMC
import time

if __name__ == '__main__':

    testing_files=["mixture.txt","example.txt","formulation.txt","alloy.txt"]
    allsave=["Problem","Method","Dimension","Constraints","Eff.Samples","Accept.ratio","Comp.Time"]

    fout=open("../results/testing_results.txt","w")
    fout.write("%s %s %s %s %s %s %s\n"%(allsave[0].center(20,' '),allsave[1].center(20,' '),allsave[2].center(20,' '),
        allsave[3].center(20,' '),allsave[4].center(20,' '),allsave[5].center(20,' '),allsave[6].center(20,' ')))
    fout.write("----------------------------------------------------------------------------------------------------------------------------------------------\n")

    for filename in testing_files:
        for method in ["SMC","AdaptiveMetropolis","Metropolis","Gibbs"]:

            if filename == "formulation.txt" and method == "Metropolis":
                continue
            if filename == "alloy.txt" and method == "AdaptiveMetropolis":
                continue
            if filename == "alloy.txt" and method == "Metropolis":
                continue

            start_time=time.time()

            input_file=filename
            output_file=filename
            n_results=5000

            constrains=Constraint(input_file)
            qstart=constrains.get_example()
            qlims=np.zeros((len(qstart),2))
            qlims[:,1]=1.0
            
            if method=="SMC":
                sampling=SMC(model=constrains,
                                qstart=constrains.get_example(),
                                qlims=qlims,
                                nsamples=int(n_results),
                                output_file=output_file[:-4]+"_"+method+".txt",
                                plotfigures=True,
                                saveoutputfile=True)
            elif method=="Metropolis" or method=="AdaptiveMetropolis"or method=="Gibbs":
                sampling=Metropolis(model=constrains,
                                qstart=constrains.get_example(),
                                qlims=qlims,
                                nsamples=int(n_results),
                                output_file=output_file[:-4]+"_"+method+".txt",
                                method=method,
                                plotfigures=True,
                                saveoutputfile=True)
            else:
                print("wrong method used:",method)
                sys.exit()
            naccept,acceptratio=sampling.sample()
            comp_time=time.time()-start_time

            nconstrnts=str(len(constrains.exprs))

            allsave=[filename[:-4],method,str(constrains.get_ndim()),nconstrnts,str(int(naccept)),str(int(100*acceptratio)/100),str(int(comp_time*100)/100)]
            fout.write("%s %s %s %s %s %s %s\n"%(allsave[0].center(20,' '),allsave[1].center(20,' '),allsave[2].center(20,' '),
                allsave[3].center(20,' '),allsave[4].center(20,' '),allsave[5].center(20,' '),allsave[6].center(20,' ')))
            fout.flush()

    fout.close()
