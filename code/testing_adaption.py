import numpy as np
import sys
from constraints import *
from sampleralgorithm import *

if __name__ == '__main__':

    input_file="../code/formulation.txt"
    output_file="../results/dum_out.txt"
    n_results=10000
    method="AdaptiveMetropolis"

    print("Input file:",input_file)
    print("Output file:",output_file)
    print("n_results",n_results)
    print("Sampling method:",method)

    constrains=Constraint(input_file)
    qstart=constrains.get_example()
    qlims=np.zeros((len(qstart),2))
    qlims[:,1]=1.0

    accpt_all=[]
    comp_all=[]
    ntrials=10
    adapt_range=range(5,200,10)
    flag=1

    for adpt in adapt_range:
        accpt_dum=0
        comp_dum=0
        for iter in range(0,ntrials):   

            print(flag," of ",len(adapt_range)*ntrials)
            flag=flag+1

            sampling=SampAlg(model=constrains,
                            qstart=constrains.get_example(),
                            qlims=qlims,
                            nsamples=int(n_results),
                            output_file=output_file,
                            method=method,
                            adapt_interval=adpt,
                            printoutput=False,
                            plotfigures=False,
                            saveoutputfile=False)

            comp_time,accpt_ratio=sampling.sample()
            accpt_dum=accpt_dum+accpt_ratio
            comp_dum=comp_dum+comp_time
        accpt_all.append(accpt_dum/ntrials)
        comp_all.append(comp_dum/ntrials)
    
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,4))
      
    ax1.plot(adapt_range,accpt_all)
    ax1.set_title("Acceptance ratio")
    ax1.set_xlabel("Adaption rate")
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    ax2.plot(adapt_range,comp_all)
    ax2.set_title("Computational time")
    ax2.set_xlabel("Adaption rate")
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    fig.savefig("../results/adaption_test.png")
    plt.close()


