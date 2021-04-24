import numpy as np
import sys
from constraints import *
from metropolis import Metropolis
from SMC import SMC
import time
from mpi4py import MPI

if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()
    size = comm.size

    start_time=time.time()

    input_file=sys.argv[1]
    output_file=sys.argv[2]
    n_results=sys.argv[3]
    try:
        method=sys.argv[4]
    except:
        method="SMC"
    
    if myrank==0:
        print("Input file:",input_file)
        print("Output file:",output_file)
        print("n_results",n_results)

    constrains=Constraint(input_file)
    qstart=constrains.get_example()
    qlims=np.zeros((len(qstart),2))
    qlims[:,1]=1.0

    if method=="SMC":
        sampling=SMC(model=constrains,
                        qstart=constrains.get_example(),
                        qlims=qlims,
                        nsamples=int(n_results),
                        output_file=output_file,
                        plotfigures=True,
                        saveoutputfile=True)
        sampling.sample()

    elif method=="Metropolis" or method=="AdaptiveMetropolis" or method=="Gibbs":
        if myrank==0:
            sampling=Metropolis(model=constrains,
                            qstart=constrains.get_example(),
                            qlims=qlims,
                            nsamples=int(n_results),
                            output_file=output_file,
                            method=method,
                            plotfigures=True,
                            saveoutputfile=True)
            sampling.sample()

    else:
        if myrank==0:
            print("Wrong choice of method for sampler algorithm")
        sys.exit(1)

    comp_time=time.time()-start_time
    if myrank==0:
        print("Computational time:",comp_time)   
        comp_array=np.array([[size,comp_time]])
        np.savetxt("../results/comp_time.log",comp_array)
