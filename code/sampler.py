import numpy as np
import sys
from constraints import *
from metropolis import Metropolis
from SMC import SMC
import time

if __name__ == '__main__':

    start_time=time.time()

    input_file=sys.argv[1]
    output_file=sys.argv[2]
    n_results=sys.argv[3]
    try:
        method=sys.argv[4]
    except:
        method="SMC"

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
    elif method=="Metropolis" or method=="AdaptiveMetropolis":
        sampling=Metropolis(model=constrains,
                        qstart=constrains.get_example(),
                        qlims=qlims,
                        nsamples=int(n_results),
                        output_file=output_file,
                        method=method,
                        plotfigures=True,
                        saveoutputfile=True)
    else:
        print("Wrong choice of method for sampler algorithm")
        sys.exit(1)

    sampling.sample()
    comp_time=time.time()-start_time
    print("Computational time:",comp_time)
