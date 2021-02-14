import numpy as np
import sys
from constraints import *
from sampleralgorithm import *

if __name__ == '__main__':

    input_file=sys.argv[1]
    output_file=sys.argv[2]
    n_results=sys.argv[3]
    try:
        method=sys.argv[4]
    except:
        method="AdaptiveMetropolis"

    print("Input file:",input_file)
    print("Output file:",output_file)
    print("n_results",n_results)
    print("Sampling method:",method)

    constrains=Constraint(input_file)
    qstart=constrains.get_example()
    qlims=np.zeros((len(qstart),2))
    qlims[:,1]=1.0

# # example paper
#     qlims[0,0]=0.6
#     qlims[0,1]=2.0
#     qlims[1,0]=-0.5
#     qlims[1,1]=0.5


    sampling=SampAlg(model=constrains,
                     qstart=constrains.get_example(),
                    #  qstart=[0.3,0.3],
                     qlims=qlims,
                     nsamples=int(n_results),
                     output_file=output_file,
                     method=method,
                     plotfigures=True,
                     saveoutputfile=True)
    sampling.sample()


