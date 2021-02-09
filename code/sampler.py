import numpy as np
import sys
from constraints import *
from sampleralgorithm import *

if __name__ == '__main__':

    input_file=sys.argv[1]
    output_file=sys.argv[2]
    n_results=sys.argv[3]

    constrains=Constraint(input_file)
    qstart=constrains.get_example()
    qlims=np.zeros((len(qstart),2))
    qlims[:,1]=1.0

    sampling=SampAlg(constrains,qstart=constrains.get_example(),qlims=qlims,nsamples=int(n_results),output_file=output_file)
    sampling.sample()


