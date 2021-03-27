# Sampling from High Dimensional Space

Python program for sampling from high dimensional space with complex, non-linear constraints. The program provides a choice of selecting one from the following four methods

1) Sequential Monte Carlo (SMC)
2) Gibbs Sampler (Gibbs)
3) Adaptive Metropolis (AdaptiveMetropolis)
4) Metropolis Random Walk (Metropolis)

The choice of the method can be provided as the input while running the program and choice name to be used at the command line are provided in the brackets. Please refer to the [report](https://github.com/karthikncsu/Sampling-from-high-dimensional-space/blob/main/report.md) for more information.

## To use the Code:

### Required python packages to run the code

* numpy
* matplotlib
* tqdm
* scipy
* seaborn
* mpi4py

### Input file format

```
#Dimension of the problem
2
#Initial starting point
0.0 0.0
# Constraints start from herr
1.0 - x[0] - x[1] >= 0.0
```

### To use the package

1. Paste the input file inside the code folder. Sample input files are in the code directory.
2. Open the terminal and go to the code colder
3. To sample from the space with the nonlinear constraints, run below arguments

      ``` cd code ```
      ``` mpirun -np <number of processors> python sampler.py <input file> <output file> <nsamples> <method: optional> ```
  
      Eg: ```mpirun -np 4 python sampler.py mixture.txt mixture_out.txt 1000 SMC```

4. The optional method is provided as input at the end of the arguments. If the method is not mentioned, sequential Monte Carlo (SMC) will be selected as the default choice.
5. The results and plots will be generated in the results folder

### Demo

The program is integrated into the Django web framework to demonstrate the results and hosted on the Heroku server. Below is the link for the website to try out different options for sampling.

[Django Website](https://sheltered-eyrie-03969.herokuapp.com/)

Due to the limitations with the Heroku free server, the web site returns a time-out error for cases with a computational time of more than 30 seconds.  The issue will be resolved in the future and the website can be used to solve any case. In the mean while, please try other cases.

Below is the link for the Git repository of the Django website.

[Github code for Django Website](https://github.com/karthikncsu/Django-website-for-sampling-high-dimensional-space)

