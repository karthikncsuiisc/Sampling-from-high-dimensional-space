import matplotlib.pyplot as plt
import numpy as np
import sys

def show_shape(patches):
    ax=plt.gca()
    for patch in patches:
        ax.add_patch(patch)
    plt.axis('scaled')
    plt.show()

	
if __name__== '__main__':
    # plt.Circle((0,0), radius= 5)

    circles=np.loadtxt(sys.argv[1])

    output_file=sys.argv[2]

    fout=open(output_file,"w")
    fout.write("2\n")
    fout.write(str(circles[0,0])+" "+str(circles[0,1])+"\n")


    fig=plt.figure()
    ax=plt.gca()

    for circle in circles:
        x0c,x1c,r0=circle

        c= plt.Circle((x0c,x1c), radius= r0)
        ax.add_patch(c)

        circstring="-x[0] * x[0] + "+str(2.0*x0c)+" * x[0] - "
        circstring=circstring+"x[1] * x[1] + "+str(2.0*x1c)+" * x[1] - "
        circstring=circstring+str(x0c**2+x1c**2-r0**2)+" >= 0\n"

        fout.write(circstring)

    plt.axis('scaled')
    # plt.show()
    fig.savefig(output_file[:-4]+".png")

    fout.close()

#------------------------------------------------------------------------
    fout=open(output_file[:-4]+"_out.txt","w")
    fout.write("2\n")
    fout.write(str(circles[0,0])+" "+str(circles[0,1])+"\n")


    fig=plt.figure()
    ax=plt.gca()

    for circle in circles:
        x0c,x1c,r0=circle

        c= plt.Circle((x0c,x1c), radius= r0)
        ax.add_patch(c)

        circstring="x[0] * x[0] - "+str(2.0*x0c)+" * x[0] + "
        circstring=circstring+"x[1] * x[1] - "+str(2.0*x1c)+" * x[1] + "
        circstring=circstring+str(x0c**2+x1c**2-r0**2)+" >= 0\n"

        fout.write(circstring)

    plt.axis('scaled')
    # plt.show()
    fig.savefig(output_file[:-4]+"_out.png")

    fout.close()
