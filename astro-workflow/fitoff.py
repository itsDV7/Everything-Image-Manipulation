#! /usr/bin/env python3
"""
Method to refine the offset fit by using the scipy.optimize.curve_fit function.
"""

import os
import argparse
import re
import time
import fitsio
import numpy as np
from scipy.optimize import curve_fit



################################################
def getoff(x,*p):
    """
    Get the offset between a pair of images.
    The difference p[xjfl[i]] - p[xifl[i]] represents the offset between a pair of images. The calculation is performed for each pair as determined by the indices in xjfl and xifl
    """
    y=np.array([p[xjfl[i]]-p[xifl[i]] for i in range(x.size)])
    return y


################################################
if __name__ == "__main__":
    # usage `python3 fitoff.py -i "out/test.g.offset_b8" -o "out/test.g.zoff_b8" -b -v 2`
    parser = argparse.ArgumentParser(description='Code to take offset measurement and find optimal set of per image offsets for the ensemble') 

    parser.add_argument('-i','--input',   action='store', type=str, default=None, required=True,  help='Input offset (pair) measurements from findoff_WCS.')
    parser.add_argument('-o','--output',  action='store', type=str, default=None, required=True,  help='Output file of optimal (per image) offsets (a zcom file).')
    parser.add_argument('-e','--exclude', action='store', type=str, default=None, required=False, help='Exclude file (list of images numbers to exclude)')
    parser.add_argument('-d','--diag',    action='store', type=str, default=None, required=False, help='Diagnostic file (optional output).')
    parser.add_argument('-w','--weight',  action='store', type=int, default=0,    required=False, help='Weighting mode (0=equal weightdefault), 1=overlap (1/sqrt(#pix)), 2=sigma (1/RMS(offset)))')
    parser.add_argument('-v','--verbose', action='store', type=int, default=0,    required=False, help='Verbosity (default=0, max=2)')
    parser.add_argument('-b','--boot',    action='store_true', default=False,     required=False, help='Use bootstrap to make an intial guess)')

    args = parser.parse_args()
    if (args.verbose > 0):
        print("Args: {:}".format(args))

    f=open(args.input,'r')

    #  Initializations
    # diff matrix stores the median offset value between pairs of images. For each pair of images identified by their indices (ifile and jfile), it records the median offset (cols[0]) in the corresponding cell.
    # The nover matrix keeps track of the number of overlapping pixels between each pair of images.
    # sigma matrix stores the standard deviation of the offsets between each pair of images.
    # The ldiff matrix is a boolean matrix indicating whether there is a recorded difference (i.e., an offset) between each pair of images.

    flist=True
    fdict={}
    y=[]
    x=[]
    xifl=[]
    xjfl=[]
    c=[]
    s=[]
    for line in f:
        cols=line.split()
        if (not(flist)):
            # end of filelist. Get offset values from findoff output
            y.append(float(cols[0])) # get median offset / medoff
            x.append(float(cols[1])) # get count
            ifile=int(cols[2]) # get iImage inum
            jfile=int(cols[3]) # get jImage jnum
            xifl.append(ifile-1) # subtract 1 because count started from 1
            xjfl.append(jfile-1) # subtract 1 because count started from 1
            c.append(int(cols[4])) # get count/ npix
            s.append(float(cols[5])) # get medsig/ std deviation
            
            diff[ifile-1,jfile-1]=float(cols[0])  # The offset for the pair (ifile, jfile) is stored 
            diff[jfile-1,ifile-1]=-float(cols[0]) # The offset for the reverse pair (jfile, ifile) is stored as the negative of the original offset
            nover[ifile-1,jfile-1]=int(cols[4]) # The number of overlapping pixels for the pair (ifile, jfile) is stored
            sigma[ifile-1,jfile-1]=float(cols[5]) # The standard deviation for the pair (ifile, jfile) is stored
            ldiff[ifile-1,jfile-1]=True  # valid offset measurement is set to True for the image pairs
        if (cols[0]=="END"):
            flist=False
            nfile=len(fdict)
            # initialization of diff, nover, sigma, ldiff matrices
            diff=np.zeros((nfile,nfile),dtype=np.float64)
            nover=np.zeros((nfile,nfile),dtype=np.int32)
            sigma=np.zeros((nfile,nfile),dtype=np.float64)
            ldiff=np.zeros((nfile,nfile),dtype=np.bool_)
        if (flist):
            fdict[int(cols[0])]=cols[1]
    f.close()
    print("Number of images identified: {:d}".format(len(fdict)))

    # convert list to numpy arrays
    x=np.array(x,dtype=np.float64) 
    xifl=np.array(xifl,dtype=np.int32)
    xjfl=np.array(xjfl,dtype=np.int32)
    y=np.array(y,dtype=np.float64) 
    c=np.array(c,dtype=np.int32) 
    s=np.array(s,dtype=np.float64) 
   
    print(np.mean(s)/np.sqrt(s.size) )

#    print(x.size)
#    print(y.size)
#    print(c.size)
#    print(s.size)

#    sval=1.0/np.sqrt(c)
#    sval=1.0/s
#    sval=s/np.sqrt(c)
#    sval=1./np.sqrt(c)
    
    sval=s/20.

    a0=np.zeros((nfile),dtype=np.float64)  # store the initial guesses for the offsets.

    if (args.boot):
        print("# Attempting bootstrap to obtain initial guess")
        ai=np.arange(nfile)  # store the indices of the images
        b=np.zeros((nfile),dtype=np.int16)  # store the status of the images
        w=np.where(xifl==0)  # store the indices of the images where the first image is 0
        a0[xjfl[w]]=y[w]  # the offset for the pair (0, jfile) is stored. reference image.
        b[0]=2  # the first image is set to 2
        b[xjfl[w]]=1  # the second image is set to 1

        print("a0 = ",a0)
        print(" b = ",b)
        print("ai = ",ai)

        boot_iter=0
        w1=np.where(b==1)
        print("Boostrap iter: {:d} complete".format(boot_iter))
        # iterative bootstrap process to update the offset
        while (ai[w1].size > 0):
            for i in ai[w1]:
                w2a=np.where(xifl == i)
                w2b=np.where(xjfl == i)
                w3a=np.where(b[xjfl[w2a]]==0)
                w3b=np.where(b[xifl[w2b]]==0)

                if (ai[xjfl[w2a]][w3a].size > 0):
                    print("NOTEMPTYa")
                    for j in ai[xjfl[w2a]][w3a]:
                        a0[j]=a0[i]+diff[i][j]
                        b[j]=1
                if (ai[xifl[w2b]][w3b].size > 0):
                    print("NOTEMPTYb")
                    for j in ai[xifl[w2b]][w3b]:
                        a0[j]=a0[i]+diff[i][j]
                        b[j]=1
                b[i]=2

            print("a0 = ",a0)
            print(" b = ",b)
            print("ai = ",ai)
            w1=np.where(b==1)
            boot_iter=boot_iter+1
            print("Boostrap iter: {:d} complete".format(boot_iter))

#       WHILE statement above ends here...
#   End Bootstrap initiative

    a0med=np.median(a0)
    print("# Offseting initial set of initial guesses by median {:f}".format(a0med))
    a0=a0-a0med

    # for i in range(y.size):
    #     if (i < 20):
    #         print(i,y[i],s[i],s[i]/np.sqrt(c[i]),sval[i])


    t0=time.time()
    aopt,acov=curve_fit(getoff,x,y,p0=a0,method='trf')
    t1=time.time()
    print("Time to fit {:d} image: {:.2f}".format(aopt.size,t1-t0))
    print(acov)
    print(aopt)

    amed=np.median(aopt)
    print("# Offseting first FIT result by median {:f}".format(amed))
    aopt=aopt-amed
    print(aopt)

    # use scipt.curve_fit to optimize the fit the model get_off
    aopt2,acov2=curve_fit(getoff,x,y,p0=aopt,sigma=sval,absolute_sigma=True,method='trf')
    aerr = np.sqrt(np.diag(acov2))
    print(acov2)
    print(aopt2)
    amed=np.median(aopt2)
    print("# Offseting first FIT result by median {:f}".format(amed))
    aopt2=aopt2-amed
    print(aopt2)

    chisq = np.sum(((getoff(x, *aopt2) - y)/sval)**2)
    chisq /= (x.size - aopt2.size)
    print(chisq)

    # write results to file
    # Write filename and the optimized offset
    fout=open(args.output,'w')
    for i in range(aopt.size):
        if (i < 190):
#            print(i,fdict[i+1],aopt[i],aerr[i])
            print(i,fdict[i+1],aopt2[i],aopt[i])
        fout.write(" {:s} {:f} \n".format(fdict[i+1],aopt2[i]))
#        print i,fdict[i+1],aopt2[i],aopt[i]
    fout.close()

    print(np.amin(diff),np.amax(diff),np.amin(aopt),np.amax(aopt))

    print("Time to fit {:d} image: {:.2f}".format(aopt.size,t1-t0))

    # Process completed
    exit(0)

