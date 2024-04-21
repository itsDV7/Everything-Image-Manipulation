#! /usr/bin/env python3
"""
"""

import os
import argparse
import re
import time
import fitsio
import numpy as np


###########################################
def get_data(file,verbose=0):
    """Function to obtain header, image, mask data
    """
    ifits=fitsio.FITS(file,'r') # Could be changed to 'rw' as needed
    ih=ifits['SCI'].read_header()
    isci=ifits['SCI'].read()
    iwh=ifits['WGT'].read_header()
    iwgt=ifits['WGT'].read()
    iwh2=ifits['WGT_ME'].read_header()
    iwgt2=ifits['WGT_ME'].read()
    imh=ifits['MSK'].read_header()
    imsk=ifits['MSK'].read()
    ifits.close()

    return ih, isci, iwh, iwgt, iwh2, iwgt2, imh, imsk
  


################################################
if __name__ == "__main__":

    t00=time.time()
    parser = argparse.ArgumentParser(description='Code to take offset measurement and find optimal set of per image offsets for the ensemble') 

    parser.add_argument('-i','--input',   action='store', type=str, default=None, required=True,  help='Input image list w/ offsets')
    parser.add_argument('-o','--output',  action='store', type=str, default=None, required=True,  help='Output file of offset measurements')
    parser.add_argument('--fluxscale',    action='store', type=str, default=None, help='Optional set of fluxscales that need to be removed from offsets')
    parser.add_argument('--magzero',      action='store', type=str, default=None, help='Optional set of ZeroPoints to convert to fluxscales and to be removed from offsets')
    parser.add_argument('--magbase',      action='store', type=float, default=30.0, help='MagBase for converting magzero to fluxscale (default=30.0)')

    parser.add_argument('-v','--verbose', action='store', type=int, default=0,    required=False, help='Verbosity (default=0, max=2)')
    parser.add_argument('-b','--boot',    action='store_true', default=False,     required=False, help='Use bootstrap to make an intial guess)')

    args = parser.parse_args()
    if (args.verbose > 0):
        print("Args: {:}".format(args))


    f=open(args.input,'r')

    i=0
    flist=[]
    fdict={}
    for line in f:
        cols=line.split()
        fname=re.sub(r"\[0\]","",cols[0])
        flist.append(fname)
        fdict[fname]={}
        fdict[fname]['inum']=i
        fdict[fname]['offset']=float(cols[1])
        i=i+1
    print("Found {:d} image files and offsets".format(len(flist)))
    f.close()

    if ((args.fluxscale is None)and(args.magzero is None)):
        if (args.verbose > 0): 
            print("No --fluxscale or --magzero.  Assuming all fluxscales are 1.0")
        for fname in fdict:
            fdict[fname]['fluxscale']=1.0
    else:
        if (args.fluxscale is not None):
            print("RAG: Going the fluxscale route")
            if (os.path.isfile(args.fluxscale)):
                f_flux=open(args.fluxscale,'r')
                useFluxScale=True
        elif (args.magzero is not None):
            print("RAG: Going the magzero route")
            if (os.path.isfile(args.magzero)):
                f_flux=open(args.magzero,'r')
                useFluxScale=False
        else:
            print("No fluxscale or magzero?  This line should never be able to execute")

        i2=0
        for line in f_flux:
            cols=line.split()
            fname=re.sub(r"\[0\]","",cols[0])
            if (useFluxScale):
                fscale=float(cols[1])
            else:
                fscale=10.**(0.4*(args.magbase-float(cols[1])))
            if (fname in fdict):
                fdict[fname]['fluxscale']=fscale
                i2=i2+1
            else:
                print("{:s} not found in fdict".format(fname))
        f_flux.close()
        if (useFluxScale):
            print("Found {:d} fluxscale entries made in fdict".format(i2))
        else:
            print("Found {:d} fluxscale entries made in fdict (from --magzero)".format(i2))
        
    i3=0
    for fname in fdict:
        if ('fluxscale' in fdict[fname]):
            i3=i3+1
    print("Found {:d} fluxscale entries in fdict".format(i3))

###############################


    file_missing=False
    ts0=time.time()
    for Img in flist:
        if (os.path.isfile(os.path.join("data", Img))):
            ih, isci, iwh, iwgt, iwh2, iwgt2, imh, imsk = get_data(os.path.join("data", Img))
            offval=fdict[Img]['offset']/fdict[Img]['fluxscale']/64.
            print(Img,fdict[Img]['offset'],fdict[Img]['fluxscale'],offval)
            isci=isci+offval

            oname=re.sub(".fits","_hack.fits",os.path.join("data", Img))

            ofits = fitsio.FITS(oname,'rw',clobber=True)
            ofits.write(isci,header=ih,extname='SCI')
            ofits.write(iwgt,header=iwh,extname='WGT')
            ofits.write(iwgt2,header=iwh2,extname='WGT_ME')
            ofits.write(imsk,header=imh,extname='MSK')
            ofits.close()

    exit(0)


