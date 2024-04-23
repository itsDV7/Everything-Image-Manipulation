#! /usr/bin/env python3
"""
Method to find offsets between images
1. Read images from list directory
2. Read fluxscale info from tan_nwgint directory
3. Read WCS info from tan_nwgint directory
4. Use FITSIO (https://github.com/esheldon/fitsio) to read images
5. Use WCS (https://github.com/DarkEnergySurvey/despyastro/blob/master/python/despyastro/wcsutil.py) to transform pixel coordinates between images
6. Use sigma clipping to find median difference between two images
7. Usage `python3 findoff.py -i "data/list/sci.g.list" -o "out/test.g.offset_b8" -v 1 --useTAN --fluxscale "data/list/flx.g.list"`
"""

import os
import argparse
import re
import time
import fitsio
import wcsutil
import numpy as np


###########################################
def get_data(filename,verbose=0):
    """Function to obtain header, image, mask data
    """

    ifits=fitsio.FITS(filename,'r') # Could be changed to 'rw' as needed
    ih=ifits['SCI'].read_header()
    isci=ifits['SCI'].read()
    imsk=ifits['MSK'].read()
    iwgt=ifits['WGT'].read()
    ifits.close()
    if (verbose > 0):
        print("Successfully read {:s}".format(filename))

    return ih,isci,imsk,iwgt
  

###########################################
def medclip(data,clipsig=5.0,maxiter=50,converge_num=0.0001,verbose=0):
    """
    Perform sigma clipping algorithm on the given dataset to robustly estimate its mean, median, and standard deviation.
    Sigma clipping is a technique used to identify and remove outliers from data, which can skew statistical measures. 
    The method iterates through the data, recalculating statistical measures each time and excluding data points that lie beyond a specified number of standard deviations (sigma) from the median. 
    This process is repeated until convergence is achieved or a maximum number of iterations is reached. 
    This method is useful in astronomical data analysis and other fields where it is important to mitigate the impact of outliers on statistical measures.
    Args:
        data (numpy.ndarray): The input data to be processed.
        clipsig (float): The number of standard deviations to clip the data at.
        maxiter (int): The maximum number of iterations to perform.
        converge_num (float): The convergence criterion for the algorithm.
        verbose (int): The verbosity level.
    Returns:
        tuple: A tuple containing the estimated mean, median, standard deviation, and the number of pixels used for the calculation.
    """
    ct = data.size
    iter = 0; c1 = 1.0 ; c2 = 0.0

    avgval = np.mean(data)
    medval = np.median(data)
    sig = np.std(data)
    wsm = np.where( abs(data-medval) < clipsig*sig )
    if ((verbose > 0)and(verbose < 4)):
        print("iter,avgval,medval,sig")
    if ((verbose > 2)and(verbose < 4)):
        print(0,avgval,medval,sig)
    if (verbose > 3):
        print("iter,avgval,medval,sig,ct,c1,c2")
        print(0,avgval,medval,sig,ct,c1,c2)
    
    # iterative clipping loop
    while (c1 >= c2) and (iter < maxiter):
        iter += 1
        lastct = ct
        avgval = np.mean(data[wsm])
        medval = np.median(data[wsm])
        sig = np.std(data[wsm])
        wsm = np.where( abs(data-medval) < clipsig*sig )
        ct = len(wsm[0])
        if ct > 0:
            c1 = abs(ct - lastct)
            c2 = converge_num * lastct
        if ((verbose > 2)and(verbose < 4)):
            print(iter,avgval,medval,sig)
        if (verbose > 3):
            print(iter,avgval,medval,sig,ct,c1,c2)
#   End of while loop
    if (iter >= maxiter):
        # convergence warning
        print("Warning: medclip had not yet converged after {:d} iterations".format(iter))

    medval = np.median(data[wsm])
    avgval = np.mean(data[wsm])
    stdval = np.std(data[wsm])
    if (verbose > 0):
        print(iter+1,avgval,medval,sig)

    return avgval,medval,stdval,ct


###########################################
def med_diff(ImgDict,iImg,jImg,minpix=500,verbose=0):
    """
    Calculate the median difference between two images, accounting for the overlap region.
    If there are not enough pixels to perform the calculation (fewer than minpix), 
        it returns 0.0 for the median difference, 
        -1 for the number of pixels, 
        and -1.0 for the standard deviation as indicators of insufficient data.
    Args:
        ImgDict (dict): A dictionary containing the image data and metadata.
        iImg (str): The name of the first image.
        jImg (str): The name of the second image.
        minpix (int): The minimum number of pixels required for the calculation.
        verbose (int): The verbosity level.
    Returns:
        tuple: A tuple containing the median difference, standard deviation, and the number of pixels used for the calculation.
    
    Steps:
    1. Read Image Data: Gets the image data from two images. This includes the images themselves along with their mask and weight maps.
    2. Scale Images: The pixel values of both images are scaled by their respective flux scales found in ImgDict. This is necessary to ensure that the images are on the same flux scale for a fair comparison.
    3. Reshape and Filter: The function reshapes the weight maps of the images to a 1D array and filters out pixels with a weight of 0.0, indicating bad or unreliable data.
    4. Coordinate Transformation: It converts the pixel coordinates of the first image to sky coordinates (Right Ascension and Declination) using the WCS (World Coordinate System) information, and then transforms these sky coordinates back to pixel coordinates in the second image's frame. This step aligns the two images spatially, allowing for direct pixel-by-pixel comparison.
    5. Calculate Differences: The function calculates the difference in pixel values between the two images for overlapping pixels (i.e., pixels that are within the bounds of both images and have non-zero weight in the second image).
    6. Median Clipping: If the number of pixels with valid differences is greater than or equal to minpix, it performs a sigma clipping algorithm (using the medclip function) on the differences to robustly estimate the median difference, its standard deviation, and the number of pixels used in the calculation. This step helps to remove outliers and focus on the core distribution of pixel value differences.

    """

    t0=time.time()
    # read the two image data
    ih,isci,imsk,iwgt=get_data(os.path.join("data",iImg))
    jh,jsci,jmsk,jwgt=get_data(os.path.join("data",jImg))
    t1=time.time()
    if (verbose > 2):
        print("Read images: {:.2f} ".format(t1-t0))

    isci=isci*ImgDict[iImg]['fluxscale']
    jsci=jsci*ImgDict[jImg]['fluxscale']
    t1a=time.time()
    if (verbose > 2):
        print("Scale mages: {:.2f} ".format(t1a-t1))

    iwgt2=np.reshape(iwgt,iwgt.size)
    iwsm=np.where(iwgt2>0.0)

    t2=time.time()
    if (verbose > 2):
        print("Reshape images: {:.2f} ".format(t2-t1a))

    iImg_iy, iImg_ix = np.indices(isci.shape)
#   remove points that are masked (have wgt=0)
    iImg_ix=np.reshape(iImg_ix,iImg_ix.size)[iwsm]
    iImg_iy=np.reshape(iImg_iy,iImg_iy.size)[iwsm]
    iImg_rx=iImg_ix.astype('f8')
    iImg_ry=iImg_iy.astype('f8')

    t3=time.time()
    if (verbose > 2):
        print("Form pixel arrays: {:.2f} ".format(t3-t2))

    iRA,iDec=ImgDict[iImg]['wcs'].image2sky(iImg_rx,iImg_ry)
    t4a=time.time()
    if (verbose > 2):
        print("Transform(1) pixel arrays: {:.2f} ".format(t4a-t3))
    jImg_ix,jImg_iy=ImgDict[jImg]['wcs'].sky2image(iRA,iDec)
    t4=time.time()
    if (verbose > 2):
        print("Transform(2) pixel arrays: {:.2f} ".format(t4-t4a))

    jImg_ix=np.rint(jImg_ix).astype(int)
    jImg_iy=np.rint(jImg_iy).astype(int)
#    print(iImg_ix.dtype)
#    print(jImg_ix.dtype)

    jwsm=np.where(np.logical_and(np.logical_and(jImg_ix>0,jImg_ix<jh['NAXIS1']),np.logical_and(jImg_iy>0,jImg_iy<jh['NAXIS2'])))
    t5=time.time()
    if (verbose > 2):
        print("Mask non-overlap: {:.2f} ".format(t5-t4))

    nover=jImg_ix[jwsm].size
    if (verbose > 2):
        print("Preliminary number of overlaping pixels: {:d}".format(nover))

    j_ix=jImg_ix[jwsm]
    j_iy=jImg_iy[jwsm]
    i_ix=iImg_ix[jwsm]
    i_iy=iImg_iy[jwsm]

    diff=isci[i_iy,i_ix]-jsci[j_iy,j_ix]
    dwgt=jwgt[j_iy,j_ix]
    dwsm=np.where(dwgt>0)

    MedPix=diff[dwsm].size

    if (MedPix >= minpix):
        mdiffval=diff[dwsm]
        AvgDiff,MedDiff,MedStd,MedPix=medclip(mdiffval,verbose=0)
    if (MedPix >= minpix):
        if (verbose > 0):
            t6=time.time()
            #print("Final values (npix,med_diff,med_std,timing): {:8d} {:12.5f} {:12.5f} {:6.2f}".format(MedPix,MedDiff,MedStd,(t6-t0)))
    else:    
        MedDiff=0.0
        MedPix=-1
        MedStd=-1.0
    
    return MedDiff,MedStd,MedPix


################################################
if __name__ == "__main__":
    # usage `python3 findoff.py -i "list/sci.g.list" -o "out/test.g.offset_b8" -v 1 --useTAN --fluxscale "list/flx.g.list"`
    t00=time.time()
    parser = argparse.ArgumentParser(description='Code to take offset measurement and find optimal set of per image offsets for the ensemble') 

    parser.add_argument('-i','--input',   action='store', type=str, default=None, required=True,  help='Input image list')
    parser.add_argument('-o','--output',  action='store', type=str, default=None, required=True,  help='Output file of offset measurements')
    parser.add_argument('--fluxscale',    action='store', type=str, default=None, help='Optional set of fluxscales that need to be applied to data')
    parser.add_argument('--magzero',      action='store', type=str, default=None, help='Optional set of ZeroPoints to convert to fluxscales and applied to data')
    parser.add_argument('--magbase',      action='store', type=float, default=30.0, help='MagBase for converting magzero to fluxscale (default=30.0)')
    parser.add_argument('--useTAN',        action='store_true', default=False, required=False, help='Flag to use tan_nwgint variant of input images')

    parser.add_argument('-v','--verbose', action='store', type=int, default=0,    required=False, help='Verbosity (default=0, max=2)')
    parser.add_argument('-b','--boot',    action='store_true', default=False,     required=False, help='Use bootstrap to make an intial guess)')

    args = parser.parse_args()
    if (args.verbose > 0):
        print("Args: {:}".format(args))

    # get information from input image list
    f=open(args.input,'r')

    i=0
    flist=[]  # list of image names
    fdict={}  # dictionary of image names, id, fluxscales, and WCS
    # read image list
    for line in f:
        cols=line.split()
        fname0=re.sub(r"\[0\]","",cols[0])
        if (args.useTAN):
            fname=re.sub("coadd_nwgint","tan_nwgint",fname0)  # just to match the filenames in tan_nwgint directory correctly
            fname=re.sub("_nwgint.fits","_nwgint_tan.fits",fname)
        flist.append(fname)
        fdict[fname]={}
        fdict[fname]['inum']=i
        fdict[fname]['fname0']=fname0
        i=i+1
    print("Found {:d} image files".format(len(flist)))
    f.close()

    # read fluxscale info
    if ((args.fluxscale is None)and(args.magzero is None)):
        if (args.verbose > 0): 
            print("No --fluxscale or --magzero.  Assuming all fluxscales are 1.0")
        for fname in fdict:
            fdict[fname]['fluxscale']=1.0
    else:
        if (args.fluxscale is not None):
#            print("RAG: Going the fluxscale route")
            if (os.path.isfile(args.fluxscale)):
                f_flux=open(args.fluxscale,'r')
                useFluxScale=True
        elif (args.magzero is not None):
#            print("RAG: Going the magzero route")
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
            if (args.useTAN):
                fname=re.sub("coadd_nwgint","tan_nwgint",fname)  # just to match the filenames correctly
                fname=re.sub("_nwgint.fits","_nwgint_tan.fits",fname)
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

    # FITSIO and WCS on images
    file_missing=False
    ts0=time.time()
    for Img in flist:
        if (os.path.isfile(os.path.join("data",Img))):
            ih,isci,imsk,iwgt=get_data(os.path.join("data",Img))
            fdict[Img]['header']=ih
            fdict[Img]['crossra0']=ih['CROSSRA0']
            fdict[Img]['ra_cent']=ih['RA_CENT']
            fdict[Img]['dec_cent']=ih['DEC_CENT']
            if (ih['CROSSRA0'] == "Y"):
                fdict[Img]['ra_size']=(360.0-ih['RACMAX'])+ih['RACMIN']
                fdict[Img]['dec_size']=ih['DECCMAX']-ih['DECCMIN']
            else:
                fdict[Img]['ra_size']=ih['RACMAX']-ih['RACMIN']
                fdict[Img]['dec_size']=ih['DECCMAX']-ih['DECCMIN']
            fdict[Img]['wcs']=wcsutil.WCS(ih)
            #print("Read WCS for {:s} will be using a fluxscale of {:.5f} ".format(Img,fdict[Img]['fluxscale']))
        else:
            print("File: {:s} not found.".format(Img))
            file_missing=True
    if (file_missing):
        print("Missing file(s).  Aborting!")
        exit(1)
    ts1=time.time()
    print("Timing (acquire WCS): {:.2f}".format(ts1-ts0))

    # identify which image pairs have overlapping regions
    nimg=len(flist)
    imatch=np.zeros((nimg,nimg),dtype='int16')  # store information about whether pairs of images overlap (1 = overlap, 0 = no overlap)
 
    for iImg in flist:
        for jImg in flist:
            if (iImg == jImg):
                imatch[fdict[iImg]['inum'],fdict[jImg]['inum']]=0
            else:
                # checks if either image has a CROSSRA0 flag set to "Y". 
                # This flag indicates that the image crosses the 0 or 360-degree line in Right Ascension (RA), which is a special case for overlap calculation.
                # The code adjusts RA values accordingly to handle this case.
                if ((fdict[iImg]['crossra0'] == "Y")or(fdict[jImg]['crossra0'] == "Y")):
                    print("Either {:s} or {:s} or both have CROSSRA0 == Y".format(iImg,jImg))
                    ra1=fdict[iImg]['ra_cent']
                    ra2=fdict[jImg]['ra_cent']
                    if (ra1 > 180.):
                        ra1=ra1-360.0
                    if (ra2 > 180.):
                        ra2=ra2-360.0
                    dra=ra1-ra2
                    printf("Checking that CROSSRA0 case is working")
                    printf("  {:12.7f} --> {:12.7f} for {:s} ".format(fdict[iImg]['ra_cent'],ra1,iImg))
                    printf("  {:12.7f} --> {:12.7f} for {:s} ".format(fdict[jImg]['ra_cent'],ra2,jImg))
                    printf("  {:12.7f}  ".format(dra))
                else:
                    dra=fdict[iImg]['ra_cent']-fdict[jImg]['ra_cent']
                # overlap calculation
                # calculate the difference in RA and Declination (Dec) centers between the two images (dra and ddec) and compute the average sizes in RA and Dec for the images (ra_size and dec_size).
                ddec=fdict[iImg]['dec_cent']-fdict[jImg]['dec_cent']
                ra_size=0.5*(fdict[iImg]['ra_size']+fdict[jImg]['ra_size'])
                dec_size=0.5*(fdict[iImg]['dec_size']+fdict[jImg]['dec_size'])
                if (args.verbose > 3):
                    print("dr={:12.7f} dd={:12.7f} rs={:12.7f} ds={:12.7f} ".format(dra,ddec,ra_size,dec_size))                    
                OverLap=False
                OverLapStr="NoOverlap"
                # Images overlap if the absolute differences in RA and Dec are less than the average sizes
                if ((abs(dra) < ra_size)and(abs(ddec) < dec_size)):
                    OverLap=True
                    OverLapStr="  Overlap"
                    imatch[fdict[iImg]['inum'],fdict[jImg]['inum']]=1
                # verbose output
                if (args.verbose > 3):
                    print("{ovstr:9s}: {r1:12.7f} {r2:12.7f} {d1:12.7f} {d2:12.7f} ".format(
                        ovstr=OverLapStr,
                        r1=fdict[iImg]['header']['RACMIN'],
                        r2=fdict[iImg]['header']['RACMAX'],
                        d1=fdict[iImg]['header']['DECCMIN'],
                        d2=fdict[iImg]['header']['DECCMAX']))
                    print("     with: {r1:12.7f} {r2:12.7f} {d1:12.7f} {d2:12.7f} ".format(
                        r1=fdict[jImg]['header']['RACMIN'],
                        r2=fdict[jImg]['header']['RACMAX'],
                        d1=fdict[jImg]['header']['DECCMIN'],
                        d2=fdict[jImg]['header']['DECCMAX']))

    #print(imatch)
    
    fout=open(args.output,'w')
    for iImg in flist:
        fout.write(" {inum:6d} {fname:s} \n".format(
            inum=fdict[iImg]['inum']+1,
            fname=fdict[iImg]['fname0']))
    fout.write("END OF FILELIST\n")

    # loop over all image pairs and write output to file
    count=1
    for iImg in flist:
        for jImg in flist:
            if (fdict[jImg]['inum'] > fdict[iImg]['inum']):
                if (imatch[fdict[iImg]['inum'],fdict[jImg]['inum']] == 1):
                    if (args.verbose > 1):
                        print(iImg,jImg)
                    medoff,medsig,npix=med_diff(fdict,iImg,jImg,minpix=500,verbose=args.verbose)
                    if (npix >= 500):
                        fout.write(" {offval:11.3f} {cval:11.3f} {inum:6d} {jnum:6d} {pixval:10d} {offsig:12.4f} \n".format(
                            offval=medoff,
                            cval=count,
                            inum=fdict[iImg]['inum']+1,
                            jnum=fdict[jImg]['inum']+1,
                            pixval=npix,
                            offsig=medsig))
                count=count+1
    fout.close()    

    print("Total execution time: {:.2f} seconds".format(time.time()-t00))

    # Process completed
    exit(0)                    
    
