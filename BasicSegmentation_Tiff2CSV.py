#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 13:16:57 2020
@author: fabien

"""
import cv2, os, time
import numpy as np, pandas as pd#, pylab as plt
from skimage import external#, io
from skimage.feature import peak_local_max

## PARAM :
folder_path = '/media/fabien/Seagate/DATA/20201009' # à définir
name_data_folder = folder_path.split(os.path.sep)[-1]

channel_order = [0, 1] # Segmentation channel, Measure channel; Often RGB order
NB_channel = 3

Size_cell = 25

## List des .tif du repertoire courant
listIMG = [folder_path + os.path.sep + f for f in sorted(os.listdir(folder_path)) if '.tif' in f]


cells = [] # Struct : (Protocole, frame, cell?, Nucleus, Cytoplasme, Ratio)
## Loop per files
for f in listIMG :
    print(f); start = time.time()
    condition = f.split(os.path.sep)[-1].split('.')[0]
    condition = tuple(condition.split('_'))
    ## Read 16bits images (and get info)
    TIFF = external.tifffile.TiffFile(f)
    IMG = TIFF.asarray()
    img_info = TIFF.info()   # IJ.runMacro('getImageInfo()'); A = MIJ.getLog
    
    # Rearrangment of image shape (if necessary)
    IMG = np.rollaxis(IMG, np.where(np.array(IMG.shape) == NB_channel)[-1], 1)
    
    #Prepare xy matrix for location of cell (np.mean(np.where(markers == label), axis = 1) is too long)
    x, y = np.arange(0, IMG[0,0].shape[1]), np.arange(0, IMG[0,0].shape[0])
    xx, yy = np.meshgrid(x, y)
    
    ## Loop per image "frame"
    n = 0
    for img in IMG :
        ## Extract channel
        segm_img, measure_img = img[channel_order[0]], img[channel_order[1]]
        
        ## Pretreatment for segmentation (8bits norm + smooth)
        img_norm_8U = cv2.normalize(segm_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img_blur = cv2.GaussianBlur(img_norm_8U,(5,5),0)
        
        ## SEGMENTATION
        # Binarisation part (thresh + dilate)
        ret, thresh = cv2.threshold(img_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((5,5),np.uint)
        dilation = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
        dilation[dilation == 255] = 1
        
        # Watershed 
        dist_transform = cv2.distanceTransform(dilation,cv2.DIST_L2,5)
        local_maxi = peak_local_max(dist_transform, indices=False, labels=thresh)
        ret, markers = cv2.connectedComponents(np.uint8(local_maxi))
        markers = cv2.watershed(cv2.cvtColor(dilation,cv2.COLOR_GRAY2BGR),markers)
        
        # Post-treatment of watershed (only cells)
        markers = markers*dilation
        
        # Extract backgrounds (index cell = 0)
        bg_thresh, bg_dilate = (1 - thresh)*measure_img, (1 - dilation)*measure_img
        m_bg_thresh, m_bg_dilate = np.mean(bg_thresh[bg_thresh != 0]), np.mean(bg_dilate[bg_dilate != 0])
        pos_x, pos_y = np.mean(xx[thresh == 0]), np.mean(yy[thresh == 0])
        cells += [condition + (n, 0, pos_x, pos_y, m_bg_thresh, m_bg_dilate, m_bg_thresh/m_bg_dilate)]
        
        # Extract info per cells (index cell = 1) :
        mask_nucleus, mask_cyto = thresh*measure_img, dilation*measure_img
        mask_cyto[thresh == 1] = 0
        
        label_list = np.unique(markers)
        label_list = label_list[np.where(label_list > 0)] # delete -1 and 0  (background)
        for label in label_list :
            cell_nucleus = mask_nucleus[markers == label]
            cell_cyto = mask_cyto[markers == label]
            # List of cells (without absurd value)
            cell_nucleus, cell_cyto = cell_nucleus[cell_nucleus != 0], cell_cyto[cell_cyto != 0]
            if len(cell_cyto) > Size_cell :
                pos_x, pos_y = np.mean(xx[markers == label]), np.mean(yy[markers == label])
                m_nuc, m_cyto = np.mean(cell_nucleus), np.mean(cell_cyto)
                cells += [condition + (n, 1, pos_x, pos_y, m_nuc, m_cyto, m_nuc/m_cyto)]
        # Increment time
        n += 1
    print(time.time()-start)
condition_name = tuple(['C'+str(i) for i in range(len(condition))])
proprieties = pd.DataFrame(cells, columns= condition_name + ('frame', 'cell', 'x', 'y', 'I_nucleus', 'I_cytoplasm', 'ratio'))
proprieties.to_csv(os.getcwd() + os.path.sep + name_data_folder + '.csv')
