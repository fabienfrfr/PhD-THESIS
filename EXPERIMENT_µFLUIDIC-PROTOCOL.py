# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 18:20:31 2018

@author: Fabien
"""
#package
import time, numpy as np, pandas as pd, pylab as plt, os, skimage.io

def imgAcquisition(m,exposure,intensity,channel,posXYZ,folder,date,n,LOGPATH):
    data = open(LOGPATH, "a")
    #camera property:
    m.mmc.setProperty(m.cam,'Exposure',exposure)
    #Start continuous camera acquisition
    m.mmc.startContinuousSequenceAcquisition(1)
    #fluo parameter :
    m.xcite.setIntensity(intensity)
    for c in channel : 
        #filter (debut de la boucle filter) :
        m.olympus.filterCube(c);time.sleep(1)
        #open shutter :
        m.xcite.openShutter()
        time.sleep(0.1)
        i = 0
        for p in posXYZ :
            #move to :
            m.setXYZpos(p);time.sleep(1)
            #get frame
            frame = m.mmc.getLastImage()
            plt.imshow(frame);plt.show();plt.close()
            skimage.io.imsave(folder + date + os.path.sep + str(i) + os.path.sep + c + '_' + str(n).zfill(4) + '.tif', frame)
            data.write('\n' + str(time.time()) + '_' + str(n) + '_' + str(c) + '_' + str(p))
            i += 1
        #close shutter :
        m.xcite.closeShutter()
    m.mmc.stopSequenceAcquisition()
    data.close()

from HARDCCC import Sherlock,ScheduleValv

### INIT

logPath = r"C:\Users\Administrateur\Documents\Python Scripts\20181120\log_20181122.txt"
m = Sherlock()

#Time parameter :
interval = 30. #minutes
duration = 7.  #days
N = int((duration*60*24)/interval)

#Medium parameter "medium, t_start, width of pulse, freq" :
i_medium = ['i3','i4'] # NA-AFK [0,1]
condition_mDPF_per_ch = [[0,0,0],[1000,0,0],[20,0,0],[40,0,0],[80,0,0],[20,3,0],[40,3,0],[80,3,0],[20,3,5],[20,3,10],[20,3,20],[20,3,40],
                  [0,0,0],[1000,0,0],[20,0,0],[40,0,0],[80,0,0],[20,3,0],[40,3,0],[80,3,0],[20,3,5],[20,3,10],[20,3,20],[20,3,40]]

#Valve Schedule initialisation
sv = ScheduleValv(condition_mDPF_per_ch,N,24*duration,i_medium,logPath)

### EXP

#exp data and create main folder
date = '20181122_BRA-GFP-NLS-mCherry_DelaiPulseAFK_flush5psi_chip2.4'; folder = r'D:/DATA/'
if(not os.path.isdir(folder + date)): os.makedirs(folder + date)


#define position and create folder
if(os.path.exists(folder + date + '_posXY.csv')) : posXYZ = pd.read_csv(folder + date + '_posXY.csv')
posXYZ = m.LiveCV(50) #2 images per chambers
posXYZ = np.asarray(posXYZ)

pd.DataFrame(posXYZ.astype('int'), columns=('x', 'y', 'z')).to_csv(folder + date + '_posXYZ.csv')
for i in range(len(posXYZ)) : 
    if(not os.path.isdir(folder + date + os.path.sep + str(i))): os.makedirs(folder + date + os.path.sep + str(i))

#Image parameter :
channel = ['RFP','GFP']
intensity = '50'; exposure = '300'

"""
### SCRIPT

START = time.time()
for n in range(N) :
    start = time.time(); print(n,' : ',((START-start)/60))
    
    #image acquisition :
    imgAcquisition(m,exposure,intensity,channel,posXYZ,folder,date,n,logPath)
    
    #CCC schedule :
    sv.applyS(n)
    # Pause between 2 times
    end = time.time()
    time.sleep(interval*60-(end-start))
"""
