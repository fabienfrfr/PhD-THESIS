#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:34:55 2020
@author: fabien
https://www.oreilly.com/library/view/python-data-science/9781491912126/ch04.html
"""
import numpy as np, pylab as plt, pandas as pd
import os, sys
from sklearn import preprocessing
from skimage import io

#################################### PARAM :
## CSV path
PATH = r'/media/fabien/F6166FAF166F7013/0_THESE/ARTICLE/20201209/201129-mESC_pSmad2-IF_curve/'
CURVE_FILE = '201129-mESC_pSmad2-IF' # 201119-mESC_pSmad2-IF  201129-mESC_pSmad2-IF
PATH_IMG = r'/media/fabien/F6166FAF166F7013/0_THESE/ARTICLE/20201209/201129-mESC_pSmad2-IF'
IS_SPLITED = False
EXP_PARAM = 'None' # String 'imageTimes', 'experiment-log' or 'None'
IS_CCC = False # MATLAB LOG VERSION (need to convert in csv)

## Columns of CSV  (see 1st line of csv)
CONDITION = np.array(['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8','C9'])
CELL_INFO = np.array(['frame', 'cell', 'x', 'y', 'I_nucleus', 'I_cytoplasm', 'ratio'])

# Figure
W, H, L, S = 3.7, 2.9, 18., 9. # width, height, label_size, scale_size
LW = 0.1
VAL_ = [0., 7.5]
T = [-.5, 1.5]
EXP_IDX = np.array([9,8,3])# values or None
STD = np.pi

# Stat-param
OUT_QUANTILE = 0.999

## Img_density_visualisation
IDX_IMG = [8,9]

## Plotting classification (hierarchy info)
'''Cond savefig sep -> curve per condition -> X_axis grouping -> Y_axis to see (2 if ratio)'''
# Indicate index (see Condition and cell info order)
IDX_SAVE_CURVE = [8, 4] #cond
IDX_AXIS = [0,-1] #cell
IDX_LABEL = [3] #cond

## Info variable output
NAME_VAR = 'Smad2-RFP'
VAR_ = 'IF-log' # (norm) or ratio

###############################################################################
#################################### INITIALISATION :
# Convert scale figure
MM2INCH = 2.54
W, H, L, S = np.array((W, H, L, S))/MM2INCH # ratio fig : 2.7/2.1
# Level
LVL_ORDER = np.concatenate((CONDITION[IDX_SAVE_CURVE], CELL_INFO[IDX_AXIS]))
# Labeling
LABEL = CONDITION[IDX_LABEL]
# Import csv
#List_csv_path = [PATH + f for f in sorted(os.listdir(PATH)) if (".csv" in f[-4:] and not(EXP_PARAM in f))]
DATA = pd.read_csv(PATH + CURVE_FILE + '.csv')

# Create the folder for analysis
if(not os.path.isdir(CURVE_FILE+'_curve')): os.makedirs(CURVE_FILE+'_curve')
# Parameter verification and Affect name str
idx2test = np.where(LVL_ORDER != None)[0]
if False in [n in DATA.columns for n in LVL_ORDER[idx2test]] : sys.exit("Error parameter !!!!")

#################################### DATASET STAT :
## Preprocessing of data distribution (Outlier/aberration, Normalisation)
outlier = DATA[LVL_ORDER[-1]].quantile(OUT_QUANTILE) # 99,5% of statistical sample follow 'I'
DATA = DATA[DATA[LVL_ORDER[-1]] < outlier]

if VAR_ == '(norm)' :
    min_max_scaler = preprocessing.MinMaxScaler()
    DATA[LVL_ORDER[-1]] = min_max_scaler.fit_transform(DATA[[LVL_ORDER[-1]]].values)
elif VAR_ == 'ratio' :
    DATA[LVL_ORDER[-1]] = DATA['I_nucleus']/DATA['I_cytoplasm'] ## NOT GOOD !! 
plt.hist(DATA[LVL_ORDER[-1]].values, 100); plt.show(); plt.close() # Poisson distrib, peak at ~0.1 (no signal)

#################################### 2D density Hist-IMG :
listIMG = [PATH_IMG + os.path.sep + f for f in sorted(os.listdir(PATH_IMG)) if '.tif' in f]
C = CONDITION[IDX_IMG]
for IMG in listIMG :
    c = IMG.split('.')[0].split('_')[-2:]
    img = io.imread(IMG)
    df_img = DATA[(DATA[C[0]] == c[0]) * (DATA[C[1]] == int(c[1]))]
    x,y = df_img.x.values, df_img.y.values
    #figures
    fig = plt.figure()
    grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
    main_ax = fig.add_subplot(grid[:-1, 1:])
    y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
    x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)
    #main axis
    main_ax.imshow(img)
    main_ax.scatter(x,y, s=3, alpha=0.5 )
    # hist
    x_hist.hist(x, 40, histtype='stepfilled', orientation='vertical', color='gray')
    x_hist.invert_yaxis()
    y_hist.hist(y, 40, histtype='stepfilled', orientation='horizontal', color='gray')
    y_hist.invert_xaxis()
    # save
    plt.savefig(CURVE_FILE+'_curve' + os.path.sep + CURVE_FILE + '_' + c[0] + '_' + c[1] + ".png", dpi=720)
    plt.show();plt.close()


#################################### HIERARCHY PLOT :
# Verify if LVL it's for one files output
if LVL_ORDER[0] == None :
    DATA_LOOP = [(CURVE_FILE,DATA)]
else :
    DATA_LOOP = DATA.groupby([LVL_ORDER[0]])

# Figure
fig = plt.figure(figsize=(W, H))

plt.rc('font', size=S)
plt.rc('axes', titlesize=S)

ax = fig.add_subplot()
ax.set_title(CURVE_FILE, fontsize=L)
#ax.set_ylabel("N", fontsize=L)
ax.set_ylabel("log " + NAME_VAR, fontsize=L)

# histogram
kwargs = dict(histtype='stepfilled', linewidth=LW, density=True, alpha=0.5, bins=40, ec="k")
# Loop data
data, labels = [], [] 
for DL in DATA_LOOP :
    label, df = DL[0], DL[1]

    sample = np.log(df[LVL_ORDER[3]])
    # Hist
    #ax.hist(sample, label = label, **kwargs)
    data += [sample]
    labels += [label]

ax.boxplot(data, notch=True, labels=labels)
#ax1.set_xticklabels(np.repeat(random_dists, 2), rotation=45, fontsize=8)
plt.xticks(list(range(1,len(labels)+1)), labels, rotation=45)
# Legend
ax.legend()
#plt.xlim(T), plt.ylim(VAL_)
# Save data
plt.savefig(CURVE_FILE+'_curve' + os.path.sep + CURVE_FILE + '_' + VAR_ + ".svg")
plt.savefig(CURVE_FILE+'_curve' + os.path.sep + CURVE_FILE + '_' + VAR_ + ".png", dpi=720)
plt.close()
