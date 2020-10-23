#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:34:55 2020
@author: fabien
"""
import numpy as np, pylab as plt, pandas as pd
import os, sys
from sklearn import preprocessing

#################################### PARAM :
## CSV path
PATH = r'/media/fabien/F6166FAF166F7013/0_UNIX/DEV/Python_scripts/TRACKER_project/171205_Stack-fitted/'
IS_SPLITED = True
EXP_PARAM = 'experiment-log' # String 'imageTimes', 'experiment-log' or 'None'
IS_CCC = True # MATLAB LOG VERSION (need to convert in csv)

## Time and Feed factor (if no exp_param file)
T_MIN, T_MAX = 24, 96
IS_FEED = False

## Columns of CSV  (see 1st line of csv)
CONDITION = np.array([None])
CELL_INFO = np.array(['frame', 'x', 'y', 'I', 'w', 'L'])

## Additional columns (if is splitted : no info in log)
NB_PLICAT = 2
NEW_COND = {'C0' : ['K', 'AFK', '20S', '40S', '80S', '20P', '40P', '80P', '20-5P', '20-10P', '20-20P', '20-40P']} #180726
"""
{'C0' : ['K', 'AFK', '20S', '40S', '80S', '20P', '40P', '80P', '20-5P', '20-10P', '20-20P', '20-40P']} #180726
{'C0' : [20, 10, 5, 3.3, 1.5, 0.03, 0.1, 0.33, 1., 3., 5.5, 6.5]} # 2018-07-27
"""
# Figure
W, H, L, S = 3.7, 2.9, 18., 9. # width, height, label_size, scale_size
VAL_ = [0., 1.1]
T = [0, 120]

# Stat-param
OUT_QUANTILE = 0.995

## Plotting classification
'''Cond savefig sep -> curve per condition -> X_axis grouping -> Y_axis to see'''
# Indicate index (see Condition and cell info order)
IDX_SAVE_CURVE = [0, 1]
IDX_AXIS = [0,3]
IDX_LABEL = [1]

## Info variable output
NAME_VAR = 'ASE-YFP'
VAR_ = '(norm)'

###############################################################################
#################################### INITIALISATION :
# Add condition
if IS_SPLITED :
    CONDITION = np.concatenate((CONDITION, list(NEW_COND.keys())))
# Convert scale figure
MM2INCH = 2.54
W, H, L, S = np.array((W, H, L, S))/MM2INCH # ratio fig : 2.7/2.1
# Level
LVL_ORDER = np.concatenate((CONDITION[IDX_SAVE_CURVE], CELL_INFO[IDX_AXIS]))
# Labeling
LABEL = CONDITION[IDX_LABEL]
# Import csv
if IS_SPLITED :
    # Path listing
    List_csv_path = [PATH + f for f in sorted(os.listdir(PATH)) if (".csv" in f[-4:] and not(EXP_PARAM) in f)]
    # Adding new condition
    DATA, n = [], 0
    for i in range(len(List_csv_path)) :
        data = pd.read_csv(List_csv_path[i])
        # Condition list
        C = []
        for c in NEW_COND :
            data[c] = data.shape[0] * [NEW_COND[c][n]]
        # Change index condition
        if i%NB_PLICAT == 0 and i > 0 : n+=1
        # Complete data incrementation
        DATA += [data]
    DATA = pd.concat(DATA)
else :
    List_csv_path = [PATH + f for f in sorted(os.listdir(PATH)) if (".csv" in f[-4:] and not(EXP_PARAM) in f[-4:])]
    DATA = pd.read_csv(List_csv_path[0])

# Extract name experiment
CURVE_FOLDER = PATH.split(os.path.sep)[-2]
# Create the folder for analysis
if(not os.path.isdir(CURVE_FOLDER)): os.makedirs(CURVE_FOLDER)
# Parameter verification and Affect name str
idx2test = np.where(LVL_ORDER != None)[0]
if False in [n in DATA.columns for n in LVL_ORDER[idx2test]] : sys.exit("Error parameter")

#################################### TIME-FEED SET :
if 'None' in EXP_PARAM :
    # Dimensional order
    frame_ = np.unique(DATA.frame)
    t = T_MIN + (T_MAX-T_MIN) * frame_ / (frame_.size - 1)
elif IS_CCC :
    # With Log CCC (feed info)
    csv_exp_param = [PATH + f for f in os.listdir(PATH) if EXP_PARAM in f]
    EXP_INFO = pd.read_csv(csv_exp_param[0])
    T_MIN, T_MAX = EXP_INFO['t'].min(), EXP_INFO['t'].max()
    frame_ = np.unique(DATA.frame)
    t = T_MIN + (T_MAX-T_MIN) * frame_ / (frame_.size - 1)
else :
    # Manual version
    csv_exp_param = [PATH + f for f in os.listdir(PATH) if EXP_PARAM in f]
    EXP_INFO = pd.read_csv(csv_exp_param, delimiter=';')
    t = EXP_INFO['image times'].values
    feed = EXP_INFO['feed times'].values
    feed = feed[np.invert(np.isnan(feed))]

#################################### DATASET STAT :
## Preprocessing of data distribution (Outlier/aberration, Normalisation)
outlier = DATA[LVL_ORDER[-1]].quantile(OUT_QUANTILE) # 99,5% of statistical sample follow 'I'
DATA = DATA[DATA[LVL_ORDER[-1]] < outlier]

min_max_scaler = preprocessing.MinMaxScaler()
DATA[LVL_ORDER[-1]] = min_max_scaler.fit_transform(DATA[[LVL_ORDER[-1]]].values)
plt.hist(DATA[LVL_ORDER[-1]].values, 100); plt.show(); plt.close() # Poisson distrib, peak at ~0.1 (no signal)

#################################### HIERARCHY PLOT :
# Verify if LVL it's for one files output
if LVL_ORDER[0] == None :
    DATA_LOOP = [(CURVE_FOLDER,DATA)]
else :
    DATA_LOOP = DATA.groupby([LVL_ORDER[0]])
# Loop data
for DL in DATA_LOOP :
    label, df = DL[0], DL[1]
    
    ## GROUPBY LVL1-2 (curve and frame)
    gf_lvl = df.groupby([LVL_ORDER[1],LVL_ORDER[2]])    
    # Extract mean of dataframe following lvl
    curve = gf_lvl.mean()[LVL_ORDER[3]].unstack()
    fill = gf_lvl.std()[LVL_ORDER[3]].unstack().values
    
    # Real time calculate
    if IS_CCC :
        time = (np.ones(fill.shape)*t)
    else :
        time = (np.ones(fill.shape)*t)
    
    ## Type of VAR output
    if VAR_ == '(norm)' :
        # second Normalisation (mean norm)
        min_val, max_val = curve.min().min(), curve.max().max() + fill.max().max()/2
        curve = (curve - min_val)/(max_val-min_val)
    
    ## Curve labelling
    lbl_list = curve.index.values.astype('str')
    
    # Figure
    fig = plt.figure(figsize=(W, H))
    
    plt.rc('font', size=S)
    plt.rc('axes', titlesize=S)
    
    ax = fig.add_subplot()
    ax.set_title(CURVE_FOLDER + ' ' + label, fontsize=L)
    ax.set_ylabel(NAME_VAR + ' ' + VAR_, fontsize=L)
    ax.set_xlabel('time (h)', fontsize=L)
    #curve.T.plot(ax=ax) # Automatic legends, otherwise : #name = curve.index.values
    i = 0
    for stack in np.dstack((curve.values, fill, time)) :
        name = lbl_list[i]; i+=1
        z = stack.T
        ax.plot(z[-1], z[0], label=name)
        ax.fill_between(z[-1], z[0] - z[1]/2, z[0] + z[1]/2, alpha=0.3)
    
    # feed bar :
    if IS_FEED :
        for f in feed :
            ax.plot([f,f],[curve.values.min(),curve.values.max()], 'k-.')
    # Legend
    ax.legend()
    plt.xlim(T), plt.ylim(VAL_)
    # Save data
    plt.savefig(CURVE_FOLDER + os.path.sep + label + '_' + VAR_ + ".svg")
    plt.close()
