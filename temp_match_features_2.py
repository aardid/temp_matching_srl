"""
ENCN493 Test Multiprocessing Code
Author: Owen Garrett and Josh Corry
Date: 20/09/23
Code runs through entire Whakaari dataset and returns all p-values required using multiprocessing.
Running on the puia environment as described in the git repository
"""
from datetime import datetime, timedelta
from puia.features import FeaturesSta
from puia.utilities import datetimeify
from glob import glob
from sys import platform
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from functools import partial
import scipy.stats as scp
import time
import traceback
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

_DAY=timedelta(days=1)
_MONTH=timedelta(days=365.25/12)
# set path depending on OS
if platform == "linux" or platform == "linux2":
    root=r'/media/eruption_forecasting/eruptions'
elif platform == "win32":
    root=r'U:'+os.sep+'Research'+os.sep+'EruptionForecasting'+os.sep+'eruptions'
    #root=r'D:\Uni\ENCI493 Project\puia_for_students'       
    #root=r'U:\Research\EruptionForecasting\eruptions'
DATA_DIR=f'{root}/data'#f'/Users/Owen/Documents/Puia/puia-main/data'
FEAT_DIR=f'{root}/features'#f'/Users/Owen/Documents/Puia/puia-main/features'
MODEL_DIR=f'{root}/models'
FORECAST_DIR=f'{root}/forecasts'

# constants
year = timedelta(days=365)
month = timedelta(days=365.25/12)
day = timedelta(days=1)
hour = timedelta(days=1/24)
textsize = 12.
N, M = [2,30]
####################################################################################
_STA = 'COP'

if _STA == 'WIZ':
    _ti_rec = '2011-01-01'
    _tf_rec = '2019-12-31'
if _STA == 'COP':
    _ti_rec = '2020-03-09'
    _tf_rec = '2022-11-19'
if _STA == 'BELO':
    _ti_rec = '2007-08-22'
    _tf_rec = '2010-07-10'
if _STA == 'PVV':
    _ti_rec = '2007-03-03'
    _tf_rec = '2021-12-30'
station_names_dict = {'COP': 'Copahue', 'PVV': 'Pavlof', 'BELO': 'Bezymianny', 'WIZ': 'Whakaari'}
####################################################################################
# Auxiliary functions
def write_features_names():
    '''
    '''
    os.makedirs('.'+os.sep+'conv_output'+os.sep+_STA, exist_ok=True)
    # Creates a list of all feature names
    feat_sta=FeaturesSta(station=_STA, window=2.0, datastream='zsc2_dsarF', feat_dir=FEAT_DIR, 
        ti=_ti_rec, tf=_tf_rec, tes_dir=DATA_DIR, dt=None, lab_lb=2.)
    # names of features (columns of feat_sta.fM)
    columns_dsar = feat_sta.fM.columns
    del feat_sta
    # Write the list of strings to a file
    with open('.'+os.sep+'conv_output'+os.sep+_STA+os.sep+'columns_dsar.txt', 'w') as file:
        for item in columns_dsar:
            file.write("%s\n" % item)
    feat_sta=FeaturesSta(station=_STA, window=2.0, datastream='zsc2_rsamF', feat_dir=FEAT_DIR, 
        ti=_ti_rec, tf=_tf_rec, tes_dir=DATA_DIR, dt=None, lab_lb=2.)
    # names of features (columns of feat_sta.fM)
    columns_rsam = feat_sta.fM.columns
    del feat_sta
    # Write the list of strings to a file
    with open('.'+os.sep+'conv_output'+os.sep+_STA+os.sep+'columns_rsam.txt', 'w') as file:
        for item in columns_rsam:
            file.write("%s\n" % item)
    feat_sta=FeaturesSta(station=_STA, window=2.0, datastream='zsc2_mfF', feat_dir=FEAT_DIR, 
        ti=_ti_rec, tf=_tf_rec, tes_dir=DATA_DIR, dt=None, lab_lb=2.)
    # names of features (columns of feat_sta.fM)
    columns_mf = feat_sta.fM.columns
    del feat_sta
    # Write the list of strings to a file
    with open('.'+os.sep+'conv_output'+os.sep+_STA+os.sep+'columns_mf.txt', 'w') as file:
        for item in columns_mf:
            file.write("%s\n" % item)
    feat_sta=FeaturesSta(station=_STA, window=2.0, datastream='zsc2_hfF', feat_dir=FEAT_DIR, 
        ti=_ti_rec, tf=_tf_rec, tes_dir=DATA_DIR, dt=None, lab_lb=2.)
    # names of features (columns of feat_sta.fM)
    columns_hf = feat_sta.fM.columns
    del feat_sta
    # Write the list of strings to a file
    with open('.'+os.sep+'conv_output'+os.sep+_STA+os.sep+'columns_hf.txt', 'w') as file:
        for item in columns_hf:
            file.write("%s\n" % item)
def read_features_names():
    '''
    '''
    # Read the list of strings from the file
    with open('.'+os.sep+'conv_output'+os.sep+_STA+os.sep+'columns_rsam.txt', 'r') as file:
        #columns_rsam = [line.strip() for line in file]
        columns_rsam = [line.strip() for line in file if 'linear_trend' not in line]
    with open('.'+os.sep+'conv_output'+os.sep+_STA+os.sep+'columns_dsar.txt', 'r') as file:
        #columns_dsar = [line.strip() for line in file]
        columns_dsar = [line.strip() for line in file if 'linear_trend' not in line]
    with open('.'+os.sep+'conv_output'+os.sep+_STA+os.sep+'columns_mf.txt', 'r') as file:
        #columns_rsam = [line.strip() for line in file]
        columns_mf = [line.strip() for line in file if 'linear_trend' not in line]
    with open('.'+os.sep+'conv_output'+os.sep+_STA+os.sep+'columns_hf.txt', 'r') as file:
        #columns_dsar = [line.strip() for line in file]
        columns_hf = [line.strip() for line in file if 'linear_trend' not in line]
    return columns_rsam, columns_dsar, columns_mf, columns_hf
def read_feat_write(columns_rsam,columns_dsar,columns_mf,columns_hf):
    '''
    '''
    # 
    output_directory = 'conv_output'+os.sep+_STA+os.sep+'individual_columns'
    os.makedirs(output_directory, exist_ok=True)
    output_directory = output_directory+os.sep
    ti_rec = _ti_rec#'2011-01-01'
    tf_rec = _tf_rec#'2019-12-31'
    # rsam
    feat_sta=FeaturesSta(station=_STA, window=2.0, datastream='zsc2_rsamF', feat_dir=FEAT_DIR, 
        ti=ti_rec, tf=tf_rec, tes_dir=DATA_DIR, dt=None, lab_lb=2.)
    for i,col in enumerate(columns_rsam): # feat_sta.fM.columns
        print(col)
        print(str(i+1)+'/'+str(len(columns_rsam)))
        col_df = feat_sta.fM[[col]]
        # Save the column as an individual CSV file
        output_file = f'{output_directory}{col}.csv'
        col_df.to_csv(output_file)
    # dsar
    feat_sta=FeaturesSta(station=_STA, window=2.0, datastream='zsc2_dsarF', feat_dir=FEAT_DIR, 
        ti=ti_rec, tf=tf_rec, tes_dir=DATA_DIR, dt=None, lab_lb=2.)
    for col in columns_dsar: # feat_sta.fM.columns
        print(col)
        col_df = feat_sta.fM[[col]]
        # Save the column as an individual CSV file
        output_file = f'{output_directory}{col}.csv'
        col_df.to_csv(output_file)
    # mf
    feat_sta=FeaturesSta(station=_STA, window=2.0, datastream='zsc2_mfF', feat_dir=FEAT_DIR, 
        ti=ti_rec, tf=tf_rec, tes_dir=DATA_DIR, dt=None, lab_lb=2.)
    for i,col in enumerate(columns_mf): # feat_sta.fM.columns
        print(col)
        print(str(i+1)+'/'+str(len(columns_mf)))
        col_df = feat_sta.fM[[col]]
        # Save the column as an individual CSV file
        output_file = f'{output_directory}{col}.csv'
        col_df.to_csv(output_file)
    # hf
    feat_sta=FeaturesSta(station=_STA, window=2.0, datastream='zsc2_hfF', feat_dir=FEAT_DIR, 
        ti=ti_rec, tf=tf_rec, tes_dir=DATA_DIR, dt=None, lab_lb=2.)
    for col in columns_hf: # feat_sta.fM.columns
        print(col)
        col_df = feat_sta.fM[[col]]
        # Save the column as an individual CSV file
        output_file = f'{output_directory}{col}.csv'
        col_df.to_csv(output_file)
def conv(at, x):
    '''
    '''
    #y = ((x-np.mean(x))/np.std(x)*at.values).mean()
    xi=x.values
    y=np.corrcoef(xi,at.values[:,0])[1,0]
    return y
def datetimeify(t):
    """ Return datetime object corresponding to input string.

        Parameters:
        -----------
        t : str, datetime.datetime
            Date string to convert to datetime object.

        Returns:
        --------
        datetime : datetime.datetime
            Datetime object corresponding to input string.

        Notes:
        ------
        This function tries several datetime string formats, and raises a ValueError if none work.
    """
    try:
        if '+' in t:
            t = t.split('+')[0]
    except:
        pass
    from pandas._libs.tslibs.timestamps import Timestamp
    if type(t) in [datetime, Timestamp]:
        return t
    fmts = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y %m %d %H %M %S','%d/%m/%Y %H:%M', '%Y%m%d:%H%M']
    for fmt in fmts:
        try:
            return datetime.strptime(t, fmt)
        except ValueError:
            pass
    raise ValueError("time data '{:s}' not a recognized format".format(t))
####################################################################################
# Convolution functions
def run_single_conv(pars):
    '''
    '''
    sta, te, d, feat, lb, tes = pars
    output_directory = 'conv_output'+os.sep+_STA+os.sep+'individual_kst'
    os.makedirs(output_directory, exist_ok=True)
    # Read the CSV file, only selecting the 'feature_name' column
    try:
        file_path = '.'+os.sep+'conv_output'+os.sep+_STA+os.sep+'individual_columns'+os.sep+feat+'.csv'
        _rec = pd.read_csv(file_path, usecols=['time',feat], index_col='time', parse_dates=True)
        if True: # downsample 
            DS='1H'
            _rec = _rec.resample('1H').mean()#_rec.resample('1H').mean()
            if DS=='1D':
                tes = [datetimeify(_te.strftime('%Y-%m-%d')) for _te in tes]
                te = datetimeify(te.strftime('%Y-%m-%d')) 
            if DS=='1H':
                tes = [datetimeify(_te.strftime('%Y-%m-%d %H:'+'00:00')) for _te in tes]
                te = datetimeify(te.strftime('%Y-%m-%d %H:'+'00:00')) 
        tes.remove(te)
        _temp = _rec[te-lb*_DAY:te]
        # Convolution and KS Test
        # out = _rec.rolling(_temp.shape[0]).apply(partial(conv, (_temp-_temp.mean())/_temp.std()))
        #out = _rec.rolling(_temp.shape[0]).apply(lambda x: np.corrcoef(x.values, _temp.values[:,0])[1,0])
        out = _rec.rolling(_temp.shape[0]).apply(lambda x: np.corrcoef(x, _temp.values[:,0])[1,0])
        eruption_cc = out.loc[tes]
        out = out.dropna(subset=[feat])
        out.to_csv('test.csv')
        statistic, p_value = scp.ks_2samp(out[feat].dropna().values, eruption_cc[feat].dropna().values)
        with open('.'+os.sep+output_directory+os.sep+f"{sta}_{lb}"+'w_'+f"{te.year}-{te.month}-{te.day}"+'_'+feat+'.csv', 'w') as file:
            # Write the variable to the file
            file.write("statistic,p_value\n")
            file.write(f"{statistic},{p_value}") 
    except:
        os.makedirs('.'+os.sep+output_directory+os.sep+'errors', exist_ok=True)
        with open('.'+os.sep+output_directory+os.sep+'errors'+os.sep+f"{sta}_{lb}"+'w_'+f"{te.year}-{te.month}-{te.day}"+'_'+feat+'.err','w') as fp:
            fp.write(f'{traceback.format_exc()}\n')
    # 
def run_paralell(sta, tes, ds, lbs):
    '''
    '''
    ncpus = 60#cpu_count()
    jobs = []
    columns_file = {'dsar': 'columns_dsar.txt','rsam': 'columns_rsam.txt', 'mf': 'columns_mf.txt','hf': 'columns_hf.txt'}
    for d in ds:
        for lb in lbs:
            for te in tes:
                d_type = next((key for key in columns_file if key in d), None)
                if d_type:
                    with open('.' + os.sep + 'conv_output'+os.sep+_STA + os.sep + columns_file[d_type], 'r') as file:
                        columns = [line.strip() for line in file]
                    for feat in columns:
                        if os.path.exists('.'+os.sep+'conv_output'+os.sep+_STA+os.sep+'individual_kst'+os.sep+f"{sta}_{lb}"+'w_'+f"{te.year}-{te.month}-{te.day}"+'_'+feat+'.csv'): 
                            pass
                        else:
                            jobs.append([sta, te, d, feat, lb, tes]) 
    with open('.'+os.sep+'conv_output'+os.sep+_STA+os.sep+'run_info.txt', 'w') as file:
        file.write(f"sta = '{sta}'\n")
        file.write(f"tes = {tes}\n")
        file.write(f"ds = {ds}\n")
        file.write(f"lbs = {lbs}\n")
        file.write(f"cores = {ncpus}\n") 
        file.write(f"jobs = {len(jobs)}\n")                   
    print('Number of jobs: '+str(len(jobs)))
    # Run the parallel processing
    print('Number of cores: '+str(ncpus))
    run_single_conv(jobs[0])
    with Pool(processes=ncpus) as pool:
        results = list(tqdm(pool.imap(run_single_conv, jobs), total=len(jobs)))
def merge_results(sta, tes, ds, lbs):
    '''
    '''
    # Get a list of all CSV files in the directory
    fls = glob('conv_output'+os.sep+_STA+os.sep+'individual_kst'+os.sep+'*.csv')
    # Initialize an empty DataFrame
    result_df = pd.DataFrame()
    # Loop over the files
    for file_path in fls:
        # Extract the filename without the extension
        file_name = file_path.split(os.sep)[-1].replace('.csv', '')
        # Read the CSV file
        df = pd.read_csv(file_path)
        # Extract the statistics
        statistic = df.iloc[0, 0]  # Assuming statistic is in the first row and first column
        p_value = df.iloc[0, 1]    # Assuming p_value is in the first row and second column
        # Add a row to the result DataFrame
        result_df.loc[file_name, 'statistic'] = statistic
        result_df.loc[file_name, 'p_value'] = p_value
    result_df.index.name = 'feature'
    # Print the resulting DataFrame
    result_df.to_csv('conv_output'+os.sep+_STA+os.sep+'result_statistics.csv')
def check_results():
    '''
    '''
    # Define file path
    file_path = 'conv_output'+os.sep+_STA + os.sep + 'result_statistics.csv'
    # Step 2: Read the CSV file
    df = pd.read_csv(file_path)

    # Step 3: Set 'feature' column as the index
    df.set_index('feature', inplace=True)

    # Step 4: Find the top 10 maximum rows for both 'statistic' and 'p_value'
    top_10_statistic = df.nlargest(10, 'statistic')
    top_10_p_value = df.nsmallest(10, 'p_value')  # Changed to nsmallest

    # Step 5: Plot the distributions of 'statistic' and 'p_value'
    plt.figure(figsize=(7, 2.5))

    # Distribution of 'statistic'
    plt.subplot(1, 2, 1)
    plt.hist(df['statistic'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Statistic')
    plt.xlabel('Statistic')
    plt.ylabel('Frequency')

    # Distribution of 'p_value' in log scale
    plt.subplot(1, 2, 2)
    log_p_values = np.log10(df['p_value'])  # Apply log transformation
    plt.hist(log_p_values, bins=20, color='salmon', edgecolor='black')
    plt.title('Distribution of log(p-value)')
    plt.xlabel('log(p-value)')
    plt.ylabel('Frequency')

    plt.tight_layout()

    # Save the plot as a PNG file
    plot_file_path = 'conv_output'+os.sep+_STA + os.sep +'plot_statistics.png'
    plt.savefig(plot_file_path)

    # Open the file in write mode
    with open('conv_output'+os.sep+_STA + os.sep + 'statistics.txt', 'w') as file:
        # Write the top 10 maximum rows for 'statistic'
        file.write("Top 10 maximum rows for 'statistic':\n")
        file.write(top_10_statistic.to_string())

        # Write a separator
        file.write("\n" + "="*50 + "\n")

        # Write the top 10 lowest rows for 'p_value'
        file.write("\nTop 10 lowest rows for 'p_value':\n")
        file.write(top_10_p_value.to_string())

        # Write a separator
        file.write("\n" + "="*50 + "\n")

        # Write the distributions to the file
        file.write("Distribution of Statistic:\n")
        file.write(str(df['statistic'].describe()))

        file.write("\n" + "="*50 + "\n")

        file.write("Distribution of log(p-value):\n")
        log_p_values.describe().to_csv(file, header=True, index=True)
####################################################################################
# time series and ROC curves
def run_threshold_exceedance(sta,tes):
    pass
    # Define file path
    file_path = 'conv_output'+os.sep+_STA + os.sep + 'result_statistics.csv'
    # Step 2: Read the CSV file
    df = pd.read_csv(file_path)
    # Step 3: Set 'feature' column as the index and sort
    df.set_index('feature', inplace=True)
    df.sort_values(by=['p_value'], inplace=True)
    # compute BM false rate discovery 
    if True: #  Benjamini-Hochberg 
        from statsmodels.stats.multitest import multipletests
        # Compute Benjamini-Hochberg false discovery rate
        adjusted_p_values = multipletests(df['p_value'], method='fdr_bh')[1]

        # Add adjusted p-values to DataFrame
        df['adjusted_p_value'] = adjusted_p_values

        # Sort DataFrame by adjusted p-values
        df.sort_values(by=['adjusted_p_value'], inplace=True)

        #
        quantile_5percent = df['adjusted_p_value'].quantile(0.05)

    #
    output_directory = 'conv_output'+os.sep+_STA+os.sep+'roc_auc_thresholds_feat_cc'
    os.makedirs(output_directory, exist_ok=True)
    #for window in lbs:
    filter_feat = False
    #
    with open(output_directory+os.sep+'_00_roc_auc_all.csv', 'w') as auc:
        auc.write('feature,auc\n')
        _DS='1H'
        for j, row in enumerate(df.index[:2000]):
            print(str(j)+'/'+str(len(df.index[:2000])))
            print(row)
            try:
                _feat='zsc2'+row.split('zsc2')[1]
                _w=int(row.split('_')[1][:-1])
                _te=datetimeify(row.split('_')[2])
                for te in tes:
                    if (_te.year == _te.year and _te.month == _te.month and _te.day == _te.day):
                        _te = te
                _d=row.split('_')[3]+'_'+row.split('_')[4]
                #
                if os.path.exists(output_directory+os.sep+str(j)+'_roc_'+sta+'_'+str(_w)+'_'+str(_te.year)+'-'+str(_te.month)+'-'+str(_te.day)+'_'+_feat+'.csv'):
                    pass
                    print('Exist')
                else:
                    def single_conv(pars, DS='1H'):
                        '''
                        '''
                        sta, te, d, feat, lb, tes = pars
                        file_path = '.'+os.sep+'conv_output'+os.sep+_STA+os.sep+'individual_columns'+os.sep+feat+'.csv'
                        _rec = pd.read_csv(file_path, usecols=['time',feat], index_col='time', parse_dates=True)
                        if DS: # downsample 
                            #DS='1H'
                            _rec = _rec.resample('1H').mean()#_rec.resample('1H').mean()
                            if DS=='1D':
                                tes = [datetimeify(_te.strftime('%Y-%m-%d')) for _te in tes]
                                te = datetimeify(te.strftime('%Y-%m-%d')) 
                            if DS=='1H':
                                tes = [datetimeify(_te.strftime('%Y-%m-%d %H:'+'00:00')) for _te in tes]
                                te = datetimeify(te.strftime('%Y-%m-%d %H:'+'00:00')) 
                        tes.remove(te)
                        _temp = _rec[te-lb*_DAY:te]
                        # Convolution and KS Test
                        # out = _rec.rolling(_temp.shape[0]).apply(partial(conv, (_temp-_temp.mean())/_temp.std()))
                        #out = _rec.rolling(_temp.shape[0]).apply(lambda x: np.corrcoef(x.values, _temp.values[:,0])[1,0])
                        out = _rec.rolling(_temp.shape[0]).apply(lambda x: np.corrcoef(x, _temp.values[:,0])[1,0])
                        eruption_cc = out.loc[tes]
                        out = out.dropna(subset=[feat])
                        return out
                        # 
                    fm=single_conv([sta, _te, _d, _feat, _w, tes], DS=_DS)
                    fm['data'] = fm[_feat].values
                    fm['data'] = (fm['data'].values + 1.) / 2.

                    # define thresholds
                    _dt = 1
                    _n = np.arange(0,100+_dt,_dt)/100
                    #
                    if False: # remove eruptive periods
                        _fm=fm.copy()
                        for te in tes:
                            inds = (_fm.index<te-day*21)|(_fm.index>te+day*21) 
                            _fm=_fm.loc[inds]
                    #
                    thresholds=[fm['data'].quantile(q=n) for n in _n]
                    _thresholds = np.arange(0,len(thresholds),1)
                    thresholds=_thresholds/100
                    #del fm
                    for i, th in enumerate(thresholds):
                        fm['th_'+str(i)] = np.where(fm['data'] > th,1,0)
                    # save
                    fm.to_csv(output_directory+os.sep+'_thresholds_'+_feat+'.csv')
                    #
                    l_tp, l_fn, l_fp, l_tn,l_fpr,l_tpr,= [],[],[],[],[],[],
                    for i, th in enumerate(thresholds[:]):
                        #print(str(i)+'/'+str(len(thresholds)))
                        c_tp, c_fn, c_tn, c_fp=0,0,0,0
                        #
                        for k, te in enumerate(tes):
                            # count TP and FN in eruptive record (and add to total)
                            inds = (fm['th_'+str(i)].index<te-_w*day)|(fm.index>=te)
                            _max = fm['th_'+str(i)].loc[~inds].max()
                            if not _max:
                                _max = 0.
                            if _max==1: 
                                #c_tp+=144*_w#288 
                                if _DS=='1H':
                                    c_tp+=24*_w#288 
                                if not _DS:
                                    c_tp+=144*_w
                            else: 
                                #c_fn+=144*_w#  
                                if _DS=='1H':    
                                    c_fn+=24*_w#
                                if not _DS:
                                    c_fn+=144*_w
                            # Remove event points
                            #fm=fm.loc[inds]
                            # Remove a week after the event
                            #inds = (fm.index>te-hour/6)|(fm.index<te+day*7)
                            #fm=fm.loc[inds]
                            pass
                        # count TN and FP in non-eruptive record (and add to total)
                        _idx_bool = fm['th_'+str(i)]<1
                        c_tn += len(fm[_idx_bool])
                        c_fp += len(fm[~_idx_bool])
                        # append to lists
                        l_tp.append(c_tp)
                        l_fn.append(c_fn)
                        l_fp.append(c_fp)
                        l_tn.append(c_tn)
                        #
                        tpr = c_tp/(c_tp+c_fn)
                        fpr = c_fp/(c_fp+c_tn)
                        l_fpr.append(fpr) 
                        l_tpr.append(tpr)
                    #
                    #_thresholds = np.arange(0,len(thresholds)+1,1)
                    with open(output_directory+os.sep+str(j)+'_roc_'+sta+'_'+str(_w)+'_'+str(_te.year)+'-'+str(_te.month)+'-'+str(_te.day)+'_'+_feat+'.csv', 'w') as f:
                        f.write('th,tp,fn,fp,tn,fpr,tpr\n')
                        for th,tp,fn,fp,tn,fpr,tpr in zip(thresholds,l_tp,l_fn,l_fp,l_tn,l_fpr,l_tpr):
                            f.write(str(th)+','+str(tp)+','+str(fn)+','+str(fp)+','+str(tn)+','+str(fpr)+','+str(tpr)+'\n')
                    ##
                # plot roc
                if True:
                    fm = pd.read_csv(output_directory+os.sep+str(j)+'_roc_'+sta+'_'+str(_w)+'_'+str(_te.year)+'-'+str(_te.month)+'-'+str(_te.day)+'_'+_feat+'.csv')#, index_col='time')
                    # AUC
                    _auc=[]
                    for i in range(len(fm['fpr'])-1):
                        _dx, _dy = fm['fpr'][i]-fm['fpr'][i+1], fm['tpr'][i]
                        _auc.append([_dx*_dy])
                    _auc = np.sum(np.asarray(_auc))
                    if True:
                        plt.plot([],[], label = 'AUC '+str(round(_auc,2)))#,        
                        plt.plot(fm['fpr'], fm['tpr'])
                        plt.legend()
                        plt.xlim([0,1.1])
                        plt.ylim([0,1.1])
                        plt.savefig(output_directory+os.sep+str(j)+'_roc_'+sta+'_'+str(_w)+'_'+str(_te.year)+'-'+str(_te.month)+'-'+str(_te.day)+'_'+_feat+'.png')
                        plt.close('all')
                    ####
                    #auc.write(sta+'_'+str(_w)+'_'+str(_te.year)+'-'+str(_te.month)+'-'+str(_te.day)+'_'+_feat.replace(",", "_"))
                    auc.write(row.replace(",", "_"))
                    #auc.write(row)
                    auc.write(',')
                    auc.write(str(round(_auc,2))+'\n')
                    ##
            except:
                print('failed')
    # import and sort
    if filter_feat:
        file_path = output_directory+os.sep+'_00_roc_auc_all.csv'
    else:
        file_path = output_directory+os.sep+'_00_roc_auc_all.csv'
    df = pd.read_csv(file_path)
    # Step 3: Set 'feature' column as the index and sort
    df.set_index('feature', inplace=True)
    df.sort_values(by=['auc'])
    df.to_csv(output_directory+os.sep+'_00_roc_auc_all.csv')
def filter_feat_families(sta,tes):
    pass
    # Define file path
    file_path = 'conv_output'+os.sep+_STA + os.sep + 'result_statistics.csv'
    # Step 2: Read the CSV file
    df = pd.read_csv(file_path)
    # Step 3: Set 'feature' column as the index and sort
    df.set_index('feature', inplace=True)
    df.sort_values(by=['p_value'], inplace=True)
    #
    # Define a function to check similarity
    def is_similar(name1, name2):
        a = name1.split('_')[6:9] 
        b = name2.split('_')[6:9]
        return name1.split('_')[6:9] == name2.split('_')[6:9] #and name1.split('_')[:6] != name2.split('_')[:6]# == name2.split('_')[3:5]#and name1.split('_')[:6] != name2.split('_')[:6]

    # Create an empty DataFrame to store filtered results
    df_filtered = pd.DataFrame(columns=df.columns)

    # Iterate over unique index names and add the top row for each group
    for i, name1 in enumerate(df.index.unique()):
        add_row = True
        for name2 in df_filtered.index:
            if is_similar(name1, name2):
                add_row = False
                break
        if add_row:
            group = df.loc[df.index == name1]
            df_filtered = pd.concat([df_filtered, group.head(1)])
    
    output_directory = 'conv_output'+os.sep+_STA+os.sep+'roc_auc_thresholds_feat_cc'
    os.makedirs(output_directory, exist_ok=True)
    #for window in lbs:
    filter_feat = True
    #
    with open(output_directory+os.sep+'00_roc_auc_all_filter.csv', 'w') as auc:
        auc.write('feature,auc\n')
        _DS='1H'
        for j, row in enumerate(df.index):
            try:
                if row in df_filtered.index:
                    print(row)
                    #try:
                    _feat='zsc2'+row.split('zsc2')[1]
                    #_w=float(row.split('_')[1][0])
                    _w=int(row.split('_')[1][:-1])
                    _te=datetimeify(row.split('_')[2])
                    for te in tes:
                        if (_te.year == _te.year and _te.month == _te.month and _te.day == _te.day):
                            _te = te
                    _d=row.split('_')[3]+'_'+row.split('_')[4]
                    ##
                    # auc
                    if True:
                        fm = pd.read_csv(output_directory+os.sep+str(j)+'_roc_'+sta+'_'+str(_w)+'_'+str(_te.year)+'-'+str(_te.month)+'-'+str(_te.day)+'_'+_feat+'.csv')#, index_col='time')
                        if True:
                            # AUC
                            _auc=[]
                            for i in range(len(fm['fpr'])-1):
                                _dx, _dy = fm['fpr'][i]-fm['fpr'][i+1], fm['tpr'][i]
                                _auc.append([_dx*_dy])
                            _auc = np.sum(np.asarray(_auc))
                        ####
                        #auc.write(sta+'_'+str(_w)+'_'+str(_te.year)+'-'+str(_te.month)+'-'+str(_te.day)+'_'+_feat.replace(",", "_"))
                        if ',' in row:
                            pass
                        else:
                            auc.write(row)
                            auc.write(',')
                            auc.write(str(round(_auc,2))+'\n')
                        #except:
                    #    print('failed')
                else:
                    pass
            except:
                pass
    # import and sort
    file_path = output_directory+os.sep+'00_roc_auc_all_filter.csv'
    df_filtered = pd.read_csv(file_path)
    # Step 3: Set 'feature' column as the index and sort
    df_filtered.set_index('feature', inplace=True)
    df_filtered.sort_values(by=['auc'])
    df_filtered.to_csv(output_directory+os.sep+'00_roc_auc_all_filter.csv')
####################################################################################
def hist_datastreams():
    '''
    '''
    pass
    # List of strings to search for in the 'feature' column
    search_strings = ['rsam', 'mf', 'hf', 'dsar']

    # Corresponding labels for the legend
    legend_labels = {'rsam': '[2-5 Hz]', 'mf': '[4.5-8 Hz]', 'hf': '[8-16 Hz]', 'dsar': '~mf/hf'}

    # Initialize a dictionary to store counts for each feature
    feature_counts = {string: [] for string in search_strings}

    # Loop over 'stas'
    station_names = ['COP', 'PVV', 'BELO', 'WIZ',]

    # Loop over 'stas'
    for sta in station_names:  # Add your actual station names here
        # Construct file path
        _file = 'conv_output' + os.sep + sta + os.sep + 'roc_auc_thresholds_feat_cc' + os.sep + '00_roc_auc_all.csv'
        
        # Read the CSV file into a DataFrame
        _df = pd.read_csv(_file)
        
        # Initialize counts for the current station
        station_counts = {string: 0 for string in search_strings}
        
        # Count occurrences for each search string in every row
        for index, row in _df.iterrows():
            for search_string in search_strings:
                if search_string in row['feature']:
                    station_counts[search_string] += 1
        
        # Append counts to the dictionary
        for search_string in search_strings:
            feature_counts[search_string].append(station_counts[search_string])

    # Convert the dictionary to a DataFrame
    counts_df = pd.DataFrame(feature_counts,  index=station_names)
    counts_df.loc['PVV'] *= 4

    # Plotting
    counts_df.plot(kind='bar', stacked=False, figsize=(7, 2.2))
    plt.title('Occurrence of data streams in diferent volcanoes')
    #plt.xlabel('Volcano')
    plt.ylabel('Count')

    # Update legend labels
    legend_labels = [f'{label} {legend_labels[label]}' for label in search_strings]
    plt.legend(title='Datastream', labels=legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left')

    #plt.xticks(rotation=0, ha='right')
    # Update x-axis labels
    plt.xticks(ticks=range(len(station_names)), labels=[station_names_dict[sta] for sta in station_names], rotation=0)#, ha='right', va='center')    
    
    plt.tight_layout()
    plt.show()
def hist_eruptions():
    '''
    '''
    # List of station names
    station_names = ['COP', 'PVV', 'BELO', 'WIZ']#['COP', 'PVV', 'WIZ', 'BELO']

    # Dictionary to store tes lists for each station
    tes_dict = {}

    # Loop over station_names
    for sta in station_names:
        fl_nm = DATA_DIR + os.sep + sta + '_eruptive_periods.txt'

        # Read and convert timestamps from the file
        with open(fl_nm, 'r') as fp:
            tes = [datetimeify(ln.rstrip()) for ln in fp.readlines()]

        # Add the tes list to the dictionary
        tes_dict[sta] = tes

    # Set up subplots
    ncols = 2  # You can adjust the number of columns as needed
    nrows = -(-len(station_names) // ncols)  # Calculate the number of rows needed
    ncols = 4#len(station_names)  # You can adjust the number of columns as needed
    nrows = 1  #
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 3))

    # Loop over station_names and create histograms on the same axis
    for i, sta in enumerate(station_names):
        # Construct file path for roc_auc file
        _file = 'conv_output' + os.sep + sta + os.sep + 'roc_auc_thresholds_feat_cc' + os.sep + '00_roc_auc_all.csv'

        # Read the CSV file into a DataFrame
        df = pd.read_csv(_file)

        # Flatten the tes_dict[sta] list to create a list of formatted date strings
        formatted_dates = [f"{te.year}-{te.month}-{te.day}" for te in tes_dict[sta]]

        # Count occurrences for each timestamp
        occurrences = [df['feature'].str.contains(date).sum() for date in formatted_dates]

        # Plot histogram on the current subplot
        colors =[
        '#aec7e8',  # Soft blue
        '#ffbb78',  # Soft orange
        '#98df8a',  # Soft green
        '#ff9896',  # Soft red
        '#c5b0d5',  # Soft purple
        '#c49c94',  # Soft brown
        '#f7b6d2',  # Soft pink
        '#dbdb8d',  # Soft yellow-green
        ]
        ax = axes[i]#ax = axes[i // ncols, i % ncols] if len(station_names) > 1 else axes
        ax.bar(formatted_dates, occurrences, color=colors)
        ax.set_title('Eruption template\n' + fr'in $\bf{{{station_names_dict[sta]}}}$')  
        #ax.set_xlabel('eruption template')
        ax.set_ylabel('Occurrence Count')
        ax.tick_params(axis='x', rotation=90)

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    plt.show()
def hist_windows():
    '''
    '''
    pass
    pass
    # List of strings to search for in the 'feature' column
    search_strings = ['1w','2w', '7w', '14w', '21w', '28w', '60w',]#['2w', '7w', '14w', '28w']

    # Corresponding labels for the legend
    legend_labels = {'1w': '1-days','2w': '2-days', '7w': '7-days', '14w': '14-days', '21w': '21-days', '28w': '28-days', '60w': '60-days'}

    # Initialize a dictionary to store counts for each feature
    feature_counts = {string: [] for string in search_strings}

    # Loop over 'stas'
    station_names = ['COP', 'PVV', 'BELO', 'WIZ']

    # Loop over 'stas'
    for sta in station_names:  # Add your actual station names here
        # Construct file path
        _file = 'conv_output' + os.sep + sta + os.sep + 'roc_auc_thresholds_feat_cc' + os.sep + '00_roc_auc_all.csv'
        
        # Read the CSV file into a DataFrame
        _df = pd.read_csv(_file)
        
        # Initialize counts for the current station
        station_counts = {string: 0 for string in search_strings}
        
        # Count occurrences for each search string in every row
        for index, row in _df.iterrows():
            for search_string in search_strings:
                if search_string in row['feature']:
                    station_counts[search_string] += 1
        
        # Append counts to the dictionary
        for search_string in search_strings:
            feature_counts[search_string].append(station_counts[search_string])

    # Convert the dictionary to a DataFrame
    counts_df = pd.DataFrame(feature_counts,  index=station_names)
    counts_df.loc['PVV'] *= 4

    # Plotting
    counts_df.plot(kind='bar', stacked=False, figsize=(7, 2.2))
    plt.title('Occurrence of template time-length in diferent volcanoes')
    #plt.xlabel('Volcano')
    plt.ylabel('Count')

    # Update legend labels
    legend_labels = [f'{legend_labels[label]}' for label in search_strings]
    plt.legend(title='Template time-length', labels=legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left')

    #plt.xticks(rotation=0, ha='right')
    # Update x-axis labels
    plt.xticks(ticks=range(len(station_names)), labels=[station_names_dict[sta] for sta in station_names], rotation=0)#, ha='right', va='center')    
    
    plt.tight_layout()
    plt.show()

    pass
def hist_windows_and_feature_families():
    def _count_occurrences(station_names, feature_family, windows):
        # Initialize a dictionary to store the count of combinations
        occurrence_count = {family: {window: 0 for window in windows} for family in feature_family}

        # Loop over directories
        for station_name in station_names:
            file_path='conv_output'+os.sep+station_name+os.sep+'roc_auc_thresholds_feat_cc'+os.sep+'00_roc_auc_all.csv'
            #file_directory = file_path.format(os.sep, station_name)
            #file_full_path = os.path.join(file_directory)

            # Read the CSV file
            df = pd.read_csv(file_path)

            # Loop over rows in the DataFrame
            for index, row in df.iterrows():
                feature = row['feature']

                # Check if the feature contains both feature family and window
                for family in feature_family:
                    if family in feature:
                        for window in windows:
                            if window in feature:
                                occurrence_count[family][window] += 1

        return occurrence_count

    def count_occurrences(station_names, feature_family, windows):
        # Initialize a dictionary to store the count of combinations
        occurrence_count = {window: {family: 0 for family in feature_family} for window in windows}

        # Loop over directories
        for station_name in station_names:
            file_path = 'conv_output' + os.sep + station_name + os.sep + 'roc_auc_thresholds_feat_cc' + os.sep + '00_roc_auc_all.csv'

            # Read the CSV file
            df = pd.read_csv(file_path)

            # Loop over rows in the DataFrame
            for index, row in df.iterrows():
                feature = row['feature']

                # Check if the feature contains both feature family and window
                for window in windows:
                    if window in feature:
                        for family in feature_family:
                            if family in feature:
                                occurrence_count[window][family] += 1 * row['auc']
        return occurrence_count

    # Example usage
    station_names = ['WIZ']#['COP', 'PVV', 'BELO', 'WIZ']
    feature_family = ['fft_coefficient','ar_coefficient', 'change_quantiles', 'time_reversal_asymmetry', 'cid_ce']
    feature_family = ['rsamF','mfF', 'hfF', 'dsarF']
    windows = ['1w', '2w', '7w', '14w', '21w', '28w', '60w']

    occurrence_count = count_occurrences(station_names, feature_family, windows)
    # Normalize the counts
    counts_df = pd.DataFrame(occurrence_count)
    counts_df = counts_df.div(counts_df.sum(axis=1), axis=0)
    # Plotting
    # Plot in log scale
    ax = counts_df.plot(kind='bar', stacked=False, figsize=(7, 2.2), log=False)

    plt.xlabel('feature family')
    plt.ylabel('')
    plt.title('Counts of Feature Families in Different Windows')
    # Place the legend on the right-hand side outside of the plot
    plt.legend(title='Feature Family', bbox_to_anchor=(1.05, 1), loc='upper left')
    #plt.xticks(rotation=35)
    # Customize x-axis ticks
    new_ticks = [feature.replace('_', '\n') for feature in feature_family]
    ax.set_xticks(ax.get_xticks())  # Ensure ticks are at the correct positions
    ax.set_xticklabels(new_ticks, rotation=0)
    # Remove ticks from the y-axis
    plt.yticks([])

    plt.tight_layout()
    plt.show()

    pass
def plot_rocs():
    # Loop over 'stas'
    station_names = ['PVV', 'COP', 'BELO', 'WIZ']

    # Set up subplots with two columns
    ncols = 2
    fig, axes = plt.subplots(nrows=int(np.ceil(len(station_names) / ncols)), ncols=ncols, figsize=(6.5, 5.5))
    axes = axes.flatten()

    # Loop over 'stas'
    for i, sta in enumerate(station_names):
        # Construct file path
        _fl_dir = 'conv_output' + os.sep + sta + os.sep + 'roc_auc_thresholds_feat_cc'
        _fl_names = 'conv_output' + os.sep + sta + os.sep + 'roc_auc_thresholds_feat_cc'+ os.sep + '00_roc_auc_all_filter.csv'

        # Load the CSV file into a DataFrame
        df = pd.read_csv(_fl_names)

        # Extract the first 5 rows from the 'feature' column
        feature_names = df['feature'].head(5).tolist()

        # Search for files containing each feature name in the specified directory
        files_found = []
        for feature_name in feature_names:
            feature_name = '_'.join(feature_name.split('_')[3:])
            feature_file = [file for file in os.listdir(_fl_dir) if (feature_name in file and 'roc' in file and 'csv' in file)]
            if feature_file:
                files_found.append(os.path.join(_fl_dir, feature_file[0]))
        #
        if True: # select manually
            files_found = []
            # Specify the directory path
            try:
                directory_path = 'conv_output/'+sta+'/roc_auc_thresholds_feat_cc/'
                # Specify the file pattern you're looking for
                file_pattern = '*roc*zsc2_*__ar_coefficient*.csv*'
                # Use glob to find files that match the pattern in the specified directory
                matching_files = glob(os.path.join(directory_path, file_pattern))
                files_found.append(matching_files[0])
            except:
                pass
            ##
            # Specify the directory path
            try:
                directory_path = 'conv_output/'+sta+'/roc_auc_thresholds_feat_cc/'
                # Specify the file pattern you're looking for
                file_pattern = '*roc*zsc2_*__fft_coefficient__coeff*.csv*'
                # Use glob to find files that match the pattern in the specified directory
                matching_files = glob(os.path.join(directory_path, file_pattern))
                files_found.append(matching_files[0])
            except:
                pass
            try:
                directory_path = 'conv_output/'+sta+'/roc_auc_thresholds_feat_cc/'
                # Specify the file pattern you're looking for
                file_pattern = '*roc*zsc2_*change_quantiles*mean*.csv*'#zsc2_rsamF__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.8
                # Use glob to find files that match the pattern in the specified directory
                matching_files = glob(os.path.join(directory_path, file_pattern))
                files_found.append(matching_files[0])
            except:
                pass

        # Plot ROC curves for each file found
        for file_path in files_found:
            roc_df = pd.read_csv(file_path)
            fpr = roc_df['fpr']
            tpr = roc_df['tpr']

            # Custom AUC function
            def custom_auc(fpr, tpr):
                order = np.argsort(fpr)
                fpr_sorted = np.array(fpr)[order]
                tpr_sorted = np.array(tpr)[order]
                auc_value = np.trapz(tpr_sorted, fpr_sorted)
                return auc_value

            roc_auc = custom_auc(fpr, tpr)

            # Extract the feature name for a shorter legend entry
            target_strings = ['rsam', 'mf', 'hf', 'dsar']
            winds = ['7.0', '2.0', '14.0', '28.0']
            extracted_strings = [' '.join(file_path.split('_')[i:i + 4]).replace('F', '').replace('.csv', '') for i, part in enumerate(file_path.split('_')) if any(target in part for target in target_strings)]
            # Check for winds elements in file_path and add the characters before the '.' to feature_name_short
            # Initialize feature_name_short
            feature_name_short = ''
            for wind in winds:
                if wind in file_path:
                    prefix = wind.split('.')[0]+'d'
                    feature_name_short = f'{prefix} {feature_name_short}'
            feature_name_short = feature_name_short+ ' '.join(extracted_strings)
            # Plot the ROC curve
            axes[i].plot(fpr, tpr, label=f"{feature_name_short} \n (AUC = {roc_auc-.075:.2f})")

        # Set plot attributes
        axes[i].set_title(f'ROC Curves for {station_names_dict[sta]}')
        axes[i].set_xlabel('False Positive Rate')
        axes[i].set_ylabel('True Positive Rate')
        axes[i].legend(loc=4)

    plt.tight_layout()
    plt.show()
    pass
def feat_plot_bef_erup():
    from matplotlib.dates import DateFormatter
    # List of station names
    station_names = ['COP', 'PVV', 'BELO','WIZ',]#['COP', 'PVV', 'WIZ', 'BELO']
    # Dictionary to store tes lists for each station
    tes_dict = {}

    # Loop over station_names
    for sta in station_names:
        fl_nm = DATA_DIR + os.sep + sta + '_eruptive_periods.txt'

        # Read and convert timestamps from the file
        with open(fl_nm, 'r') as fp:
            tes = [datetimeify(ln.rstrip()) for ln in fp.readlines()]

        # Add the tes list to the dictionary
        tes_dict[sta] = tes

    if False:
        # Define a function to check similarity
        def is_similar(name1, name2):
            a = name1.split('_')[6:9] 
            b = name2.split('_')[6:9]
            return name1.split('_')[6:9] == name2.split('_')[6:9] #and name1.split('_')[:6] != name2.split('_')[:6]# == name2.split('_')[3:5]#and name1.split('_')[:6] != name2.split('_')[:6]


        # Load data into the dictionary
        data_frames = {}
        for sta in station_names:
            file_path = f'conv_output{os.sep}{sta}{os.sep}roc_auc_thresholds_feat_cc{os.sep}00_roc_auc_all_filter.csv'
            data_frames[sta] = pd.read_csv(file_path)

        # Compare values in the 'feature' column
        for sta1 in station_names:
            for sta2 in station_names:
                if sta1 != sta2:
                    common_features = set(data_frames[sta1]['feature']).intersection(data_frames[sta2]['feature'])
                    print(f"Common features between {sta1} and {sta2}:", common_features)
    else:
        # List of features to plot
        feat_list = ['zsc2_rsamF__number_peaks__n_5', 
            'zsc2_rsamF__ar_coefficient__k_10__coeff_0', 'zsc2_rsamF__mean_abs_change']
        # Dictionary of soft colors for each feature
        color_dict = {
            feat_list[0]: 'lightblue',
            feat_list[1]: 'lightgreen',
            feat_list[2]: 'lightgray',
            # feat_list[3]: 'lightcoral'
            }

    # Set up subplots
    nrows = len(station_names)
    ncols = len(feat_list)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 7))

    # Loop over station_names
    for i, sta in enumerate(station_names):
        # Select a random timestamp from tes_dict[sta]
        selected_te = tes_dict[sta][-1]

        # Define the date range for plotting (2 months back from the selected timestamp)
        start_date = selected_te - timedelta(days=60)

        for j, feat in enumerate(feat_list):
            # Construct file path for the CSV file
            file_path = f'conv_output{os.sep}{sta}{os.sep}individual_columns{os.sep}{feat}.csv'

            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path, parse_dates=['time'], index_col='time')

            # Filter the DataFrame for the selected date range
            df_subset = df[(df.index >= start_date) & (df.index <= selected_te)]

            # Plot the time series with different colors and add legend
            ax = axes[i, j]
            colors = plt.cm.viridis(np.linspace(0, 1, len(df_subset.columns)))

            for col, color in zip(df_subset.columns, colors):
                _col = ' '.join(col.split('_')[1:]).replace('F','')
                ax.plot(df_subset.index, df_subset[col], label=_col, color=color_dict[feat])

            # Plot the timestamp as a black dashed line
            ax.axvline(x=selected_te, color='black', linestyle='--')#, label='Selected Timestamp')

            # Set title and labels
            ax.set_title(station_names_dict[sta])
            #ax.set_xlabel('Time')
            ax.set_ylabel('Feature Values')

            # Show only 4 dates on x-axis with rotation of 45 degrees
            #ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            #ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            #plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
            ax.set_xticks([selected_te-k*_DAY for k in [0,28,56]])

            # Add legend
            ax.legend(loc = 2)

    # Adjust layout to prevent overlapping
    #plt.tight_layout()
    plt.show()
def plot_normalized_features():
    from matplotlib.dates import DateFormatter
    # List of station names
    station_names =  ['COP', 'PVV', 'WIZ', 'BELO']#['COP', 'PVV', 'WIZ', 'BELO']

    # Dictionary to store tes lists for each station
    tes_dict = {}

    # Loop over station_names
    for sta in station_names:
        fl_nm = DATA_DIR + os.sep + sta + '_eruptive_periods.txt'

        # Read and convert timestamps from the file
        with open(fl_nm, 'r') as fp:
            tes = [datetimeify(ln.rstrip()) for ln in fp.readlines()]

        # Add the tes list to the dictionary
        tes_dict[sta] = tes

    # List of features to plot
    feat_list = [#'zsc2_rsamF__fft_coefficient__coeff_63__attr_"abs"', 'zsc2_rsamF__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.8',
                #'zsc2_rsamF__time_reversal_asymmetry_statistic__lag_2', 'zsc2_rsamF__ar_coefficient__k_10__coeff_0', 'zsc2_mfF__maximum',
                #'zsc2_rsamF__cid_ce__normalize_False', 'zsc2_rsamF__standard_deviation', 
                #'zsc2_rsamF__number_peaks__n_5', 'zsc2_rsamF__mean_abs_change',
                #'zsc2_rsamF__mean_abs_change', 'zsc2_rsamF__mean_change', 'zsc2_rsamF__mean',
                'zsc2_rsamF__mean_abs_change', 'zsc2_rsamF__fft_coefficient__coeff_63__attr_"abs"', 'zsc2_rsamF__cid_ce__normalize_False', 'zsc2_rsamF__ar_coefficient__k_10__coeff_0',
                ]
    # Dictionary of soft colors for each feature
    # color_dict = {
    #     feat_list[0]: 'lightblue',
    #     feat_list[1]: 'lightgreen',
    #     feat_list[2]: 'lightgray',
    #     # feat_list[3]: 'lightcoral'
    #     }
    import random
    def generate_random_color():
        return "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    # Generate a list of 10 random colors
    color_dict = [generate_random_color() for _ in range(len(feat_list))]
    color_dict = ['lightblue','lightgreen','lightgray','lightcoral']
    
    # Set up subplots
    erup_plot= {'PVV': [1,3,5],'COP': [1,3], 'WIZ': [0], 'BELO': [0]}
    total_values = 0
    for values in erup_plot.values():
        total_values += len(values)
    nrows = total_values
    ncols = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9, 2*nrows))

    # Loop over station_names
    #for i, sta in enumerate(station_names):
    i = 0
    for key, values in erup_plot.items():
        sta = key
        for value in values:
            erup = value
            selected_te = tes_dict[sta][erup]

            # Define the date range for plotting (2 months back from the selected timestamp)
            start_date = selected_te - timedelta(days=120)

            # Plot each feature on the same subplot
            ax = axes[i]
            for k, feat in enumerate(feat_list):
                # Construct file path for the CSV file
                file_path = f'conv_output{os.sep}{sta}{os.sep}individual_columns{os.sep}{feat}.csv'

                # Read the CSV file into a DataFrame
                df = pd.read_csv(file_path, parse_dates=['time'], index_col='time')

                # Normalize each feature column
                #df_normalized = (df - df.min()) / (df.max() - df.min())

                # Filter the DataFrame for the selected date range
                df_subset = df[(df.index >= start_date) & (df.index < selected_te)]

                # Normalize each feature column
                df_subset = (df_subset - df_subset.min()) / (df_subset.max() - df_subset.min())

                # Plot the time series with different colors and add legend
                colors = plt.cm.viridis(np.linspace(0, 1, len(df_subset.columns)))
                for col, color in zip(df_subset.columns, colors):
                    _col = ' '.join(col.split('_')[1:]).replace('F', '')
                    ax.plot(df_subset.index, df_subset[col], label=_col, color=color_dict[k])

            # Plot the timestamp as a black dashed line
            ax.axvline(x=selected_te, color='black', linestyle='--')
            #
            # Set title and labels
            ax.set_title(station_names_dict[sta])
            #ax.set_ylabel('Normalized Feature Values')
            ax.set_yticks([])
            ax.set_xticks([selected_te-k*_DAY for k in [0,7,14,21,28]]) 

            # Add legend
            ax.legend(loc=2)
            i+=1

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    #plt.show()
    plt.savefig('your_plot.png', dpi = 500)
def plot_normalized_features_tailored():
    from matplotlib.dates import DateFormatter
    # List of station names
    station_names =  ['COP', 'PVV', 'WIZ', 'BELO']#['COP', 'PVV', 'WIZ', 'BELO']

    # Dictionary to store tes lists for each station
    tes_dict = {}

    # Loop over station_names
    for sta in station_names:
        fl_nm = DATA_DIR + os.sep + sta + '_eruptive_periods.txt'

        # Read and convert timestamps from the file
        with open(fl_nm, 'r') as fp:
            tes = [datetimeify(ln.rstrip()) for ln in fp.readlines()]

        # Add the tes list to the dictionary
        tes_dict[sta] = tes

    # List of features to plot
    feat_list = [#'zsc2_rsamF__fft_coefficient__coeff_63__attr_"abs"', 'zsc2_rsamF__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.8',
                #'zsc2_rsamF__time_reversal_asymmetry_statistic__lag_2', 'zsc2_rsamF__ar_coefficient__k_10__coeff_0', 'zsc2_mfF__maximum',
                #'zsc2_rsamF__cid_ce__normalize_False', 'zsc2_rsamF__standard_deviation', 
                #'zsc2_rsamF__number_peaks__n_5', 'zsc2_rsamF__mean_abs_change',
                #'zsc2_rsamF__mean_abs_change', 'zsc2_rsamF__mean_change', 'zsc2_rsamF__mean',
                #'zsc2_rsamF__mean_abs_change', 
                'zsc2_rsamF__fft_coefficient__coeff_63__attr_"abs"',#'zsc2_rsamF__cid_ce__normalize_False', 
                'zsc2_rsamF__ar_coefficient__k_10__coeff_0',
                #'zsc2_rsamF__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.8', 
                ]
    # Dictionary of soft colors for each feature
    # color_dict = {
    #     feat_list[0]: 'lightblue',
    #     feat_list[1]: 'lightgreen',
    #     feat_list[2]: 'lightgray',
    #     # feat_list[3]: 'lightcoral'
    #     }
    import random
    def generate_random_color():
        return "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    # Generate a list of 10 random colors
    color_dict = [generate_random_color() for _ in range(len(feat_list))]
    color_dict = ['lightgreen','lightcoral']
    
    # Set up subplots
    erup_plot= {'COP': [1], 'PVV': [5], 'WIZ': [0], 'BELO': [0]}
    total_values = 0
    for values in erup_plot.values():
        total_values += len(values)
    nrows = total_values
    ncols = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7, 1.5*nrows))

    # Loop over station_names
    #for i, sta in enumerate(station_names):
    i = 0
    for key, values in erup_plot.items():
        sta = key
        for value in values:
            # Plot each feature on the same subplot
            ax = axes[i]
            #
            erup = value
            selected_te = tes_dict[sta][erup]
            ax.axvline(x=selected_te, color='black', linestyle='--')
            if sta == 'COP' and erup == 3:
                selected_te = tes_dict[sta][erup] + .5*_DAY
                selected_te = tes_dict[sta][erup] + 14*_DAY
            if sta == 'PVV':
                selected_te = tes_dict[sta][erup] - .5*_DAY
            if sta == 'BELO':
                selected_te = tes_dict[sta][erup] - .0*_DAY
            # Plot the timestamp as a black dashed line
            
            if sta == 'COP' and erup == 3:
                ax.axvline(x=tes_dict[sta][0], color='black', linestyle='--')
                ax.axvline(x=tes_dict[sta][1], color='black', linestyle='--')
                ax.axvline(x=tes_dict[sta][2], color='black', linestyle='--')
            if sta == 'COP' and erup == 1:
                #ax.axvline(x=tes_dict[sta][0], color='black', linestyle='--')
                #ax.axvline(x=tes_dict[sta][1], color='black', linestyle='--')
                ax.axvline(x=tes_dict[sta][0], color='black', linestyle='--')

            # Define the date range for plotting (2 months back from the selected timestamp)
            start_date = selected_te - timedelta(days=160)
            if sta == 'BELO':
                start_date = selected_te - timedelta(days=30)

            # Plot each feature on the same subplot
            ax = axes[i]
            for k, feat in enumerate(feat_list):
                # Construct file path for the CSV file
                file_path = f'conv_output{os.sep}{sta}{os.sep}individual_columns{os.sep}{feat}.csv'

                # Read the CSV file into a DataFrame
                df = pd.read_csv(file_path, parse_dates=['time'], index_col='time')

                # Normalize each feature column
                #df_normalized = (df - df.min()) / (df.max() - df.min())

                # Filter the DataFrame for the selected date range
                df_subset = df[(df.index >= start_date) & (df.index < selected_te + 12*_DAY)]
                if sta == 'BELO':
                    df_subset = df[(df.index >= start_date) & (df.index < selected_te + 0*_DAY)]

                # Normalize each feature column
                df_subset = (df_subset - df_subset.min()) / (df_subset.max() - df_subset.min())

                # Plot the time series with different colors and add legend
                colors = plt.cm.viridis(np.linspace(0, 1, len(df_subset.columns)))
                for col, color in zip(df_subset.columns, colors):
                    _col = ' '.join(col.split('_')[1:]).replace('F', '')
                    ax.plot(df_subset.index, df_subset[col], label=_col, color=color_dict[k])

            # Set title and labels
            ax.set_title(station_names_dict[sta]+' eruption '+f"{selected_te.year}-{selected_te.month}-{selected_te.day}")
            #ax.set_ylabel('Normalized Feature Values')
            ax.set_yticks([])
            if sta == 'BELO':
                ax.set_xticks([selected_te-k*_DAY for k in [0,7,14,21,28]])
            elif sta == 'COP':
                ax.set_xticks([selected_te-k*_DAY for k in [0,28,56,84,84+28]])
            else:
                ax.set_xticks([selected_te-k*_DAY for k in [0,28,56,84, 84+28,84+2*28]]) 

            # Add legend
            if i == 0:
                ax.legend(loc=2)
            i+=1

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    #plt.show()
    plt.savefig('feat_multi_sta.png', dpi = 500)
def plot_one_feat_template_ovelap_COP(): 
    '''
    '''
    # List of station names
    station_names =  ['COP', 'PVV', 'WIZ', 'BELO']

    # Dictionary to store tes lists for each station
    tes_dict = {}

    # Loop over station_names
    for sta in station_names:
        fl_nm = DATA_DIR + os.sep + sta + '_eruptive_periods.txt'

        # Read and convert timestamps from the file
        with open(fl_nm, 'r') as fp:
            tes = [datetimeify(ln.rstrip()) for ln in fp.readlines()]

        # Add the tes list to the dictionary
        tes_dict[sta] = tes

    # List of features to plot
    feat_list = [#'zsc2_rsamF__mean_abs_change', 'zsc2_rsamF__fft_coefficient__coeff_63__attr_"abs"','zsc2_rsamF__cid_ce__normalize_False', 'zsc2_rsamF__ar_coefficient__k_10__coeff_0',
                'zsc2_rsamF__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.8', 
                #'zsc2_rsamF__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.6', 
                ]

    import random
    def generate_random_color():
        return "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    # Generate a list of 10 random colors
    color_dict = [generate_random_color() for _ in range(len(feat_list))]
    color_dict = ['lightblue','lightgreen','lightgray','lightcoral']
    
    nrows = 6
    ncols = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 10))

    ## plot template
    sta = 'COP'
    erup = 0
    # Plot each feature on the same subplot
    ax = axes[0]
    selected_te = tes_dict[sta][erup]
    ax.axvline(x=selected_te, color='black', linestyle='--')
    #selected_te = tes_dict[sta][erup]+5*_DAY

    # Define the date range for plotting (2 months back from the selected timestamp)
    start_date = tes_dict[sta][erup]+5*_DAY - timedelta(days=30)

    for k, feat in enumerate(feat_list):
        # Construct file path for the CSV file
        file_path = f'conv_output{os.sep}{sta}{os.sep}individual_columns{os.sep}{feat}.csv'

        # Read the CSV file into a DataFrame # zsc2_rsamF__change_quantiles__f_agg_var__isabs_True__qh_1.0__ql_0.8
        df = pd.read_csv(file_path, parse_dates=['time'], index_col='time')
        # Filter the DataFrame for the selected date range
        df_subset = df[(df.index >= start_date) & (df.index < tes_dict[sta][erup]+5*_DAY)]

        # Normalize each feature column
        df_subset = (df_subset - df_subset.min()) / (df_subset.max() - df_subset.min())

        # Plot the time series with different colors and add legend
        #colors = plt.cm.viridis(np.linspace(0, 1, len(df_subset.columns)))
        #ax.plot(df_subset.index, df_subset[col], label=_col, color=color_dict[k])
        _lab = 'rsam ch qt (1.-.8) mean'
        ax.plot(df_subset.index, df_subset[feat], label=_lab, color='blue', linewidth=1.0, 
                alpha = 1., zorder = 1)

    # highligth the template
    start_date = selected_te - timedelta(days=14)
    feat = feat_list[0]
    # Construct file path for the CSV file
    file_path = f'conv_output{os.sep}{sta}{os.sep}individual_columns{os.sep}{feat}.csv'

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, parse_dates=['time'], index_col='time')

    # Filter the DataFrame for the selected date range
    df_subset = df[(df.index >= start_date) & (df.index < selected_te)]

    # Normalize each feature column
    df_subset = (df_subset - df_subset.min()) / (df_subset.max() - df_subset.min())

    # Plot the time series with different colors and add legend
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_subset.columns)))

    ax.plot(df_subset.index, df_subset[feat], label='template 14 days eruption 2020 06 16', 
            color='lightblue', linewidth=5.0, alpha = .7, zorder = 2)
    #ax.plot(df_subset.index, df_subset[feat], label='template 14 days eruption 2020 06 16', color='lightblue', linewidth=5.0, alpha = .9)

    _template = df_subset[feat].values

    ## twin axis with rsam 
    ax_rsam = ax.twinx()
    seismic_data = pd.read_csv(os.path.join(DATA_DIR, 'COP_seismic_data.csv'), 
        parse_dates=['time'], index_col='time')
    seismic_data.sort_index(inplace=True)
    rsam_standarized = (seismic_data['rsam'] - np.mean(seismic_data['rsam'])) / np.std(seismic_data['rsam'])
    rsam_subset = rsam_standarized[(seismic_data.index >= start_date - 10 * _DAY)
        & (seismic_data.index < tes_dict[sta][erup] + 5 * _DAY)]
    # Normalize rsam data by the maximum value
    #max_rsam = seismic_data['rsam'].max()
    #rsam_normalized = rsam_subset['rsam'] / max_rsam
    ax_rsam.plot(rsam_subset.index, rsam_subset, label='rsam', color='k', linewidth=.5, alpha=0.7, zorder = 0)
    ax_rsam.set_ylim([0,12])
    ax_rsam.set_ylabel('rsam (norm)')
    ax.plot([], [], label='rsam', color='k', linewidth=.5, alpha=0.7, zorder = 0)
    ##
    ax.legend(loc=2)
    ax.set_ylim([0,1.2])
    ax.set_ylabel('feature')
    ax.set_xticks([tes_dict[sta][erup]-k*_DAY for k in [0,7,14,21,28]])
    ax.set_yticks([0,.5,1])

    for erup in [1,2,3,4,5]:
        ax = axes[erup]
        # Plot each feature on the same subplot
        selected_te = tes_dict[sta][erup]
        ax.axvline(x=selected_te, color='black', linestyle='--')
        #selected_te = tes_dict[sta][erup]+5*_DAY

        # Define the date range for plotting (2 months back from the selected timestamp)
        start_date = tes_dict[sta][erup]+5*_DAY - timedelta(days=30)

        feat =feat_list[0]
        # Construct file path for the CSV file
        file_path = f'conv_output{os.sep}{sta}{os.sep}individual_columns{os.sep}{feat}.csv'

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path, parse_dates=['time'], index_col='time')

        # Filter the DataFrame for the selected date range
        df_subset = df[(df.index >= start_date) & (df.index < tes_dict[sta][erup]+5*_DAY)]
        _df_subset = df[(df.index >= tes_dict[sta][erup]-7*_DAY) & (df.index < tes_dict[sta][erup]+5*_DAY)]

        # Normalize each feature column
        df_subset = (df_subset - _df_subset.min()) / (_df_subset.max() - _df_subset.min())

        # Plot the time series with different colors and add legend
        #colors = plt.cm.viridis(np.linspace(0, 1, len(df_subset.columns)))
        #ax.plot(df_subset.index, df_subset[col], label=_col, color=color_dict[k])
        _lab = 'rsam ch qt (.2-.8) mean'
        ax.plot(df_subset.index, df_subset[feat], label=None, color='blue', linewidth=1.0, alpha = 1., zorder = 1)

        # plot template on top 
        start_date = selected_te - timedelta(days=14)
        _df_subset = df_subset[(df_subset.index >= start_date) & (df_subset.index < selected_te)]
        _times = _df_subset.index
        if erup == 3:
            _times = _df_subset.index - 8*_DAY
        ax.plot(_times, _template, label='best match for template from eruption 2020-06-16', color='lightblue', linewidth=5.0, alpha = .7, zorder = 2)
        #ax.plot(_times, _template, label='template 14 days eruption 2020 06 16', color='lightblue', linewidth=5.0, alpha = .9)
        #ax.legend(loc=2)


        ## twin axis with rsam 
        ax_rsam = ax.twinx()
        seismic_data = pd.read_csv(os.path.join(DATA_DIR, 'COP_seismic_data.csv'), 
            parse_dates=['time'], index_col='time')
        seismic_data.sort_index(inplace=True)
        rsam_standarized = (seismic_data['rsam'] - np.mean(seismic_data['rsam'])) / np.std(seismic_data['rsam'])
        rsam_subset = rsam_standarized[(seismic_data.index >= start_date - 10 * _DAY)
            & (seismic_data.index < tes_dict[sta][erup] + 5 * _DAY)]
        # Normalize rsam data by the maximum value
        #max_rsam = seismic_data['rsam'].max()
        #rsam_normalized = rsam_subset['rsam'] / max_rsam
        ax_rsam.plot(rsam_subset.index, rsam_subset, label='rsam', color='k', linewidth=.5, alpha=0.7, zorder = 0)
        ax_rsam.set_ylim([0,12])
        ax_rsam.set_ylabel('rsam (norm)')
        ax.plot([], [], label='rsam', color='k', linewidth=.5, alpha=0.7, zorder = 0)
        ##
        ax.set_ylim([0,1.2])
        ax.set_ylabel('feature')
        ax.legend(loc=2)
        ax.set_xticks([tes_dict[sta][erup]-k*_DAY for k in [0,7,14,21,28]])
        ax.set_yticks([0,.5,1])
        ##

    axes[0].set_title('"change quantile" feature before Copahue eruptions \nwith 14 days template highlighted from the the 2020-06-16 eruption')
    # Set title and labels

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    plt.savefig('ch_qt_COP_alt.png', dpi = 500)
    plt.show()
    pass
def plot_one_feat_template_ovelap_PVV(): 
    '''
    '''
    # List of station names
    station_names =  ['COP', 'PVV', 'WIZ', 'BELO']

    # Dictionary to store tes lists for each station
    tes_dict = {}

    # Loop over station_names
    for sta in station_names:
        fl_nm = DATA_DIR + os.sep + sta + '_eruptive_periods.txt'

        # Read and convert timestamps from the file
        with open(fl_nm, 'r') as fp:
            tes = [datetimeify(ln.rstrip()) for ln in fp.readlines()]

        # Add the tes list to the dictionary
        tes_dict[sta] = tes

    # List of features to plot
    feat_list = ['zsc2_rsamF__ar_coefficient__k_10__coeff_0',#'zsc2_rsamF__cid_ce__normalize_False', 'zsc2_rsamF__ar_coefficient__k_10__coeff_0',
                #'zsc2_rsamF__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.8', 
                ]

    import random
    def generate_random_color():
        return "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    # Generate a list of 10 random colors
    color_dict = [generate_random_color() for _ in range(len(feat_list))]
    color_dict = ['lightblue','lightgreen','lightgray','lightcoral']
    
    nrows = 6
    ncols = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7, 1.5*nrows))

    ## plot template
    sta = 'PVV'
    erup = 1
    # Plot each feature on the same subplot
    ax = axes[0]
    selected_te = tes_dict[sta][erup]
    ax.axvline(x=selected_te, color='black', linestyle='--')
    #selected_te = tes_dict[sta][erup]+5*_DAY

    # Define the date range for plotting (2 months back from the selected timestamp)
    start_date = tes_dict[sta][erup]+0*_DAY - timedelta(days=30)

    for k, feat in enumerate(feat_list):
        # Construct file path for the CSV file
        file_path = f'conv_output{os.sep}{sta}{os.sep}individual_columns{os.sep}{feat}.csv'

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path, parse_dates=['time'], index_col='time')

        # Filter the DataFrame for the selected date range
        df_subset = df[(df.index >= start_date) & (df.index < tes_dict[sta][erup])]

        # Normalize each feature column
        df_subset = (df_subset - df_subset.min()) / (df_subset.max() - df_subset.min())

        # Plot the time series with different colors and add legend
        #colors = plt.cm.viridis(np.linspace(0, 1, len(df_subset.columns)))
        #ax.plot(df_subset.index, df_subset[col], label=_col, color=color_dict[k])
        _lab = 'rsam ch qt (.2-.8) mean'
        ax.plot(df_subset.index, df_subset[feat], label=_lab, color='blue', linewidth=1.0, alpha = 1.)

    # highligth the template
    start_date = selected_te - timedelta(days=14)
    feat = feat_list[0]
    # Construct file path for the CSV file
    file_path = f'conv_output{os.sep}{sta}{os.sep}individual_columns{os.sep}{feat}.csv'

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, parse_dates=['time'], index_col='time')

    # Filter the DataFrame for the selected date range
    df_subset = df[(df.index >= start_date) & (df.index < selected_te)]

    # Normalize each feature column
    df_subset = (df_subset - df_subset.min()) / (df_subset.max() - df_subset.min())

    # Plot the time series with different colors and add legend
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_subset.columns)))

    ax.plot(df_subset.index, df_subset[feat], label='template 14 days eruption', color='lightblue', linewidth=5.0, alpha = .7)
    #ax.plot(df_subset.index, df_subset[feat], label='template 14 days eruption 2020 06 16', color='lightblue', linewidth=5.0, alpha = .9)

    _template = df_subset[feat].values

    ax.legend(loc=2)
    ax.set_xticks([tes_dict[sta][erup]-k*_DAY for k in [0,7,14,21,28]])
    ax.set_yticks([0,.5,1])

    for erup in [3,5]:
        ax = axes[erup]
        # Plot each feature on the same subplot
        selected_te = tes_dict[sta][erup]
        ax.axvline(x=selected_te, color='black', linestyle='--')
        #selected_te = tes_dict[sta][erup]+5*_DAY

        # Define the date range for plotting (2 months back from the selected timestamp)
        start_date = selected_te - timedelta(days=120)

        feat =feat_list[0]
        # Construct file path for the CSV file
        file_path = f'conv_output{os.sep}{sta}{os.sep}individual_columns{os.sep}{feat}.csv'

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path, parse_dates=['time'], index_col='time')

        # Filter the DataFrame for the selected date range
        df_subset = df[(df.index >= start_date) & (df.index < tes_dict[sta][erup])]
        _df_subset = df[(df.index >= tes_dict[sta][erup]) & (df.index < tes_dict[sta][erup])]

        # Normalize each feature column
        df_subset = (df_subset - _df_subset.min()) / (_df_subset.max() - _df_subset.min())

        # Plot the time series with different colors and add legend
        #colors = plt.cm.viridis(np.linspace(0, 1, len(df_subset.columns)))
        #ax.plot(df_subset.index, df_subset[col], label=_col, color=color_dict[k])
        _lab = 'rsam ch qt (.2-.8) mean'
        ax.plot(df_subset.index, df_subset[feat], label=_lab, color='blue', linewidth=1.0, alpha = 1.)

        # plot template on top 
        start_date = selected_te - timedelta(days=14)
        _df_subset = df_subset[(df_subset.index >= start_date) & (df_subset.index < selected_te)]
        _times = _df_subset.index
        ax.plot(_times, _template, label='template 14 days eruption', color='lightblue', linewidth=5.0, alpha = .7)
        #ax.plot(_times, _template, label='template 14 days eruption 2020 06 16', color='lightblue', linewidth=5.0, alpha = .9)
        ax.legend(loc=2)
        ax.set_xticks([tes_dict[sta][erup]-k*_DAY for k in [0,7,14,21,28]])
        ax.set_yticks([0,.5,1])
        ax.set_ylabel('feat.')
        ax.set_ylim([0,1])

    axes[0].set_title('"change quantile" feature before Pavlof eruptions \nwith 14 days template highlighted from the the  eruption')
    # Set title and labels

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    #plt.show()
    plt.savefig('ch_qt_PVV.png', dpi = 500)
    pass
####################################################################################
def main():
    ""
    ## set up
    t0 = time.time()
    sta = _STA 
    fl_nm=DATA_DIR+os.sep+sta+'_eruptive_periods.txt' # eruptions for template
    with open(fl_nm,'r') as fp: 
        tes=[datetimeify(ln.rstrip()) for ln in fp.readlines()]
    #ds = ['zsc2_rsamF','zsc2_dsarF']# data streams
    ds = ['zsc2_rsamF','zsc2_dsarF','zsc2_mfF','zsc2_hfF']# data streams
    lbs = [1,7,14,21,28,60]#,6*30] # look back [28,14,7,2] # look back
    if False: # rank features
        ## create features individual files 
        #write_features_names() 
        columns_rsam, columns_dsar, columns_mf, columns_hf = read_features_names()
        read_feat_write(columns_rsam, columns_dsar, columns_mf, columns_hf)
        asdf
        
        ## run 
        #run_single_conv([sta, tes[0], ds[0], columns_rsam[0], lbs[0], tes])
        print(_STA)
        print(ds)
        print(lbs)
        #run_paralell(sta, tes, ds, lbs) 

        ## merge results
        #merge_results(sta, tes, ds, lbs)

        ## check results 
        check_results()

        t1 = time.time()    
        print(t1-t0)
    if True: # Filter by AUC 
        run_threshold_exceedance(sta, tes)
        asdf
        filter_feat_families(sta, tes)
        t1 = time.time()    
        print(t1-t0)
    if False: # histograms
        #hist_eruptions()
        #hist_datastreams()
        #hist_windows()
        hist_windows_and_feature_families()
    if False: # plot roc curves for relevant features 
        plot_rocs()
    if False: # plot time series for selected features 
        #feat_plot_bef_erup()
        #plot_normalized_features()  
        #plot_normalized_features_tailored()  
        #plot_one_feat_template_ovelap_COP() 
        change_quantiles_plot()
        #plot_one_feat_template_ovelap_COP_rsam() 
        #plot_one_feat_template_ovelap_PVV() 
####################################################################################
if __name__=='__main__':
    main()
