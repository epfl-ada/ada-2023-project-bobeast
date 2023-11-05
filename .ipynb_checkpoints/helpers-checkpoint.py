import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


def process_interventions(interventions):
    for col in interventions.columns:
        if col != 'lang':
            interventions.loc[:, col] = pd.to_datetime(interventions.loc[:, col],errors='coerce')
        else :
            interventions[col]=interventions[col].astype('string')
    return interventions.copy()

def process_topics_linked(topics_linked):
    tl=topics_linked.copy()
    idx=np.where(tl.dtypes==bool)[0]
    tl[tl.columns[idx]]=tl[tl.columns[idx]].astype(int)
    return tl

def process_agg_time_series(agg_ts):
    res=pd.DataFrame()
    if len(agg_ts.values())==3:
        res=pd.DataFrame(agg_ts)
    else:
        res=pd.DataFrame(agg_ts.values())
        res.index=agg_ts.keys()
    res.index=pd.to_datetime(res.index)
    res['date']=res.index
    res['year']=res['date'].dt.year
    res['month']=res['date'].dt.month
    res['day']=res['date'].dt.day
    res.index=range(len(res))
    return res