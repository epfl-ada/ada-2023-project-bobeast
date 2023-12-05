import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import re


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

def process_agg_time_series(agg_ts, col_label='sum'):
    res=pd.DataFrame()
    if len(agg_ts.values())==3:
        res=pd.DataFrame(agg_ts)
    else:
        res=pd.DataFrame(agg_ts.values())
        res.index=agg_ts.keys()
        res=res.rename(columns={0 : col_label})
    res.index=pd.to_datetime(res.index)
    res['date']=res.index
    res['year']=res['date'].dt.year
    res['month']=res['date'].dt.month
    res['day']=res['date'].dt.day
    res.index=range(len(res))
    return res

def process_eu_unemployment(dfeu,countries):
    dfeuc = dfeu.copy()
    dfeuc['C2'] = dfeu['s_adj,age,unit,sex,geo\\time'].apply(lambda x: x.split(",")[-1])
    dfeuc['age'] = dfeu['s_adj,age,unit,sex,geo\\time'].apply(lambda x: x.split(",")[1])
    dfeuc['unit'] = dfeu['s_adj,age,unit,sex,geo\\time'].apply(lambda x: x.split(",")[2])
    dfeuc['sex'] = dfeu['s_adj,age,unit,sex,geo\\time'].apply(lambda x: x.split(",")[3])
    dfeuc['s_adj'] = dfeu['s_adj,age,unit,sex,geo\\time'].apply(lambda x: x.split(",")[0])
    #remove unwanted years
    selected_cols = ['C2','age','unit','sex', 's_adj', 
                     '2020M07 ', '2020M06 ', '2020M05 ', '2020M04 ','2020M03 ','2020M02 ','2020M01 ',
                     '2019M12 ','2019M11 ','2019M10 ','2019M09 ','2019M08 ','2019M07 ',
                     '2019M06 ','2019M05 ','2019M04 ','2019M03 ','2019M02 ','2019M01 ',
                     '2018M12 ','2018M11 ','2018M10 ','2018M09 ','2018M08 ','2018M07 ',
                     '2018M06 ','2018M05 ','2018M04 ','2018M03 ','2018M02 ','2018M01 ',]
    dfeuc = dfeuc[selected_cols]
    #remove NSA rows
    dfeuc.drop(dfeuc.loc[dfeuc['s_adj'] !='SA'].index, inplace=True)
    #we remove the adj column as it is no longer useful
    dfeuc.drop('s_adj', axis=1,inplace=True)
    dfeuc = dfeuc.rename(columns={'C2': "Country_code"})
    dfeuc_merged = dfeuc.merge(countries, on="Country_code")
    #we use the melt function to pivot the dataset
    dfts = dfeuc_merged.melt(id_vars=["Country", "age", "unit", "sex", "Country_code"], 
            var_name="Date", 
            value_name="Value")
    
    #removing non digit data and space
    dfts['Value'] = dfts['Value'].apply(lambda x: re.sub(r"[a-zA-Z: ]", "", x))
    dfts['Value'] = dfts['Value'].apply(lambda x: x.replace(" ",""))
    
    dfts = dfts.loc[~(dfts.Value=="")]
    
    dfts['Value'] = dfts['Value'].apply(lambda x: float(x))
    dfts['Date']=pd.to_datetime(dfts['Date'].apply(lambda x : x.replace('M','/')))
    dfts['Year']=dfts['Date'].dt.year
    dfts['Month']=dfts['Date'].dt.month
    
    return dfts
