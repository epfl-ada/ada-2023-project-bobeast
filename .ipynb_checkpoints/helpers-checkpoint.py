import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats



def process_agg_time_series(agg_ts):
    res=pd.DataFrame(agg_ts)
    res.index=pd.to_datetime(res.index)
    res['date']=res.index
    res['year']=res['date'].dt.year
    res['month']=res['date'].dt.month
    res['day']=res['date'].dt.day
    res.index=range(len(res))
    return res