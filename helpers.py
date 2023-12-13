#Create a copy of the dataframe
df_copy=df.copy()

#Create the dataframe containing the number of left and right handed players
dfLeft = df_copy[(df_copy['bats'] == 'L') | (df_copy['throws'] == 'L')].copy()
dfRight = df_copy[(df_copy['bats'] == 'R') | (df_copy['throws'] == 'R')].copy()

#Access the two different series of this dataframe
dfLeft_ay = dfLeft.groupby('yearID')['salary'].agg(lambda x : (x.isna().sum()/len(x))*100)
dfRight_ay = dfRight.groupby('yearID')['salary'].agg(lambda x : (x.isna().sum()/len(x))*100)

#Confidence Interval Computation
sem_percent_right=dfRight_ay.sem()
sem_percent_left=dfLeft_ay.sem()
confRight_percent = stats.t.interval(0.95, len(dfRight_ay) - 1, loc = dfRight_ay.values, scale = sem_percent_right)
confLeft_percent = stats.t.interval(0.95, len(dfLeft_ay) - 1, loc = dfLeft_ay.values, scale = sem_percent_left)

#Time to plot
plt.plot(figsize=(10,4),sharex=True)
plt.plot(dfLeft_ay.index,dfLeft_ay.values,label='Left',color='red')
plt.plot(dfRight_ay.index,dfRight_ay.values,label='Right',color='blue')
plt.fill_between(dfRight_ay.index,dfRight_ay.values-confRight_percent[0],dfRight_ay.values+confRight_percent[1],alpha=0.15,color='blue',label='CI Right')
plt.fill_between(dfLeft_ay.index,dfLeft_ay.values-confLeft_percent[0],dfLeft_ay.values+confLeft_percent[1],alpha=0.15,color='red',label='CI Left')
plt.legend(loc='upper right')
plt.xlabel('Years')
plt.ylabel('Percentage of missing salaries')
plt.title('Percentage of missing salaries per year per type of player')
plt.tight_layout()

#the percentage of left-handed pitchers with salary data missing for the year 2002
dfLeft_2002 = df_copy[(df_copy['bats'] == 'L') | (df_copy['throws'] == 'L') | (df_copy['yearID']==2002)].copy()
num=len(dfLeft_2002[dfLeft_2002['salary'].isna()])/len(dfLeft_2002)*100
print(f'The percentage of left-handed pitchers with salary data missing for the year 2002 is: {num:.2f}%')







---------------------------------------------------------------------------------------------------------------------
If 95% of the data of right-handed pitchers salaries is missing in our analisis, it would make any conclusion drawn not representative of the real data, as seen in the 95% confidence intervals in the last graph, where the max and min values of these intervals are far away from the actual values in our study. The comparison between the left and right handed players would make the study obsolete.
