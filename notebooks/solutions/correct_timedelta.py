#Lecture_1
#Task1
df.DateTime = pd.to_datetime(df.DateTime)
df.set_index('DateTime', inplace = True, drop = True)


#Task2
def correct_timedelta(df, time_diff):
    '''
    df.index must be DateTimeIndex
    Returns two lists
    df=table_of_interest
    col="column_of_interest"
    time_diff=time_diff in seconds as int
    '''
    lst_1 = []
    lst_2 = []
    
    for i in range(1,df.shape[0]):
        delta = abs(df.index[i] - df.index[i-1])
        if int(delta.total_seconds()) != int(time_diff):
            lst_1.append((f'At time stamp {df.index[i]}, the interval is {int(delta.total_seconds()/60)} min or {round(float(delta.total_seconds()/3600),2)} h.'))
            lst_2.append((df.index[i], int(delta.total_seconds()/60)))
            
    return lst_1, lst_2

#Task3
new_time_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="5min")
len(new_time_range)

#lecture_3
#exercise_1
def time_columns(df):
    
    df.loc[:,'minute'] = df.index.minute
    df['hour'] = df.index.hour
    df['weekday'] = np.where(df.index.weekday < 5, 0, 1)
    df['month'] = df.index.month
    df['year'] = df.index.year

    df['minute_sin'] = np.sin(2 * np.pi * df.loc[:,'minute']/60.0)
    df['minute_cos'] = np.cos(2 * np.pi * df.loc[:,'minute']/60.0)

    df['hour_sin'] = np.sin(2 * np.pi * df.loc[:,'hour']/24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df.loc[:,'hour']/24.0)
    
    df['weekday_sin'] = np.sin(2 * np.pi * df.loc[:,'weekday']/7.0)
    
    df['month_sin'] = np.sin(2 * np.pi * df.loc[:,'month']/12.0)
    df['month_cos'] = np.cos(2 * np.pi * df.loc[:,'month']/12.0)
    
    df.drop(columns=['minute', 'weekday', 'hour', 'month'], inplace=True)
    
    return df

#exercise_2
def lag_horizon(df, lag, horizon):
    '''
    Returns dataset with additional features defined by lag and modified target defined by horizon
    lag=integer of how far back time series should look
    horizon=integer of how far into the future the model shall predict; horizon=0 means prediciton 1 step into future
    '''
    for i in range(1,lag+1):
        df['lag{}'.format(i)] = df['t CO2-e / MWh'].shift(i)
    
    for i in range(1,horizon+2):
        df['horizon{}'.format(i-1)] = df['t CO2-e / MWh'][lag+i:].shift(-i+1)
        
    return df