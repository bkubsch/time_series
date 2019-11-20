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