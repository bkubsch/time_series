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

    df['minute_sin'] = np.sin(2 * np.pi * df.loc[:,'minute']/60)
    df['minute_cos'] = np.cos(2 * np.pi * df.loc[:,'minute']/60)

    df['hour_sin'] = np.sin(2 * np.pi * df.loc[:,'hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df.loc[:,'hour']/24)
    
    df['month_sin'] = np.sin(2 * np.pi * df.loc[:,'month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df.loc[:,'month']/12)
    
    df.drop(columns=['minute', 'hour', 'month'], inplace=True)
    
    return df

#exercise_2
def lag_horizon(df, lag, horizon):
    '''
    Returns dataset with additional features defined by lag and modified target defined by horizon
    lag=integer of how far back time series should look
    horizon=integer of how far into the future the model shall predict; horizon=0 means prediciton 1 step into future
    '''
    for i in range(1,lag+1):
        df[f'lag{i}'] = df['t CO2-e / MWh'].shift(i)
    
    for i in range(1,horizon+2):
        df[f'horizon{i-1}'] = df['t CO2-e / MWh'][lag:]
        
    return df

#lecture 4
#exercise_1
def train_validation_ts(df, relative_train, maximal_lag, horizon):
    '''
    Time series (ts) split function creates a train/test set under consideration of potential overlap between the two due to lag processing
    X_train, y_train, X_test, y_test = ...
    df=must contain target column as "target"; all other columns must be used as features
    percentage_train=how much of the total dataset shall be used for training; must be added between 0 - 1
    maximal_lag=out of all lag feature engineering, enter the maximal lag number
    '''
    k = int(df.shape[0] * relative_train)
    data_train = df.iloc[:k,:]
    #to avoid overlapping of train and test data, a gap of the maximal lag - 1 must be included between the two sets
    data_test = df.iloc[k+maximal_lag:,:]
    
    assert data_train.index.max() < data_test.index.min()
    
    #returns in the sequence X_train, y_train, X_test, y_test
    return (data_train.drop(columns=[f"horizon{horizon}","t CO2-e / MWh"], axis=1), data_train[f"horizon{horizon}"],
            data_test.drop(columns=[f"horizon{horizon}","t CO2-e / MWh"], axis=1), data_test[f"horizon{horizon}"])


#lecture_5
#Exercise_1
def errors(model, X_train, y_train, X_test, y_test):

    train_mae = (sum(abs(y_train - model.predict(X_train)))/len(y_train))
    train_mape = (sum(abs((y_train - model.predict(X_train))/y_train)))*(100/len(y_train))
    train_smape = sum(abs(y_train - model.predict(X_train)))/sum(y_train + model.predict(X_train))

    test_mae = (sum(abs(y_test - model.predict(X_test)))/len(y_test))
    test_mape = (sum(abs((y_test - model.predict(X_test))/y_test)))*(100/len(y_test))
    test_smape = sum(abs(y_test - model.predict(X_test)))/sum(y_test + model.predict(X_test))

    print(f'train_MAE: {train_mae}')
    print(f'test_MAE: {test_mae}')
    
    print(f'train_MAPE: {train_mape}')
    print(f'test_MAPE: {test_mape}')
    
    print(f'train_SMAPE: {train_smape}')
    print(f'test_SMAPE: {test_smape}')
    
#Exercise_2
plt.figure(figsize=(7,5))
plt.style.use('ggplot')

fig = plt.plot_date(y_validation.index[300:600],y_validation.iloc[300:600], linestyle='solid', marker=None, label="test", color='darkblue')
fig = plt.plot_date(y_validation.index[300:600],model.predict(X_validation)[300:600], linestyle='solid', marker=None, color='darkorange', label="pred")
plt.legend(fontsize=15)

plt.xlabel("Time of Day", labelpad=15, fontsize=15, fontweight='bold')
plt.ylabel("t CO2-e / MWh", labelpad=15, fontsize=15, fontweight='bold')

date_format = mdates.DateFormatter('%H:%M')
plt.gca().xaxis.set_major_formatter(date_format)

plt.style.use('seaborn')



plt.figure(figsize=(7,5))
plt.style.use('ggplot')

fig = plt.plot_date(y_validation.index[425:450],y_validation.iloc[425:450], linestyle='solid', marker=None, label="test", color='darkblue')
fig = plt.plot_date(y_validation.index[425:450],model.predict(X_validation)[425:450], linestyle='solid', marker=None, color='darkorange', label="pred")
plt.legend(fontsize=15)

plt.xlabel("Time of Day", labelpad=15, fontsize=15, fontweight='bold')
plt.ylabel("t CO2-e / MWh", labelpad=15, fontsize=15, fontweight='bold')

date_format = mdates.DateFormatter('%H:%M')
plt.gca().xaxis.set_major_formatter(date_format)

plt.style.use('seaborn')

#Exercise_3
X_train, y_train, X_validation, y_validation = train_test_ts(
    df=df,
    relative_train=0.8,
    maximal_lag=12,
    horizon=0)

print(df.columns)

print(X_train.index.max())
print(X_validation.index.min())

assert X_train.index.max() < X_validation.index.min()

model = xgb.XGBRegressor(max_depth=5,
                         learning_rate=0.1,
                         num_estimators=100,
                         n_jobs=3,
                         reg_alpha=0.05,
                         reg_lambda=0,
                        )

model.fit(X_train, y_train)
joblib.dump(model, '../model_all_features.pkl')

errors(model, X_train, y_train, X_validation, y_validation)