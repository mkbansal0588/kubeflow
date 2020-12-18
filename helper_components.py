#!/usr/bin/env python
# coding: utf-8

# In[13]:


def preprocessingstep(bucket_name, train_file, test_file) -> str:
    import pandas as pd
    import numpy as np
    
    # Now it is time to download the CSV from gcloud storage
    import os
    os.chdir("/")
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/august-sandbox-298320-580249f0836f.json"
    
    for file in [train_file, test_file]:
        path = 'gs://' + bucket_name + '/' + file
        if 'train' in path:
            df_train = pd.read_csv(path)
        else:
            df_test = pd.read_csv(path)
            
        
    # Here start the preprocessing step
    data = pd.concat([df_train, df_test],sort=True)
    target = pd.DataFrame(data.pop('SalePrice'))
    data = data.dropna(axis=1)
    list1 = []
    for key, value in data.dtypes.iteritems():
        if (value == 'object'):
            list1.append(key)
            
    for item in list1:
        data[item] = pd.Categorical(data[item])
        data[item] = data[item].cat.codes

    data['SalePrice'] = target
    
    
    # now the files are ready to be uploaded.
    
    resultant_file = 'data.csv'
    resultant_path = 'gs://' + bucket_name + '/'
    processed_file_location = resultant_path + resultant_file
    data.to_csv(processed_file_location)
    
    print(resultant_file)
    
    return resultant_file


# In[16]:


def model_training(bucket_name, processeddatafile, paramas: dict) -> float:
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    
    # Now it is time to download the CSV from gcloud storage
    import os
    os.chdir("/")
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/august-sandbox-298320-580249f0836f.json"
    
    print(processeddatafile)
    path = 'gs://' + bucket_name + '/' + processeddatafile
    print(path)
    df_data = pd.read_csv(path)
    
    #Creating train and test split -
    df_train = df_data[df_data['SalePrice'].isna() == False]
    df_test = df_data[df_data['SalePrice'].isna() == True]
    
    #Creating features and labels
    label=pd.DataFrame(df_train.pop('SalePrice'))
    x_train = df_train.iloc[:int(df_train.count()[0] * .9)].to_numpy()
    x_validation=df_train.iloc[int(df_train.count()[0] * .9):].to_numpy()
    y_train = label.iloc[:int(df_train.count()[0] * .9)].to_numpy()
    y_validate = label.iloc[int(df_train.count()[0] * .9):].to_numpy()
    
    dtrain = xgb.DMatrix(x_train, y_train)
    dvalidate = xgb.DMatrix(x_validation, y_validate)
    
    params = paramas
    
    #Metric to use to evaluate decision trees.
    params['eval_metric'] = "mae"
    # Number of rounds of boosting or number of trees to build
    num_boost_round = 999

    
    #Model fitting and predictions
    

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dvalidate, "Test")],
        early_stopping_rounds=10  #Train until eval matrix hasn't improved till N(10) round
    )
    
    return model.best_score
    


# In[15]:


def hypertune(bucket_name, processeddatafile, paramas: dict) -> dict :
    
    params = paramas
    
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    
    # Now it is time to download the CSV from gcloud storage
    import os
    os.chdir("/")
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/august-sandbox-298320-580249f0836f.json"
    
    print(processeddatafile)
    path = 'gs://' + bucket_name + '/' + processeddatafile
    print(path)
    df_data = pd.read_csv(path)
    
    #Creating train and test split -
    df_train = df_data[df_data['SalePrice'].isna() == False]
    df_test = df_data[df_data['SalePrice'].isna() == True]
    
    #Creating features and labels
    label=pd.DataFrame(df_train.pop('SalePrice'))
    x_train = df_train.iloc[:int(df_train.count()[0] * .9)].to_numpy()
    x_validation=df_train.iloc[int(df_train.count()[0] * .9):].to_numpy()
    y_train = label.iloc[:int(df_train.count()[0] * .9)].to_numpy()
    y_validate = label.iloc[int(df_train.count()[0] * .9):].to_numpy()
    
    dtrain = xgb.DMatrix(x_train, y_train)
    dvalidate = xgb.DMatrix(x_validation, y_validate)

    params = paramas
    #Metric to use to evaluate decision trees.
    params['eval_metric'] = "mae"
    # Number of rounds of boosting or number of trees to build
    num_boost_round = 999
   
    
    ######################### First two parameters ##########################
    gridsearch_params = [
        (max_depth, min_child_weight)
        for max_depth in range(5,12)
        for min_child_weight in range(5,8)]
    
    # Define initial best params and MAE
    min_mae = float("Inf")
    best_params = None
    for max_depth, min_child_weight in gridsearch_params:
        print("CV with max_depth={}, min_child_weight={}".format(max_depth, min_child_weight))
    
        # Update our parameters
        params['max_depth'] = max_depth
        params['min_child_weight'] = min_child_weight
    
        # Run CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics={'mae'},
            early_stopping_rounds=10)
    
        # Update best MAE
        mean_mae = cv_results['test-mae-mean'].min()
        boost_rounds = cv_results['test-mae-mean'].argmin()
        print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
        if mean_mae < min_mae:
            min_mae = mean_mae
            best_params = (max_depth,min_child_weight)
    print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))
    
    params['max_depth'] = best_params[0]
    params['min_child_weight'] = best_params[1]
    
    
    ############################# Next parameter ###############################
    
    # This can take some time…
    min_mae = float("Inf")
    best_params = None
    for eta in [.3, .2, .1, .05, .01, .005]:
        print("CV with eta={}".format(eta))
        print(params)
    
        # We update our parameters
    
        params['eta'] = eta
    
        # Run and time CV
    
        cv_results = xgb.cv(params, dtrain, num_boost_round=num_boost_round, seed=42, nfold=5, metrics=['mae'], early_stopping_rounds=10)
        # Update best score
        mean_mae = cv_results['test-mae-mean'].min()
        boost_rounds = cv_results['test-mae-mean'].argmin()
        print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))
        if mean_mae < min_mae:
            min_mae = mean_mae
            best_params = eta
    print("Best params: {}, MAE: {}".format(best_params, min_mae))
    params['eta'] = best_params
    
    ############################# Testing next two parameters ############################

    gridsearch_params = [
        (subsample, colsample)
        for subsample in [i/10. for i in range(7,11)]
        for colsample in [i/10. for i in range(7,11)]
    ]

    min_mae = float("Inf")
    best_params = None
    # We start by the largest values and go down to the smallest
    for subsample, colsample in reversed(gridsearch_params):
        print("CV with subsample={}, colsample={}".format(subsample, colsample))
    
        # We update our parameters
        params['subsample'] = subsample
        params['colsample_bytree'] = colsample
        # Run CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics={'mae'},
            early_stopping_rounds=10
        )
        # Update best score
        mean_mae = cv_results['test-mae-mean'].min()
        boost_rounds = cv_results['test-mae-mean'].argmin()
        print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
        if mean_mae < min_mae:
            min_mae = mean_mae
            best_params = (subsample,colsample)
    print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))
    
    params['subsample'] = best_params[0]
    params['colsample_bytree'] = best_params[1]
    
    return params
    
    
    


# In[17]:


def prettyprintmatrix(pretuned_metric, posttuned_metric):
    print("pretuned: {} vs postuned: {}".format(pretuned_metric, posttuned_metric))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




