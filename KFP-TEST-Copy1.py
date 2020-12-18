#!/usr/bin/env python
# coding: utf-8

# In[22]:


from helper_components import hypertune
from helper_components import preprocessingstep
from helper_components import model_training
from helper_components import prettyprintmatrix


# In[1]:


import kfp


# In[112]:





# In[114]:





# In[81]:





# In[23]:


import os

#from helper_components import evaluate_model
#from helper_components import retrieve_best_run
from jinja2 import Template
import kfp
from kfp.components import func_to_container_op
from kfp.dsl.types import Dict
from kfp.dsl.types import GCPProjectID
from kfp.dsl.types import GCPRegion
from kfp.dsl.types import GCSPath
from kfp.dsl.types import String
from kfp.gcp import use_gcp_secret

    
from typing import NamedTuple
import pandas as pd
#def preprocessingstep(train_file, test_file) -> NamedTuple('Outputs', [('data_train', pd.DataFrame), ('data_test', pd.DataFrame), ('target_train', pd.DataFrame), ('target_test', pd.DataFrame)]):

#def preprocessingstep(bucket_name, train_file, test_file) -> NamedTuple('Outputs', [('resultant_file', str)]): 

    



    
from kfp.components import func_to_container_op
BASE_IMAGE="docker.io/mkbansal0588/kubeflow"
preprocessingstep_op = func_to_container_op(preprocessingstep, base_image=BASE_IMAGE)
model_training_op = func_to_container_op(model_training, base_image=BASE_IMAGE)
hypertune_op = func_to_container_op(hypertune, base_image=BASE_IMAGE)
prettyprint_op = func_to_container_op(prettyprintmatrix, base_image=BASE_IMAGE)
    
@kfp.dsl.pipeline(
    name='Housing Pricing preidiction',
    description='The pipeline for training and deploying the house price prediction algorithm'
)
def Pipelinetest():
    bucket_name = 'mohittest0432'
    train_file = "train.csv"
    test_file = "test.csv"
    
    preprocessingstepxyz = preprocessingstep_op(bucket_name, train_file, test_file)
    
    step1_processed_file = preprocessingstepxyz.outputs['Output']
    
    params: dict = {
        # Parameters that we are going to tune.
        'max_depth':6,
        'min_child_weight': 1,
        'eta':.3,
        'subsample': 1,
        'colsample_bytree': 1,
        # Other parameters
        'objective':'reg:squarederror',
    }
    
    modeltrainingxyx = model_training_op(bucket_name, step1_processed_file, params)
    
    pretuned_metric = modeltrainingxyx.outputs['Output']
    
    hypertunexyz = hypertune_op(bucket_name, step1_processed_file, params)
    
    hypertuned_params = hypertunexyz.outputs['Output']
    
    fineTunedmodeltrainingxyz = model_training_op(bucket_name, step1_processed_file, hypertuned_params)
    
    posttuned_metric = fineTunedmodeltrainingxyz.outputs['Output']
    
    prettyprintmatrixxyz = prettyprint_op(pretuned_metric, posttuned_metric)
    
    
    
    


# In[ ]:





# In[72]:


#bigquery_query_op = component_store.load_component('bigquery/query')


# In[ ]:





# In[ ]:





# In[1]:





# In[47]:


#!jupyter nbconvert --to script /Users/mohit.k.bansal/kubeflow/pipeline/MohitTest/KFP-TEST-Copy1.ipynb


# In[49]:


#!dsl-compile --py /Users/mohit.k.bansal/kubeflow/pipeline/MohitTest/KFP-TEST-Copy1.py --output /Users/mohit.k.bansal/kubeflow/pipeline/MohitTest/KFP-TEST-Copy1.yaml


# In[ ]:





# In[2]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[45]:





# In[47]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




