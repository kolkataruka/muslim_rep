import os
import pandas as pd
import numpy as np

import pickle

from collections import Counter


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
#import gc

#import tensorflow as tf
#from tensorflow.keras.layers import Activation, AlphaDropout, Dense, BatchNormalization,\
#SpatialDropout1D,Dropout, GlobalMaxPool1D,Concatenate,Input,Conv1D, Embedding
#from tensorflow.keras.models import Model
#from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
#from tensorflow.keras.models import load_model

#import matplotlib.pyplot as plt
from sklearn.utils import class_weight

loaded_model = pickle.load(open("model_2class_svm_concat_True.sav", 'rb'))
loaded_vectorizer = pickle.load(open("vectorizer_2class_svm_concat_True.sav", "rb"))

assembly_data = pd.read_csv('all_states_assembly.csv')[['State_Name', 'Constituency_No', 'Year', 'Position', 'Candidate', 'Party', 'Candidate_Type', 'Election_Type']]
general_elections_data = pd.read_csv('all_states_general.csv')[['State_Name', 'Constituency_No', 'Year', 'Position', 'Candidate', 'Party', 'Candidate_Type', 'Election_Type']]

def clean_cand_names(df):
    df.Candidate = df.Candidate.str.upper()
    df['Candidate'].replace('2', '', regex=True, inplace=True)
    df['Candidate'].replace(',', '', regex=True, inplace=True)
    df['Candidate'].replace('[)(\'/;034`5]', '', regex=True, inplace=True)
    df['Candidate'].replace('[\x9f9\x9f]','', regex=True, inplace=True)
    df['Candidate'].replace('-','', regex=True, inplace=True)                     
    df['Candidate'].replace('\\\\', '', regex=True, inplace=True)      
    df['Candidate'].replace('\[', '', regex=True, inplace=True)                     
    df['Candidate'].replace('\]', '', regex=True,inplace=True) 
    df.Candidate.replace("\.", " ", regex=True, inplace=True)
    df.Candidate.replace("\s+", " ", regex=True, inplace=True)
    df.Candidate = df["Candidate"].str.strip()
    df = df[df["Candidate"].notna()]
    df = df[df["Candidate"] != ""]
    df.Candidate.replace(" ", "}{", regex=True, inplace=True)
    df.Candidate = "{" + df.Candidate.astype(str) + "}"
    #df.Candidate = '#' + df.Candidate.astype(str) + '#'
    return df

def reclean_names(df):
    df.Candidate.replace("}{", " ", regex=True, inplace=True)
    df['Candidate'] = df["Candidate"].apply(lambda x: x.strip("{}"))
    return df


clean_assembly = clean_cand_names(assembly_data)
clean_general_elections = clean_cand_names(general_elections_data)

tfidf_matrix_assembly = loaded_vectorizer.transform(clean_assembly['Candidate'].values.tolist())
tfidf_matrix_gen = loaded_vectorizer.transform(clean_general_elections['Candidate'].values.tolist())

# Make predictions using the loaded model
assembly_predictions = loaded_model.predict(tfidf_matrix_assembly)
gen_predictions = loaded_model.predict(tfidf_matrix_gen)


clean_assembly['Predicted community'] = assembly_predictions
clean_general_elections['Predicted community'] = gen_predictions
clean_assembly['Predicted community'] = clean_assembly['Predicted community'].map({1: 'Muslim', 0: 'Non-muslim'})
clean_general_elections['Predicted community'] = clean_general_elections['Predicted community'].map({1: 'Muslim', 0: 'Non-muslim'})

cleaned_assembly = reclean_names(clean_assembly)
cleaned_general_elections = reclean_names(clean_general_elections)

print_assembly = cleaned_assembly.to_csv("assembly_with_preds.csv")
print_general = cleaned_general_elections.to_csv('general_with_preds.csv')

