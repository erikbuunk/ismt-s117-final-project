# -*- coding: utf-8 -*-
"""
Base on the 04 and 05 notebeooks and put into a script that can 
be run on the server
"""
print("Setup...")
# setup
import sys
import subprocess
import pkg_resources
import re

# required = {'spacy', 'scikit-learn', 'numpy',
#             'pandas', 'torch', 'matplotlib',
#             'transformers', 'allennlp==0.9.0'}

# installed = {pkg.key for pkg in pkg_resources.working_set}
# missing = required - installed

# if missing:
#     python = sys.executable
#     subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)

import numpy as np
import pandas as pd

# PyTorch
import torch

# SciKit Learn
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# File managment
import os
from os import listdir
from pathlib import Path
import pickle
import gzip


## SETTINGS --------------------------------
path = "./data/"
genres = ["Rock", "Hip Hop", "Metal"] # possible ["Rock", "Hip Hop", "Metal", "Pop"]
n = 12000 # size of the data set to use
test_size = 0.3


# Helper functions ----------------------------

def create_subset(df, n, genres):

    df_temp = pd.DataFrame(columns=list(df.columns))

    for g in genres:

      df_small = df.query(f"Genre == '{g}'")
      df_genre = df_small.sample(n=round(n/len(genres)), replace="False", random_state=42)
      df_temp = df_temp.append(df_genre)

    return df_temp

def print_confustion_matrix(model, y_val_set, predictions):
  cm = confusion_matrix(y_val_set, predictions)
  df = pd.DataFrame(cm, columns = model.classes_, index= model.classes_)
  print(df)

def wrong_classifications(X_train, y, predictions, genres):
  print("Truth - predicted")
  predictions_df = pd.DataFrame(predictions, columns = ["Genre_Predicted"])
  truth_df = pd.DataFrame(y)
  truth_df.columns = ["Genre_Truth"]
  combined_df = pd.concat([truth_df.reset_index(drop=True), predictions_df.reset_index(drop=True)], axis=1)
  for i in genres:
    for j in genres:
      if i!=j:
        idx = combined_df.query(f"Genre_Truth =='{i}' != Genre_Predicted=='{j}'").index
        if len(idx)>0:
          print("------------------------------")
          print(f"{i} - {j}")
          print("------------------------------")
          print(X_train.iloc[idx]["Lyric"])




# Prepare data  ------------------------------------------
def prep_data():
    print("Preparing data...")
    """# Load Data"""

    # Load data
    file_name = "df_total_cleaned"
    df_total_cleaned = pd.read_pickle(f'{path}{file_name}.pkl.gz')

    """## Combine with BERT vectors"""
    file_name = "lyrics_bert_vectors_total"
    df_bert = pd.DataFrame(pd.read_pickle(f'{path}{file_name}.pkl.gz'))


    # Combine the two sets
    df_total  = pd.concat([df_total_cleaned, df_bert], axis=1)

    """## Create Training and Test Set"""

    df_subset = create_subset(df_total, n, genres)
    df_subset["Genre"].value_counts()

    X=df_subset.drop("Genre", axis = 1)
    y=df_subset["Genre"]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0, shuffle = True, stratify = y)

    #pickle.dump(X_train, gzip.open(f'{path}X_train.pkl.gz', 'wb'))
    #pickle.dump(X_test, gzip.open(f'{path}X_test.pkl.gz', 'wb'))
    #pickle.dump(y_train, gzip.open(f'{path}y_train.pkl.gz', 'wb'))
    #pickle.dump(y_test, gzip.open(f'{path}y_test.pkl.gz', 'wb'))
    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, y_train, y_test):
    # Model and Prediction --------------------------------
    # Train model ------------------------------------------
    print("Training model...")
    text_cols = ["SName", "Lyric", "Artist"]
    genres = list(pd.DataFrame(y_train)["Genre"].unique())

    test_size = 0.3
    tmp = X_train.drop(text_cols,axis=1)

    X_train_final = X_train.drop(text_cols, axis=1)


    model = SVC(max_iter=10000, kernel='rbf', C=5)
    model.fit(X_train_final, y_train)

    # save model
    # add _new so the original model will not be overwritten without first viewing the results
    pickle.dump(model, gzip.open(f'{path}{"final_lyrics_model_new.pkl.gz"}', 'wb'))

    X_test_final = X_test.drop(text_cols, axis=1)
    test_predictions = model.predict(X_test_final)
    return (model, test_predictions )

def show_metrics(model, test_predictions):
    # Check metrics
    acc = accuracy_score(y_test, test_predictions)
    print(f"Accuracy: {round(acc,2)}")

    print_confustion_matrix(model, y_test, test_predictions)

    print(classification_report(y_test, test_predictions, target_names=genres))

    wrong_classifications(X_train, y_test, test_predictions , genres)

# main function -------------------------------------
#  def retrain_model():
(X_train, X_test, y_train, y_test) = prep_data()
(model, test_predictions) = train_model(X_train, X_test, y_train, y_test)
show_metrics(model, test_predictions)
