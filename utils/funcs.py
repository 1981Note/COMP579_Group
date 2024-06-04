# Reference: https://github.com/AI4Finance-Foundation/FinRL/blob/master/finrl/main.py

from __future__ import annotations

import os
import pandas as pd 
import numpy as np
import shutil

# "./" will be added in front of each directory
def check_and_make_directories(directories: list[str]):
    for directory in directories:
        full_path = "./" + directory
        if os.path.exists(full_path):
            shutil.rmtree(full_path)
        os.makedirs(full_path)
            
            
def process_df_for_mvo(df, stock_dimension):
  df = df.sort_values(['date','tic'],ignore_index=True)[['date','tic','close']]
  fst = df
  fst = fst.iloc[0:stock_dimension, :]
  tic = fst['tic'].tolist()

  mvo = pd.DataFrame()

  for k in range(len(tic)):
    mvo[tic[k]] = 0

  for i in range(df.shape[0]//stock_dimension):
    n = df
    n = n.iloc[i * stock_dimension:(i+1) * stock_dimension, :]
    date = n['date'][i*stock_dimension]
    mvo.loc[date] = n['close'].tolist()

  return mvo

# Helper functions for mean returns and variance-covariance matrix
# Codes in this section partially refer to Dr G A Vijayalakshmi Pai
# https://www.kaggle.com/code/vijipai/lesson-5-mean-variance-optimization-of-portfolios/notebook

def StockReturnsComputing(StockPrice, Rows, Columns):
  StockReturn = np.zeros([Rows-1, Columns])
  for j in range(Columns):        # j: Assets
    for i in range(Rows-1):     # i: Daily Prices
      StockReturn[i,j]=((StockPrice[i+1, j]-StockPrice[i,j])/StockPrice[i,j])* 100

  return StockReturn