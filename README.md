In this project, we aim to explore the application of various deep reinforcement learning (DRL) algorithms in stock trading strategies. This project is to reproduce and re-implement the papers "Practical Deep Reinforcement Learning Approach for Stock Trading" and "Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy." The code is based on the FinRL repository: https://github.com/AI4Finance-Foundation/FinRL

# Folders Description:
## data_processors
Contains essential python files for processing data. 
#### data_processor.py: 
Code for downloading general data and adding technical indicators and other useful information to the dataframe. 
#### processor_alpaca.py: 
Provides methods for retrieving daily stock data when using Alpaca API. 
#### processor_wrds.py: 
Provides methods for retrieving daily stock data when using Wharton Research Data Services. 
#### processor_yahoofinance.py: 
Provides methods for retrieving daily stock data from Yahoo Finance API. 

## env_stock_trading
#### env_stocktrading.py: 
A stock trading environment. 


## models
Contains python files for DRL agents. 
#### DRLAgent.py: 
Provides implementations for DRL algorithms using single agents. 
#### DRLEnsembleAgent.py: 
Provides implementations for DRL algorithms using ensemble method. 


## preprocessor
Contains python files for data preprocessing. 
#### preprocessors.py: 
Provides methods and FeatureEngineer class for preprocessing the stock price data. 
#### yahoodownloader.py: 
Contains methods and classes to collect data from Yahoo Finance API. 

## DQN
Contains sample code to explore the application of deep Q-learning in stock trading strategies.


## Reimplement_Comp579_Project_final_version.ipynb  
This notebook implements stock trading strategies using both individual DRL agents and an ensemble approach.

## Reproduce_FinRL_Ensemble_StockTrading_ICAIF_2020.ipynb
This notebook is the original implementation provided by the authors of the paper "Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy" It serves as a reference for reproducing the results and methodologies discussed in the paper. 

## Reproduce_Stock_NeurIPS2018_SB3.ipynb
This notebook is the original implementation provided by the authors of the paper "Practical Deep Reinforcement Learning Approach for Stock Trading" It serves as a reference for reproducing the results and methodologies discussed in the paper. 