# DataExt.py - Contains DataExt class for the data gathering from Yahoo Finance
# @author - Alejandro Granizo Castro (alejandro.granizo.castro@alumnos.upm.es)
# Thesis - Enhancing accuracy of estimators for financial options using importance sampling procedures: 
#           A practical approach for European call options on Euro Stoxx 50 Index

#----------------IMPORTS-----------------------
import yfinance as API #yahoo finance API
import datetime
import os
import pandas as pd


#--------------CLASS DEFINITION----------------
class DataExt:
    #-------------------ATTRIBUTES-----------------------------------
    #Saves the ticker (stock symbol) of the aimed stock - SP&500 as default
    ticker = "" 

    # start and end date - retrieving data from last 20 years
    # Starting date - 20 years from 2023-11-01 08:00:00) -> Efectively is 2003-11-01 08:00:00)
    # If data not available that far will get less data, but not fail
    start_date = datetime.datetime(2003, 11, 1, 8, 0, 0)
    # Ending date set to present day and moment
    end_date = datetime.datetime.now() 

    #Variable set as True if the data file already exists - False as default
    exist = False
#------------------------CONSTRUCTOR METHODS------------------------------
    #Default constructor method
    def __init__(self):
        self.ticker = "^GSPC" #Default ticker, S&P 500 index
        if(os.path.exists(self.ticker+".csv")):
            self.exist = True
        
        
    #constructor using a specific ticker
    def __init__(self,tck):
        self.ticker = tck
        if(os.path.exists(self.ticker+".csv")):
            self.exist = True

#------------------CLASS PUBLIC METHODS---------------
    #returns the stock data requested and the number of entries that the data has    
    def getData(self):
        #If the file already exists, check if needs update or not,
        if(self.exist):
            data = pd.read_csv(self.ticker + ".csv")
            length = len(data)
            return data,length
        
        #If the file does not exists, it retrieves the data and returns the data and the number of entries
        #Also creates the file
        else:
            #Warning - needs to be connected to a VPN if internet has any special firewall (e.g. China)
            data = API.download(self.ticker, start=self.start_date, end=self.end_date) #downloads data
            data.to_csv(self.ticker+".csv")
            length = len(data) #gets the number of entries of the data
            return data,length #return data(csv format) and number of entries
