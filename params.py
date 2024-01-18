# params.py - parameters configuration file
# @author - Alejandro Granizo Castro (alejandro.granizo.castro@alumnos.upm.es)
# Thesis - Enhancing accuracy of estimators for financial options using importance sampling procedures: 
#           A practical approach for European call options on Euro Stoxx 50 Index

#Imports
import numpy as np

#Number of samples
R = 250
#Size of one sample
sample_size = 4200
#Strike price set for the options
strike_price = 3000
#Strike date for the options, in days from initial price
strike_date = 400
#interest rate per day to discount the payoff
interest_rate = 0.0082 #3% interest per year/365=0.0082
#initial step size for FD method
step_size = 20e-11
#delta value for FD method - based on Asmussen & Glynn
delta = np.power(R,-(1/6))
#K - number of batch elements to FD method
K = 10
#early stop iterations for FD method
earlyStop = 20
#seed value for the random generation of samples
seed = 42

#Parameters for experimental comparison

#Number of samples for comparison
RComparison = [250,500,1000,1500,2000,2500,3000]
#Different values of delta for comparison
deltaComparison = [delta*0.8, delta*0.9, delta*0.95, delta, delta*1.05, delta*1.1, delta*1.2]