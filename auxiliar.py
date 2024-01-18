# auxiliar.py - auxiliar methods to estimations
# @author - Alejandro Granizo Castro (alejandro.granizo.castro@alumnos.upm.es)
# Thesis - Enhancing accuracy of estimators for financial options using importance sampling procedures: 
#           A practical approach for European call options on Euro Stoxx 50 Index

#Library imports
import numpy as np
import scipy.stats as stats
from DataExt import DataExt
import params

#Take parameters from params.py
strike_price = params.strike_price
strike_date = params.strike_date
interest_rate = params.interest_rate

#Auxiliar functions

#Gathers the data making use of DataExt.py class
#Receives the ticker of the underlying asset
#Returns the closing price series for the ticker
# Warning - make sure yahoo finance is operative in region you are executing the code
def dataGathering(ticker):
    raw_data = DataExt(ticker)
    data, length = raw_data.getData()
    return data['Close']

#Creates and fills the variable Y of price change ratios from day to day
#Receives a data series of an asset price
#Return the series of price change ratios
def create_Yi_model(data):
    Yi = []

    for i in range(1, len(data)):
        Yi.append(data[i] / data[i - 1])
    return Yi

#Fits the data inputed into a lognormal distribution using MLE method
#Receives the input data to fit to a lognormal distribution
#Return the fitted to lognormal values for mu and sigma
def lognormal_fitting(Yi):
    params = stats.lognorm.fit(Yi,floc=0)
    #obtain the following parameters - From the lognormal fitting
    #   Shape = sigma -> shape=std(log(x)) -only in LOGNORMAL FIT
    #   Location = 0 - set to 0 because we do not need to shift the distribution
    #   Scale = exp(mu) -> scale=exp⁡(mean(log⁡(x))) -only in LOGNORMAL FIT
    sigma, loc, scale= params
    #We obtain mu from scale as scale = exp(mu)
    mu = np.log(scale)
    return mu,sigma

#Computes the price on the strike date based on the sampple path inputed
#Receives the sample path (Y variable) and the initial price for t=0
#Returns the price of the asset on strike date based on the sample path
def priceOnStrike(sample,initPrice):
    price = initPrice
    for i in range(0,strike_date):
        price = price*sample[i]
    return price

#Obtains the likelihood ratio of a sample path based on two lognormal distributions whose parameters are inputed
#Receives sample path and parameters [[muTarget,sigmaTarget],[muImportance,sigmaImportance]]
#Returns the likelihood ratio of the sample path using the target and importance distributions described
def obtainLR(sample,params):
    mu,sigma = params[0]
    muEst,sigmaEst = params[1]
    #Alerts of possible error when obtaining likelihood ratio and avoids crash by returning last mu
    #For fixing check mu value
    if(stats.lognorm.pdf(sample[0], sigmaEst, scale=np.exp(muEst))==0):
        print("next step throws error because df is 0 in denominator")
        return muEst
    likelihood_ratio = stats.lognorm.pdf(sample[0], sigma, scale=np.exp(mu))/stats.lognorm.pdf(sample[0], sigmaEst, scale=np.exp(muEst))
    for i in range(1,strike_date):
        likelihood_ratio *= stats.lognorm.pdf(sample[i], sigma, scale=np.exp(mu))/stats.lognorm.pdf(sample[i], sigmaEst, scale=np.exp(muEst))
    return likelihood_ratio


#Computes the payoff of the option discounted to the present day at set interest_rate
#Receives sample path and initial price for the option
#Returns the payoff, positive or 0 for the option
def option_return(sample,initPrice):    
    #calculate price in strike day using sample and initial price
    price = priceOnStrike(sample,initPrice)
    #the return is the max between 0 or the benefit obtained
    return_on_strike_date = 0
    if(price-strike_price>0):
        return_on_strike_date = price-strike_price
    
    return np.exp(-interest_rate*strike_date)*return_on_strike_date

#Computes the payoff of the option using Importance Sampling at interest rate
#Receives sample path, initial price and target and importance distribution parameters as
#   [[muTarget,sigmaTarget],[muImportance,sigmaImportance]]
#Returns the payoff, positive or 0, or the option with the importance sampling method
def option_return_IS(sample,initPrice,params):
    #calculate price in strike day using sample and initial price
    price = priceOnStrike(sample,initPrice)
    #the return is the max between 0 or the benefit obtained
    return_on_strike_date = 0
    if((price-strike_price)>0):
        return_on_strike_date = price-strike_price
    #Obtain the likelihood ratio
    likelihood_ratio = obtainLR(sample,params)
    
    return ((np.exp(-interest_rate*strike_date)*return_on_strike_date))*likelihood_ratio

#Computes the inside term of the finite differences method - called in muFDEst
#Receives sample path, initial price and target and importance distribution parameters as
#   [[muTarget,sigmaTarget],[muImportance,sigmaImportance]]
#Returns the payoff squared * likelihood ratio, according to expression 27 of the paper
def option_ret_LR(sample,initPrice,params):
    #calculate price in strike day using sample and initial price
    price = priceOnStrike(sample,initPrice)
    #the return is the max between 0 or the benefit obtained
    return_on_strike_date = 0
    if((price-strike_price)>0):
        return_on_strike_date = price-strike_price
    #Obtain the likelihood ratio
    likelihood_ratio = obtainLR(sample,params)
    #Return r^2*LR 

    return ((np.exp(-interest_rate*strike_date)*return_on_strike_date)**2)*likelihood_ratio

#Calculates the approximation of mu for the importance distribution
#Receives the sample paths generated, mu and sigma fitted, delta value and initial price
#Returns the value of fitted mu with new batch each iteration
def muFDEst_newBatch(paths, mu, sigma, deltaArg, init_price):
    #Initial mu is set as fitted mu to begin
    muEst=mu
    #Recover size for batch
    K = params.K
    delta = deltaArg*(10**-5)#change the order to fit the initial mu order
    step = 1
    step_size = params.step_size
    previous_mu = 9999
    #For each sample path
    for i in range(0,3000):
        plusK =0
        minusK =0
        #Each iteration introduces new batch of K elems
        for u in range(i*K,(i*K)+K):
            params1 = [[mu,sigma],[(muEst+delta),sigma]]
            #Computes with mu+delta
            plusDelta = option_ret_LR(paths[u],init_price,params1)

            #Same as in previous case
            params2 = [[mu,sigma],[(muEst-delta),sigma]]
            #Computes with mu-delta
            minusDelta = option_ret_LR(paths[u],init_price,params2)
            plusK+=plusDelta
            minusK+=minusDelta
        
        #For each iteration we sum the difference
        previous_mu = muEst
        muEst -= step_size*((plusK-minusK)/(2*delta*K))
        #According to Su & Fu
        step_size *= np.power(step,(-3/4))
        step +=1
        #Conditions to stop loop because convergence is obtained
        if((previous_mu - muEst < 10e-8) and i>params.earlyStop):
            break
    #Returns final value of muEst
    return muEst

#Calculates the approximation of mu for the importance distribution
#Receives the sample paths generated, mu and sigma fitted, delta value and initial price
#Returns the value of fitted mu with new batch each iteration
def muFDEst_substitution(paths, mu, sigma, deltaArg, init_price):
    #Initial mu is set as fitted mu to begin
    muEst=mu
    #Recover size for batch
    K = params.K
    delta = deltaArg*(10**-5)#change the order to fit the initial mu order
    step = 1
    step_size = params.step_size
    previous_mu = 9999
    #For each sample path
    for i in range(0,3000):
        plusK =0
        minusK =0
        #Each iteration introduces new batch of K elems
        for u in range(0,K):
            params1 = [[mu,sigma],[(muEst+delta),sigma]]
            #Computes with mu+delta
            plusDelta = option_ret_LR(paths[u+i],init_price,params1)

            #Same as in previous case
            params2 = [[mu,sigma],[(muEst-delta),sigma]]
            #Computes with mu-delta
            minusDelta = option_ret_LR(paths[u+i],init_price,params2)
            plusK+=plusDelta
            minusK+=minusDelta
        
        #For each iteration we sum the difference
        previous_mu = muEst
        muEst -= step_size*((plusK-minusK)/(2*delta*K))
        #According to Su & Fu
        step_size *= np.power(step,(-3/4))
        step +=1
        #Conditions to stop loop because convergence is obtained
        if((previous_mu - muEst < 10e-8) and i>params.earlyStop):
            break
    #Returns final value of muEst
    return muEst

#Calculation of confidence intervals on 95% confidence for estimation
#Receives the estimation (data), the sample expectation (mean) and the sample variance
#Returns the limits of the confidence interval on 95%
def calculate_confidence_interval(data, mean, variance, confidence_level=0.95):
    std_dev = np.sqrt(variance)
    n = len(data)

    # Calculate the standard error of the mean
    sem = std_dev / np.sqrt(n)

    # Calculate the z-score for the desired confidence level
    z_score = stats.norm.ppf((1 + confidence_level) / 2)

    # Calculate the margin of error
    margin_of_error = z_score * sem

    # Calculate the confidence interval
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error

    return (lower_bound, upper_bound)
