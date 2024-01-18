# analysis.py - executes the different experiments described in the paper
# @author - Alejandro Granizo Castro (alejandro.granizo.castro@alumnos.upm.es)
# Thesis - Enhancing accuracy of estimators for financial options using importance sampling procedures: 
#           A practical approach for European call options on Euro Stoxx 50 Index

import auxiliar as aux
import scipy as sc
import numpy as np


#Estimation using CMC and IS under standard conditions
def experiment1(data,Yi,R,seed,sample_size,delta):
    #2 - Adjust to lognormal - obtain fitted mu,sigma
    mu,sigma = aux.lognormal_fitting(Yi)

    #3 - Generate R samples of Ythota with fitted values
    YiThota = []
    #Set the seed
    seed_value = seed
    np.random.seed(seed_value)
    # Generate a sequence of 5 random numbers
    random_numbers = np.random.rand(R)

    #Generate R sampples
    for i in range(0,R):
        YiThota.append(sc.stats.lognorm.rvs(sigma, scale=np.exp(mu), size=sample_size,random_state=int(np.round(random_numbers[i]*1000))))


    #4 - Use FD method to estimate mu
    delta = delta
    muEst = aux.muFDEst_newBatch(YiThota, mu, sigma, delta, data[0])
    sigmaEst = sigma #We consider sigma fix parameter to FD method with 1 dimension

    #5 - Generate R samples of Ythota with IS values
    YiThotaEst = []
    for i in range(0,R):
       YiThotaEst.append(sc.stats.lognorm.rvs(sigmaEst, scale=np.exp(muEst), size=sample_size))

    #6 - Calculate the sample average and variance with likelihood ratio - for samples with CMC and IS values 
    rCrude = []
    for sample in YiThota:
       rCrude.append(aux.option_return(sample,data[0]))
    rEst = []
    params =[[mu,sigma],[muEst,sigmaEst]] 
    for sample in YiThotaEst:
       rEst.append(aux.option_return_IS(sample,data[0],params))

    #7 - Calculate the expected value and the variance of the samples
    true_sample_average = np.mean(rCrude)
    true_sample_var = np.var(rCrude)

    estimator_sample_average = np.mean(rEst)
    estimator_sample_var = np.var(rEst)

    #8 - Calculate confidence intervals
    conf_interval_crude = aux.calculate_confidence_interval(data, true_sample_average, true_sample_var)
    conf_interval_is = aux.calculate_confidence_interval(data, estimator_sample_average, estimator_sample_var)

    #9 - Return values
    return (muEst,[[true_sample_average,true_sample_var,conf_interval_crude],[estimator_sample_average,estimator_sample_var,conf_interval_is]])

#Comparison for convergence on deltas and effect in accuracy
def experiment2(data,Yi,R,seed,sample_size,deltaComparison):
    #2 - Adjust to lognormal - obtain fitted mu,sigma
    mu,sigma = aux.lognormal_fitting(Yi)

    #3 - Generate R samples of Ythota with fitted values
    YiThota = []
    #Set the seed
    seed_value = seed
    np.random.seed(seed_value)
    # Generate a sequence of 5 random numbers
    random_numbers = np.random.rand(R)

    #Generate R sampples
    for i in range(0,R):
        YiThota.append(sc.stats.lognorm.rvs(sigma, scale=np.exp(mu), size=sample_size,random_state=int(np.round(random_numbers[i]*1000))))


    #4 - Use FD method to estimate mu
    muEst = []
    for delta in deltaComparison:
        muEstim = aux.muFDEst_newBatch(YiThota, mu, sigma, delta, data[0])
        muEst.append(muEstim)
    sigmaEst = sigma #We consider sigma fix parameter to FD method with 1 dimension

    avgs_cmc = []
    avgs_is = []
    vars_cmc = []
    vars_is = []
    conf_intervals_cmc = []
    conf_intervals_is = []
    #5 - Generate R samples of Ythota with IS values
    for muEstim in muEst:
        YiThotaEst = []
        for i in range(0,R):
          YiThotaEst.append(sc.stats.lognorm.rvs(sigmaEst, scale=np.exp(muEstim), size=sample_size))

        #6 - Calculate the sample average and variance with likelihood ratio - for samples with CMC and IS values 
        rCrude = []
        for sample in YiThota:
            rCrude.append(aux.option_return(sample,data[0]))
        rEst = []
        params =[[mu,sigma],[muEstim,sigmaEst]] 
        for sample in YiThotaEst:
            rEst.append(aux.option_return_IS(sample,data[0],params))

        #7 - Calculate the expected value and the variance of the samples
        true_sample_average = np.mean(rCrude)
        true_sample_var = np.var(rCrude)

        estimator_sample_average = np.mean(rEst)
        estimator_sample_var = np.var(rEst)

        #8 - Calculate confidence intervals
        conf_interval_crude = aux.calculate_confidence_interval(data, true_sample_average, true_sample_var)
        conf_interval_is = aux.calculate_confidence_interval(data, estimator_sample_average, estimator_sample_var)

        #Append results for each metric
        avgs_cmc.append(true_sample_average)
        avgs_is.append(estimator_sample_average)
        vars_cmc.append(true_sample_var)
        vars_is.append(estimator_sample_var)
        conf_intervals_cmc.append(conf_interval_crude)
        conf_intervals_is.append(conf_interval_is)
    return [muEst,[avgs_cmc,avgs_is],[vars_cmc,vars_is],[conf_intervals_cmc,conf_intervals_is]]

#Comparison between whole new batches vs 1 substitution per batch
def experiment3(data,Yi,R,seed,sample_size,delta):
    #2 - Adjust to lognormal - obtain fitted mu,sigma
    mu,sigma = aux.lognormal_fitting(Yi)

    #3 - Generate R samples of Ythota with fitted values
    YiThota = []
    #Set the seed
    seed_value = seed
    np.random.seed(seed_value)
    # Generate a sequence of 5 random numbers
    random_numbers = np.random.rand(R)

    #Generate R sampples
    for i in range(0,R):
        YiThota.append(sc.stats.lognorm.rvs(sigma, scale=np.exp(mu), size=sample_size,random_state=int(np.round(random_numbers[i]*1000))))


    #4 - Use FD method to estimate mu
    delta = delta
    muEst = aux.muFDEst_substitution(YiThota, mu, sigma, delta, data[0])
    sigmaEst = sigma #We consider sigma fix parameter to FD method with 1 dimension

    #5 - Generate R samples of Ythota with IS values
    YiThotaEst = []
    for i in range(0,R):
       YiThotaEst.append(sc.stats.lognorm.rvs(sigmaEst, scale=np.exp(muEst), size=sample_size))

    #6 - Calculate the sample average and variance with likelihood ratio - for samples with CMC and IS values 
    rCrude = []
    for sample in YiThota:
       rCrude.append(aux.option_return(sample,data[0]))
    rEst = []
    params =[[mu,sigma],[muEst,sigmaEst]] 
    for sample in YiThotaEst:
       rEst.append(aux.option_return_IS(sample,data[0],params))

    #7 - Calculate the expected value and the variance of the samples
    true_sample_average = np.mean(rCrude)
    true_sample_var = np.var(rCrude)

    estimator_sample_average = np.mean(rEst)
    estimator_sample_var = np.var(rEst)

    #8 - Calculate confidence intervals
    conf_interval_crude = aux.calculate_confidence_interval(data, true_sample_average, true_sample_var)
    conf_interval_is = aux.calculate_confidence_interval(data, estimator_sample_average, estimator_sample_var)

    #9 - Return values
    return (muEst,[[true_sample_average,true_sample_var,conf_interval_crude],[estimator_sample_average,estimator_sample_var,conf_interval_is]])    

#comparison between different levels of samples
def experiment4(data,Yi,RComparison,seed,sample_size,delta):
    results = []
    for R in RComparison:
        mu,res = experiment1(data,Yi,R,seed,sample_size,delta)
        results.append(res)
    return results