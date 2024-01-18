# main.py - main executions for the experimental part
# @author - Alejandro Granizo Castro (alejandro.granizo.castro@alumnos.upm.es)
# Thesis - Enhancing accuracy of estimators for financial options using importance sampling procedures: 
#           A practical approach for European call options on Euro Stoxx 50 Index

#Imports needed
import numpy as np
import auxiliar as aux
import params
import analysis
import matplotlib.pyplot as plt

# Experiments

#Take parameters from params.py
R = params.R
RComparison = params.RComparison
seed = params.seed
sample_size = params.sample_size
delta = params.delta
deltaComparison = params.deltaComparison



#0 - Get the data and the distribution Y
data = aux.dataGathering("^STOXX50E")
Yi = aux.create_Yi_model(data)

#--------------------------------------------------------------------------------------------------
# #1 - comparison under standard circumstances - refer to paper section experimental configuration
exp1 = analysis.experiment1(data,Yi,R,seed,sample_size,delta)
    #Separate part for experiment 3 and results of experiment 1
exp3_newBatch_mu, exp1_results = exp1
    #Results for Crude Monte Carlo and Importance Sampling
exp1_cmc_res, exp1_is_res = exp1_results

#Show results 

#Mean and variance
fig, ax = plt.subplots()
table_data = [["Method", "Expected Value", "Variance"],
              ["CMC", exp1_cmc_res[0], exp1_cmc_res[1]],
              ["IS", exp1_is_res[0], exp1_is_res[1]]]

exp1_cell_colors = [['#D3D3D3', '#D3D3D3', '#D3D3D3'], ['#D3D3D3', '#FFFFFF', '#FFFFFF'], ['#D3D3D3', '#FFFFFF', '#FFFFFF']]


table = ax.table(cellText=table_data, cellLoc='center', loc='center', cellColours=exp1_cell_colors)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
ax.axis('off')

plt.show(block=False)

# Plotting confidence intervals
fig, ax = plt.subplots()

# Plot confidence intervals
ax.plot(exp1_cmc_res[2], [0.5, 0.5], marker='o', markersize=10, color='red', label='CMC')
ax.plot(exp1_is_res[2], [1.5, 1.5], marker='o', markersize=10, color='green', label='IS')

# Highlight intervals
ax.fill_betweenx([0, 1], exp1_cmc_res[2][0], exp1_cmc_res[2][1], color='red', alpha=0.3)
ax.fill_betweenx([1, 2], exp1_is_res[2][0], exp1_is_res[2][1], color='green', alpha=0.3)

# Set labels and title
ax.set_yticks([0.5, 1.5])
ax.set_yticklabels(['CMC', 'IS'])
ax.set_xlabel('Confidence Intervals')
ax.set_title('Confidence Intervals for CMC and IS')

# Add legend
ax.legend()

plt.show(block=False)




#-------------------------------------------------------------------------------------------------

#2 - Difference in deltas
exp2 = analysis.experiment2(data,Yi,R,seed,sample_size,deltaComparison)
exp2_mus = exp2[0]
exp2_cmc_avgs = exp2[1][0]
exp2_is_avgs = exp2[1][1]
exp2_cmc_vars = exp2[2][0]
exp2_is_vars = exp2[2][1]
exp2_cmc_ci = exp2[3][0]
exp2_is_ci = exp2[3][1]

exp2_cmc_ci_lower = []
exp2_cmc_ci_higher = []
exp2_is_ci_lower = []
exp2_is_ci_higher = []

for i in range(0,len(deltaComparison)):
    exp2_cmc_ci_lower.append(exp2_cmc_ci[i][0])
    exp2_cmc_ci_higher.append(exp2_cmc_ci[i][1])
    exp2_is_ci_lower.append(exp2_is_ci[i][0])
    exp2_is_ci_higher.append(exp2_is_ci[i][1])
    
# Plot 1: MuThota values
plt.figure(figsize=(8, 6))
plt.plot(deltaComparison, exp2_mus, label='MuThota', color='blue')
plt.xlabel('Delta')
plt.ylabel('MuThota')
plt.title('MuThota Values')
plt.show(block=False)

# Plot 2: Expected Value Variation on Delta
plt.figure(figsize=(8, 6))
plt.plot(deltaComparison, exp2_cmc_avgs, label='CMC', color='red')
plt.plot(deltaComparison, exp2_is_avgs, label='IS', color='green')
plt.xlabel('Delta')
plt.ylabel('Expected Value')
plt.title('Expected Value Variation on Delta')
plt.legend()
plt.show(block=False)

# Plot 3: Variance of Each Method Depending on Delta
plt.figure(figsize=(8, 6))
plt.plot(deltaComparison, exp2_cmc_vars, label='CMC', color='red')
plt.plot(deltaComparison, exp2_is_vars, label='IS', color='green')
plt.xlabel('Delta')
plt.ylabel('Variance')
plt.title('Variance of Each Method Depending on Delta')
plt.legend()
plt.show(block=False)

# Plot 4: Confidence Intervals
plt.figure(figsize=(8, 6))
plt.plot(deltaComparison, exp2_cmc_ci_lower, label='CMC lower limit', color='red')
plt.plot(deltaComparison, exp2_cmc_ci_lower, label='CMC higher limit', color='red')
plt.plot(deltaComparison, exp2_is_ci_lower, label='IS lower limit', color='green')
plt.plot(deltaComparison, exp2_is_ci_higher, label='IS higher limit', color='green')
plt.fill_between(deltaComparison, exp2_cmc_ci_lower, exp2_cmc_ci_higher, color='red', alpha=0.3, label='CMC CI')
plt.fill_between(deltaComparison, exp2_is_ci_lower, exp2_is_ci_higher, color='green', alpha=0.3, label='IS CI')
plt.xlabel('Delta')
plt.ylabel('Confidence Intervals')
plt.title('Confidence Intervals')
plt.legend()
plt.show(block=True)




#--------------------------------------------------------------------------------------------------

# 3 - New batch vs reused batch
exp3 = analysis.experiment3(data,Yi,R,seed,sample_size,delta)
#Obtain data from newBatch - performed in experiment 1
exp3_newBatch_results = exp1_results
#Obtain results from experiment 3
exp_3_subs_mu,exp3_subs_results = exp3

print(exp3_newBatch_results)
print(exp3_subs_results)

# Create a table for the mu values
mu_table_data = [["", "New Batch", "Reused Batch 1"],
                 ["Mu Value", exp3_newBatch_mu, exp_3_subs_mu]]

# Create figure and axis
fig, axs = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [1, 2], 'hspace': 0.4})

# Plot 1: Table with Mu Values
axs[0].axis('off')  # Turn off axis for the table
table = axs[0].table(cellText=mu_table_data, loc='center', cellLoc='center', colWidths=[0.2, 0.2, 0.2])

# Adjust table layout
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

# Plot 2: Confidence Intervals for Experiments 1 and 3
axs[1].plot(exp3_newBatch_results[1][2], [0.5, 0.5], marker='o', markersize=10, color='blue', label='Experiment 3 (New Batch)')
axs[1].fill_betweenx([0, 1], exp3_newBatch_results[1][2][0], exp3_newBatch_results[1][2][1], color='blue', alpha=0.3)
axs[1].plot(exp3_subs_results[1][2], [1.5, 1.5], marker='o', markersize=10, color='orange', label='Experiment 3 (Reused Batch 1)')
axs[1].fill_betweenx([1, 2], exp3_subs_results[1][2][0], exp3_subs_results[1][2][1], color='orange', alpha=0.3)
axs[1].set_xlabel('Confidence Intervals')
axs[1].set_ylabel('Experiment 1 and 3')
axs[1].set_title('Confidence Intervals for Experiments 1 and 3')
axs[1].legend()
plt.show(block=False)


#----------------------------------------------------------------------------------------------------

#4 - Compare under standard conditions with different sample size
exp4 = analysis.experiment4(data,Yi,RComparison,seed,sample_size,delta)

exp4_cmc_res = []
exp4_is_res = []
#there will be 1 result for each R value
for res in exp4:
    #Separate results for CMC and IS
    a,b = res
    exp4_cmc_res.append(a)
    exp4_is_res.append(b)

exp4_cmc_avgs = []
exp4_cmc_vars = []
exp4_cmc_ci_lower = []
exp4_cmc_ci_higher = []
exp4_is_avgs = []
exp4_is_vars = []
exp4_is_ci_lower = []
exp4_is_ci_higher = []
    
for i in range(0,len(RComparison)):
    exp4_cmc_avgs.append(exp4_cmc_res[i][0])
    exp4_cmc_vars.append(exp4_cmc_res[i][1])
    exp4_cmc_ci_lower.append(exp4_cmc_res[i][2][0])
    exp4_cmc_ci_higher.append(exp4_cmc_res[i][2][1])
    exp4_is_avgs.append(exp4_is_res[i][0])
    exp4_is_vars.append(exp4_is_res[i][1])
    exp4_is_ci_lower.append(exp4_is_res[i][2][0])
    exp4_is_ci_higher.append(exp4_is_res[i][2][1])
    
# Plot 2: Expected Value Variation on R
plt.figure(figsize=(8, 6))
plt.plot(RComparison, exp4_cmc_avgs, label='CMC', color='red')
plt.plot(RComparison, exp4_is_avgs, label='IS', color='green')
plt.xlabel('R')
plt.ylabel('Expected Value')
plt.title('Expected Value Variation on R')
plt.legend()
plt.show(block=False)

# Plot 3: Variance of Each Method Depending on Delta
plt.figure(figsize=(8, 6))
plt.plot(RComparison, exp4_cmc_vars, label='CMC', color='red')
plt.plot(RComparison, exp4_is_vars, label='IS', color='green')
plt.xlabel('R')
plt.ylabel('Variance')
plt.title('Variance of Each Method Depending on R')
plt.legend()
plt.show(block=False)

# Plot 4: Confidence Intervals
plt.figure(figsize=(8, 6))
plt.plot(RComparison, exp4_cmc_ci_lower, label='CMC lower limit', color='red')
plt.plot(RComparison, exp4_cmc_ci_lower, label='CMC higher limit', color='red')
plt.plot(RComparison, exp4_is_ci_lower, label='IS lower limit', color='green')
plt.plot(RComparison, exp4_is_ci_higher, label='IS higher limit', color='green')
plt.fill_between(RComparison, exp4_cmc_ci_lower, exp4_cmc_ci_higher, color='red', alpha=0.3, label='CMC CI')
plt.fill_between(RComparison, exp4_is_ci_lower, exp4_is_ci_higher, color='green', alpha=0.3, label='IS CI')
plt.xlabel('R')
plt.ylabel('Confidence Intervals')
plt.title('Confidence Intervals')
plt.legend()
plt.show(block=True)
