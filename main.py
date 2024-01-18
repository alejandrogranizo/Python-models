# main.py - main executions for the experimental part
# @author - Alejandro Granizo Castro (alejandro.granizo.castro@alumnos.upm.es)
# Thesis - Enhancing accuracy of estimators for financial options using importance sampling procedures: 
#           A practical approach for European call options on Euro Stoxx 50 Index

#Imports needed
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

#1 - comparison under standard circumstances - refer to paper section experimental configuration
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

plt.show(block=True)

exit()


#2 - Difference in deltas
exp2 = analysis.experiment2(data,Yi,R,seed,sample_size,deltaComparison)

#there will be 1 result for each delta
for res in exp2:
    #Separate results related to muThota value and accuracy/variance
    exp2_muThota,exp2_results = exp2
    #Separate results for CMC and IS
    exp2_cmc_res, exp2_is_res = exp2_results




#3 - New batch vs reused batch
exp3 = analysis.experiment3(data,Yi,R,seed,sample_size,delta)
#Obtain data from newBatch - performed in experiment 1
exp3_newBatch_results = exp1_results
#Obtain results from experiment 3
exp_3_subs_mu,exp3_subs_results = exp3



#4 - Compare under standard conditions with different sample size
exp4 = analysis.experiment4(data,Yi,RComparison,seed,sample_size,delta)

#there will be 1 result for each R value
for res in exp4:
    #Separate results for CMC and IS
    exp4_cmc_res, exp4_is_res = exp4
