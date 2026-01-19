import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps


plt.close('all')

#read the data and create second column for sorted data
data = pd.read_csv('iat_data.txt', names=['data'])
data['sorted'] = data['data'].sort_values().reset_index(drop=True)


#plot histogram
plt.figure('Histogram')
plt.hist(data['data'], bins=50)
plt.xlim([0,100])
plt.xticks(range(0,100,5))
plt.xlabel('Intervall')
plt.ylabel('Number of Datapoints in Intervall')
plt.show()


#get data summary
mean = round(data['data'].mean(),4)
print(f'the mean of the data is {mean}')
var = round(data['data'].var(),4)
print(f'the variance of the data is {var}')
skw = round(sps.skew(data['data']),4)
print(f'the skewness of the data is {skw}')
kur = round(sps.kurtosis(data['data']),4)
print(f'the kurtosis of the data is {kur}')
co_var = round(sps.variation(data['data']),4)
print(f'the coefficient of variation of the data is {co_var}')



#create dataframe with lag for scatter and correlation plot
data_lag = pd.read_csv('iat_data.txt', names=['data_lag0'])
for i in range(20):
    data_lag.loc[max(data_lag.index)+1, :] = 0
    data_lag[f'data_lag{i+1}'] = data_lag[f'data_lag{i}']

    for j in range(i+1):
        data_lag[f'data_lag{j}'] = data_lag[f'data_lag{j}'].shift(periods=1, fill_value=0)


#scatter plot with x from data at i and y as value at i+1
plt.figure('Scatterplot unsorted')
plt.scatter(data_lag['data_lag0'], data_lag['data_lag1'], s=3)
plt.xlim([0,100])
plt.ylim([0,100])
plt.xlabel('Observation k')
plt.ylabel('Observation k+1')
plt.show()

#lag k
corr = np.array(data_lag.corr())[0][1:]  #get only first row with [0] and then skip first entry with [1:]
plt.figure('Lag-Correlation Plot unsorted')
plt.plot(range(1, 21), corr)
plt.ylim(0,1)
plt.xlim(0.5,20.5)
plt.xticks(range(1,21))
plt.xlabel('Lag')
plt.ylabel('Correlation')
plt.show()




#create dataframe with sorted data and lag for sorted scatter and correlation plot
data_lag_sorted = pd.read_csv('iat_data.txt', names=['data_lag0']).sort_values(by=['data_lag0']).reset_index(drop=True)

for i in range(20):
    data_lag_sorted.loc[max(data_lag_sorted.index)+1, :] = 0
    data_lag_sorted[f'data_lag{i+1}'] = data_lag_sorted[f'data_lag{i}']

    for j in range(i+1):
        data_lag_sorted[f'data_lag{j}'] = data_lag_sorted[f'data_lag{j}'].shift(periods=1, fill_value=0)



#scatter plot of sorted data
plt.figure('Scatterplot sorted')
plt.scatter(data_lag_sorted['data_lag0'], data_lag_sorted['data_lag1'], s=3)
plt.xlim([0,100])
plt.ylim([0,100])
plt.xlabel('Observation k')
plt.ylabel('Observation k+1')
plt.show()

#lag k sorted
corr_sorted = np.array(data_lag_sorted.corr())[0][1:]  #get only first row with [0] and then skip first entry with [1:]
plt.figure('Lag-Correlation Plot sorted')
plt.plot(range(1, 21), corr_sorted)
plt.ylim(0,1)
plt.xlim(0.5,20.5)
plt.xticks(range(1,21))
plt.xlabel('Lag')
plt.ylabel('Correlation')
plt.show()


#kolmogorov-smirnov goodness of fit test for a list of distributions
print('\n')
list_of_dists = ['bradford','cosine','crystalball','expon','gamma','johnsonsb','laplace','logistic','lognorm','norm','rayleigh','rice','uniform','wald','weibull_min']
results = []
for i in list_of_dists:
    param = getattr(sps, i).fit(data['data'])
    ks_result = sps.kstest(data['data'], i, args=param)
    results.append((i,ks_result[0],ks_result[1]))

for j in results:
    print(f"test-statistic for {j[0]}: {j[1].round(4)}")




# Analyze the three best fitting distributions for their mean, variance, skewness and kurtosis.

#get parameters for fitted distribution: sps.SCIPY_DISTRB_NAME.fit(data['data'])
#use parameters to create distribution
#analyze distribution with .stats(‘mvsk’)

best_dists = ['johnsonsb', 'lognorm', 'weibull_min']

dist_objects = {}

print('\nMoment comparison (mean, variance, skewness, kurtosis):\n')

for dist_name in best_dists:
    dist = getattr(sps, dist_name)
    params = dist.fit(data['data'])
    dist_objects[dist_name] = params
    
    mean_d, var_d, skew_d, kurt_d = dist.stats(*params, moments='mvsk')
    
    print(f"{dist_name}:")
    print(f"  mean     = {mean_d:.4f}")
    print(f"  variance = {var_d:.4f}")
    print(f"  skewness = {skew_d:.4f}")
    print(f"  kurtosis = {kurt_d:.4f}\n")




#PP-plot
#create sample cdf

n = len(data)
sample_cdf = np.arange(1, n + 1) / (n + 1)

plt.figure('PP-Plot')

for dist_name in best_dists:
    dist = getattr(sps, dist_name)
    params = dist_objects[dist_name]
    model_cdf = dist.cdf(data['sorted'], *params)
    plt.plot(sample_cdf, model_cdf, label=dist_name)

# reference line
plt.plot([0, 1], [0, 1], 'r--', label='ideal fit')


plt.figure('PP-Plot')
#plot distr 1
#plot distr 2
#plot distr 3
#plot sample cdf as reference
plt.legend()
plt.xlabel('Sample Value')
plt.ylabel('Model Value')
plt.show()




#QQ-plot

#use .ppf() to create inverse of your distribution

plt.figure('QQ-Plot')
#plot distr 1
#plot distr 2
#plot distr 3
plt.plot(data['sorted'], data['sorted'], color='r', label='sample')
plt.legend()
plt.xlabel('Sample Value')
plt.ylabel('Model Value')
plt.show()





#distribution-function-difference plot

#calculate difference between the cdf of each distribution (.cdf(data['sorted'])) to the sample cdf

plt.figure('Distribution-Function-Difference-Plot')
#plot distr 1
#plot distr 2
#plot distr 3
plt.plot([0, max(data['sorted'])], [0,0], color='r', label='sample', linewidth=1.5)
plt.legend()
plt.xlim(0, max(data['sorted']))
plt.ylim(-0.2, 0.2)
plt.xlabel('x')
plt.ylabel('Difference')
plt.show()











