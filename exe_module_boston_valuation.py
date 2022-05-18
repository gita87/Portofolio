from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

# Getting Data

boston_dataset = load_boston()
# load_boston() is a dataset from >>> from sklearn.datasets import load_boston

data = pd.DataFrame(data = boston_dataset.data,
                   columns = boston_dataset.feature_names)


featurex = data.drop(['INDUS', 'AGE','B','ZN'], axis=1)
featurex.head(2)

log_prices = np.log(boston_dataset.target)
target = pd.DataFrame(log_prices, columns=['PRICES'])

# Making some indexes to 'property_stats' as dummy data

CRIME_IDX = 0
CHAS_IDX = 1
NOX_IDX = 2
RM_IDX = 3
DIS_IDX = 4
RAD_IDX = 5
TAX_IDX = 6
PTRATIO_IDX = 7
LSTAT_IDX = 8



property_statx = featurex.mean().values.reshape(1,9)


# Calculate the MSE and R-MSE
regrx = LinearRegression().fit(featurex, target)

fitted_valx = regrx.predict(featurex)
MSEx = mean_squared_error(target, fitted_valx)

RMSEx = MSEx**0.5

# Making a Function
def x_log_estimate(total_rooms,
                   polution_index,
                   students_per_classroom,
                   weighted_distance_to_workplace,
                   next_to_river=False,
                   high_confidence=True) :
    
    # Configure property
    property_statx[0][RM_IDX] = total_rooms
    property_statx[0][PTRATIO_IDX] = students_per_classroom
    property_statx[0][NOX_IDX] = polution_index
    property_statx[0][DIS_IDX] = weighted_distance_to_workplace
    property_statx[0][CHAS_IDX] = next_to_river
    
    if next_to_river:
        property_statx[0][CHAS_IDX] = 1
    else :
        property_statx[0][CHAS_IDX] = 0
    
    # make prediction
    log_estimatex = regrx.predict(property_statx)[0][0]
    
    
    # Calculation Range 
    if high_confidence :
        upper_bound = log_estimatex + 2*RMSEx
        lower_bound = log_estimatex - 2*RMSEx
        interval = 95
    else :
        upper_bound = log_estimatex + RMSEx
        lower_bound = log_estimatex - RMSEx
        interval = 68
        
    return log_estimatex, upper_bound, lower_bound, interval

# Adapting Valuation Tool to The Latest Situation
median_propertyx = np.median(boston_dataset.target)

# Step 1 : Getting Median from our dataset
featurex.mean()
median_propertyx = np.median(boston_dataset.target)

# Step 2 : Getting Median from Today's dataset
# taking data from zillow.com
zillow_median_price = 583

# Step 3 : Scalling Median Factor from Past vs Now 
scale_factor = zillow_median_price / median_propertyx
'''
# Step 4 : Converting Price into Today's Inflation
# convert to today's dollar
dollar_est = np.e**log_est * 1000 * scale_factor
print(round(dollar_est,-2))

upper_price = np.e**upper*1000*scale_factor
print(round(upper_price,-2))

lower_price = np.e**lower*1000*scale_factor
print(round(lower_price,-2))

'''
# Wrapping Them Up
def shock_maker_PRO_MAX_version(total_rooms,
                                polution_index,
                                students_per_classroom,
                                weighted_distance_to_workplace,
                                next_to_river=False,
                                high_confidence=True) :
    
    log_est, upper, lower, conf = x_log_estimate(total_rooms,
                                                 polution_index,
                                                 students_per_classroom,
                                                 weighted_distance_to_workplace,
                                                 next_to_river=False,
                                                 high_confidence=True) 
    
    dollar_est = np.e**log_est * 1000 * scale_factor
    upper_price = np.e**upper*1000*scale_factor
    lower_price = np.e**lower*1000*scale_factor
    
    print('The estimated property value is $', round(dollar_est,-2))
    print(f'At {conf}% confidence the valuation range is')
    print(f'USD {round(lower_price,-2)} at the lower end to USD {round(upper_price,-2)} at the high end.')
























