Method Description:
- I used hybrid recommendation model combined from model based using XGBRegressor and item-based recommendation.
Model-based recommendation provides a better RMSE after I adding more features including:
1. checkin-in counts for business from checkin.json
2. photos counts for business from photos.json
3. count of True attributes of the attributes column from business.json
4. More columns from user.json and business.json
- I run a recursive feature elimination with cross-validation and found that 'compliment_funny' is not a useful feature.
- I use grid serach to do XGBoost parameter tuning to find the best parameters for the model.
- Another trick to slight decrease the RMSE (0.0003) is that, for a user with more than 200 reviews count,
we consider the item-based recommendation with a factor of 0.07, otherwise, with 0 (only use model-based result).

Error Distribution:

\>=0 and <1: 102019

\>=1 and <2: 33041

\>=2 and <3: 6172

\>=3 and <4: 811

\>=4: 1

RMSE:
0.979820

Execution Time:
223s
