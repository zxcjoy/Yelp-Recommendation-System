"""
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
>=0 and <1: 102019
>=1 and <2: 33041
>=2 and <3: 6172
>=3 and <4: 811
>=4: 1

RMSE:
0.979820

Execution Time:
223s
"""

from pyspark import SparkContext
import time
import sys
import os
import math
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import json
import numpy as np
import pandas as pd

# export PYSPARK_PYTHON=python3.9
# test case
# python competition.py './data' './data/yelp_val.csv' './competition_output.csv'
# /opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit --executor-memory 4G --driver-memory 4G competition.py ../resource/asnlib/publicdata ../resource/asnlib/publicdata/yelp_val.csv ./task2_3_result.txt

def dict_neat_print(d):
    for key, value in list(d.items())[:10]: 
        print('key: ', key, 'value: ', value)

def compute_similarity(bus1, bus2):
    users1 = bus_user_rate_dict.get(bus1, {})
    users2 = bus_user_rate_dict.get(bus2, {})
    max_rating = 5

    users_same = set(users1.keys()) & set(users2.keys())
    users_same_list = list(users_same)

    if len(users_same) == 2:
        user1 = users_same_list[0]
        user2 = users_same_list[1]
        diff_user1 = bus_user_rate_dict[bus1][user1] - bus_user_rate_dict[bus2][user1]
        diff_user2 = bus_user_rate_dict[bus1][user2] - bus_user_rate_dict[bus2][user2]
        sim = (max_rating * 2 - (abs(diff_user1) + abs(diff_user2))) / max_rating 

        return sim
    
    if len(users_same) == 1:
        user = users_same_list[0]
        diff = bus_user_rate_dict[bus1][user] - bus_user_rate_dict[bus2][user]
        sim = (max_rating - abs(diff)) / max_rating
        return sim
    
    if len(users_same) < 1:
        return 0.5

    r1 = [users1[user] for user in users_same]
    r2 = [users2[user] for user in users_same]

    avg1 = sum(r1) / len(r1)
    avg2 = sum(r2) / len(r2)

    # correlation coefficient
    # similarity =  x / y
    x = sum((a - avg1) * (b - avg2) for a, b in zip(r1, r2))
    y = math.sqrt(sum((a - avg1) ** 2 for a in r1) * sum((b - avg2) ** 2 for b in r2))
    
    return x / y if y != 0 else 0
  
def item_based_re(bus, user):
    rate_avg = 3.7511703308515445

    if user not in user_bus_dict.keys():
        # Default rating if new user
        return bus_avg_dict.get(bus, rate_avg)
    
    if bus not in bus_user_dict.keys():
         # User's average rating  if new bus
        return user_avg_dict.get(user, rate_avg) 
    
    # temp sim and rate list, (sim, rating)
    temp = []
    user_buses = user_bus_dict[user]

    for each_bus in user_buses:

        if each_bus == bus:
            continue  
        
        pair = tuple(sorted((each_bus, bus)))
        if pair in sim_dict:
            similarity = sim_dict[pair]
        else:
            similarity = compute_similarity(each_bus, bus)
            sim_dict[pair] = similarity  

        if similarity > 0:
            rating = bus_user_rate_dict[each_bus][user]
            temp.append((similarity, rating))

    # Compute the weighted average of top similarities
    top_num = min(100, len(temp))

    # sort by similarity
    temp = sorted(temp, key=lambda x: -x[0])[:top_num]  
    weighted_sum = sum(w * r for w, r in temp)
    sum_of_weights = sum(abs(w) for w, _ in temp)
    return weighted_sum / sum_of_weights if sum_of_weights != 0 else rate_avg

def construct_df(file_path, columns, filter_key = None, filter_set = None):
    data = []
    with open(file_path) as file:
        for line in file:
            record = json.loads(line)
            if not filter_key or record[filter_key] in filter_set:
                data.append([record[x] for x in columns])
    return pd.DataFrame(data, columns=columns).set_index(columns[0])

def construct_feature_matrix(rdd, u_df, b_df, add_df_1, add_df_2):

    def get_features(record):
        uid, bid = record[0], record[1]
        user_features = u_df.loc[uid].tolist() if uid in u_df.index else [np.nan] * len(u_df.columns)
        business_features = b_df.loc[bid].tolist() if bid in b_df.index else [np.nan] * len(b_df.columns)
        add_df_1_features = add_df_1.loc[bid].tolist() if bid in add_df_1.index else [np.nan] * len(add_df_1.columns)
        add_df_2_features = add_df_2.loc[bid].tolist() if bid in add_df_2.index else [np.nan] * len(add_df_2.columns)
        return user_features + business_features + add_df_1_features + add_df_2_features

    return np.array(rdd.map(get_features).collect())

def count_true_attributes(attributes):
    if not isinstance(attributes, dict):
        return 0  # if the value is missing 
    
    # count the number of True attributes 
    count_true = sum(1 for value in attributes.values() if isinstance(value, str) and value.lower() == 'true')
    
    # nested dict
    for key, value in attributes.items():
        if isinstance(value, str) and value.startswith('{'):
            try:
                nested_attrs = json.loads(value.replace("'", "\""))  
                count_true += sum(1 for v in nested_attrs.values() if isinstance(v, bool) and v)
            except json.JSONDecodeError:
                continue  

    return count_true

def find_rmse(output_file):

    csv_file_path1 = './data/yelp_val.csv'
    csv_file_path2 = './' + output_file

    column_index = 2
    data1 = pd.read_csv(csv_file_path1, usecols=[column_index])
    data2 = pd.read_csv(csv_file_path2, usecols=[column_index])

    squared_differences = (data1.iloc[:, 0] - data2.iloc[:, 0]) ** 2
    mean_squared_error = squared_differences.mean()
    rmse = np.sqrt(mean_squared_error)
    return rmse

def find_error_distribution(output_file):

    csv_file_path1 = './data/yelp_val.csv'
    csv_file_path2 = './' + output_file

    column_index = 2
    data1 = pd.read_csv(csv_file_path1, usecols=[column_index])
    data2 = pd.read_csv(csv_file_path2, usecols=[column_index])

    # Compute the absolute differences
    absolute_differences = np.abs(data1.iloc[:, 0] - data2.iloc[:, 0])

    # Categorize differences
    error_levels = {
        '>=0 and <1': np.sum((absolute_differences >= 0) & (absolute_differences < 1)),
        '>=1 and <2': np.sum((absolute_differences >= 1) & (absolute_differences < 2)),
        '>=2 and <3': np.sum((absolute_differences >= 2) & (absolute_differences < 3)),
        '>=3 and <4': np.sum((absolute_differences >= 3) & (absolute_differences < 4)),
        '>=4': np.sum(absolute_differences >= 4)
    }

    # Print the error distribution
    for level, count in error_levels.items():
        print(f'{level}: {count}')

    return None

def dynamic_alpha(user_id):
    # return 0
    num_reviews = user_activity_dict.get(user_id, 0)
    if num_reviews > 200:
        # print('one more!')
        return 0.07
    else:
        return 0

folder_path = sys.argv[1]
val_file_path = sys.argv[2]
output_file_path = sys.argv[3]

sc = SparkContext('local[*]', 'competition')
sc.setLogLevel('WARN')

start_time = time.time()

# Train data
# user_id, bus_id, rate
train_file_path = folder_path + '/yelp_train.csv'
file_train = sc.textFile(train_file_path)
header_train = file_train.first()
data_train = file_train.filter(lambda x: x != header_train).map(lambda x: x.split(','))
train = data_train.map(lambda row: (row[0], row[1], float(row[2])))

# Validation data
file_val = sc.textFile(val_file_path)
header_val = file_val.first()
data_val = file_val.filter(lambda x: x != header_val).map(lambda x: x.split(','))
val = data_val.map(lambda row: (row[0], row[1]))

# Case 1, item based
# Preprocess data
bus_user_dict = train.map(lambda x: (x[1], x[0])).groupByKey().mapValues(set).collectAsMap()
user_bus_dict = train.map(lambda x: (x[0], x[1])).groupByKey().mapValues(set).collectAsMap()

# Additional variable to record the frequency of how often a user rates
user_activity_dict = {user: len(businesses) for user, businesses in user_bus_dict.items()}

# user_activity_values = user_activity_dict.values()
# average_activity = sum(user_activity_values) / len(user_activity_values) if user_activity_values else 0
# print(f'Average number of businesses interacted with by users: {average_activity}')
# # Average number of businesses interacted with by users: 40.44844720496894

# Find sum of rates
# Find average for bus
user_sum_count = train.map(lambda row: (row[0], (float(row[2]), 1)))\
                     .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
user_avg = user_sum_count.map(lambda x: (x[0], x[1][0] / x[1][1]))
user_avg_dict = user_avg.collectAsMap()

bus_sum_count = train.map(lambda row: (row[1], (float(row[2]), 1)))\
                      .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
bus_avg = bus_sum_count.map(lambda x: (x[0], x[1][0] / x[1][1]))
bus_avg_dict = bus_avg.collectAsMap()

# prepare a dict with bus: {user: rate}    
bus_user_rate = train.map(lambda row: (row[1], (row[0], row[2])))
bus_user_rate_dict = bus_user_rate.groupByKey()\
    .mapValues(lambda pairs: {x[0]: x[1] for x in pairs})\
    .collectAsMap()

# dict_neat_print(bus_user_rate_dict)

sim_dict = {}
predictions_item_based = val.map(lambda x: item_based_re(x[1], x[0])).collect()

# item_based_file_path = './item_based_result.csv'
# with open(output_file_path, "w") as file:
#         file.write('user_id, business_id, prediction\n')
#         for (uid, bid), prediction in zip(val.collect(), predictions_item_based):
#             file.write(f"{uid},{bid},{prediction}\n")

# Case 2, model  based
user_all = train.map(lambda x: x[0]).\
    union(val.map(lambda x: x[0])).\
    distinct().collect()
business_all = train.map(lambda x: x[1]).\
    union(val.map(lambda x: x[1])).\
    distinct().collect()

user_df = construct_df(folder_path + '/user.json', 
                          ['user_id', 'average_stars', 'review_count', 'fans','useful', 'funny', 'cool',
                           'compliment_hot', 'compliment_more', 'compliment_profile', 'compliment_cute',
                           'compliment_list', 'compliment_note', 'compliment_plain', 'compliment_cool',
                           'compliment_funny', 'compliment_writer', 'compliment_photos'],
                            'user_id', set(user_all))

business_df = construct_df(folder_path + '/business.json', 
                              ['business_id', 'stars', 'review_count','is_open','attributes'],
                                'business_id', set(business_all))

# count the true attributes
business_df['true_attribute_count'] = business_df['attributes'].apply(count_true_attributes)
business_df['attributes'] = business_df['true_attribute_count']
business_df.drop('true_attribute_count', axis=1, inplace=True)

### checkin counts
checkin_df = pd.read_json(folder_path + '/checkin.json', lines=True)
checkin_counts = {}
for index, row in checkin_df.iterrows():
    total_checkins = sum(row['time'].values())
    checkin_counts[row['business_id']] = total_checkins


checkin_counts_df = pd.DataFrame(list(checkin_counts.items()), columns=['business_id', 'total_checkins'])
checkin_counts_df.set_index('business_id', inplace=True)

### photos
photos_df = pd.read_json(folder_path + '/photo.json', lines =True)
photo_counts = photos_df.groupby('business_id').size()
photo_counts_df = photo_counts.reset_index(name = 'total_photos')
photo_counts_df.set_index('business_id', inplace =True)

X_train = construct_feature_matrix(train, user_df, business_df, checkin_counts_df, photo_counts_df)
# print(X_train[:10])

y_train = np.array(train.map(lambda x: float(x[2])).collect())

model = xgb.XGBRegressor(verbosity=0, n_estimators=300, random_state=1, max_depth=5, learning_rate = 0.1)
model.fit(X_train, y_train)

X_test = construct_feature_matrix(val, user_df, business_df,checkin_counts_df, photo_counts_df)
predictions_model_based = model.predict(X_test)

'''
### Recursive Feature Elimination with cross-validation ###
# Result : Select 22 features out of 23
# Selected features: ['average_stars' 'review_count' 'fans' 'useful' 'funny' 'cool'
#  'compliment_hot' 'compliment_more' 'compliment_profile' 'compliment_cute'
#  'compliment_list' 'compliment_note' 'compliment_plain' 'compliment_cool'
#  'compliment_writer' 'compliment_photos' 'stars' 'review_count' 'is_open'
#  'attributes' 'checkin_counts' 'photo_counts']

from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt

rfecv = RFECV(estimator=xgb.XGBRegressor(), step=1, cv=5, scoring='neg_mean_squared_error')
rfecv.fit(X_train, y_train)

print("Optimal number of features : %d" % rfecv.n_features_)
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross-validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

print("Mask of selected features:", rfecv.support_)  
print("Ranking of features:", rfecv.ranking_)  

feature_names = np.array(['average_stars', 'review_count', 'fans','useful', 'funny', 'cool',
                          'compliment_hot', 'compliment_more', 'compliment_profile', 'compliment_cute',
                           'compliment_list', 'compliment_note', 'compliment_plain', 'compliment_cool',
                           'compliment_funny', 'compliment_writer', 'compliment_photos', 
                           'stars', 'review_count','is_open','attributes',
                           'checkin_counts', 'photo_counts'])  # Example feature names

selected_features = feature_names[rfecv.support_]
print("Selected features:", selected_features)
'''

'''
### XGBoost Parameter Tuning Using Grid Search ###

model = xgb.XGBRegressor()
param_grid = {
    'n_estimators': [50, 100, 150, 300, 500],
    'max_depth': [5, 10, 15, 20],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best RMSE:", np.sqrt(-grid_search.best_score_))
# ### Results:
# Best parameters: {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 150}    
# Best RMSE: 0.9876296131549399
'''

final_predictions = []
for (uid, bid), item_pred, model_pred in zip(val.collect(), predictions_item_based, predictions_model_based):
    alpha = dynamic_alpha(uid)
    final_pred = alpha * item_pred + (1 - alpha) * model_pred
    final_predictions.append(final_pred)

with open(output_file_path, "w") as file:
        file.write('user_id, business_id, prediction\n')
        for (uid, bid), prediction in zip(val.collect(), final_predictions):
            file.write(f"{uid},{bid},{prediction}\n")

rmse = find_rmse(output_file_path)

print('RMSE = ', rmse)

find_error_distribution(output_file_path)

end_time = time.time()
print('Time used: ', end_time - start_time)


# Current_Best_RMSE = 0.9798194417440538
# model_rmse = 0.9798438485579618
