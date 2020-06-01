import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import surprise as spr

# raw import
df_user_artist = pd.read_csv("data/user_artists.dat", sep = '\\t', engine='python')

df_artist = pd.read_csv("data/artists.dat", sep = '\\t', engine='python')

df_user_taggedartist = pd.read_csv("data/user_taggedartists.dat", sep = '\\t', engine='python')

df_tag = pd.read_csv("data/tags.dat", sep = '\\t', engine='python')

# log transform user artist interactions
df_user_artist['weight_log'] = np.log(df_user_artist['weight'])

# summary stats user artist interactions
df_user_artist.describe()

# create user artist interaction matrix NOT NECESSARY WITH SURPRISE
# df_user_artist.pivot(index = 'userID', columns= 'artistID', values = 'weight_log')

# SIMPLE MODELS only user-item matrix
# load user artist matrix into surprise

# define rating scale as min to max log user artist interaction count to nearest int
reader = spr.Reader(rating_scale=(0,13))

df_user_artist_spr = spr.Dataset.load_from_df(df_user_artist[['userID','artistID','weight_log']], reader)

# simple data split
train, test = spr.model_selection.train_test_split(df_user_artist_spr, test_size=.20)

# We'll use the famous SVD algorithm.
algo_svd = spr.prediction_algorithms.matrix_factorization.SVD(biased = True, verbose = True)
algo_pmf = spr.prediction_algorithms.matrix_factorization.SVD(biased = False, verbose = True)
# Train the algorithm on the trainset, and predict ratings for the testset
algo_svd.fit(train)
algo_pmf.fit(train)
predictions = algo_svd.test(test)
predictions_other = algo_pmf.test(test)

# Then compute RMSE
print('SVD: ',spr.accuracy.rmse(predictions))
print('PMF: ',spr.accuracy.rmse(predictions_other))


#loop over
for l in [0.2,0.02,0.002,0.0002]:
    algo_svd = spr.prediction_algorithms.matrix_factorization.SVD(biased=True,  reg_all = l)
    algo_svd.fit(train)
    predictions = algo_svd.test(test)
    print('SVD: ', spr.accuracy.rmse(predictions))




from funk_svd.dataset import fetch_ml_ratings
test = fetch_ml_ratings(variant='100k')