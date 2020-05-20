import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import surprise as spr
from cmfrec import CMF_explicit
from scipy.sparse import csgraph
import networkx

# raw import
df_user_artist = pd.read_csv("data/user_artists.dat", sep = '\\t', engine='python')

df_artist = pd.read_csv("data/artists.dat", sep = '\\t', engine='python')

df_user_taggedartist = pd.read_csv("data/user_taggedartists.dat", sep = '\\t', engine='python')

df_tag = pd.read_csv("data/tags.dat", sep = '\\t', engine='python')

df_user_friends = pd.read_csv("data/user_friends.dat", sep = '\\t', engine='python')

# one hot encode df_tag
df_one_hot_tagged = (df_user_taggedartist.groupby(['artistID', 'tagID'], as_index=False)
     .count()
     .pivot(index='artistID', columns='tagID', values='userID')
     .fillna(0)
     .astype(bool).astype(int)
 )

df_one_hot_tagged['ItemId'] = df_one_hot_tagged.index

# transform social graph to laplace matrix
df_user_friends['value'] = 1

# friends_list = np.array(df_user_friends).tolist()
# friends_graph = networkx.Graph() # create empty directed graph

# for i in range(len(friends_list)):
#    friends_graph.add_edge(friends_list[i][0], friends_list[i][1]) # add edges from original adjacency list

# friends_adjacent = networkx.adjacency_matrix(friends_graph, nodelist= sorted(friends_graph.nodes())).A
# friends_laplace = pd.DataFrame(csgraph.laplacian(friends_adjacent))

friends_adjacent = (df_user_friends.groupby(['userID', 'friendID'], as_index=False)
 .count()
 .pivot(index='userID', columns='friendID', values='value')
 .fillna(0)
 )

friends_adjacent['UserId'] = friends_adjacent.index

# log transform user artist interactions
df_user_artist['weight_log'] = np.log(df_user_artist['weight'])

# give appropriate fields correct name for cmfrec
ratings = df_user_artist.drop('weight', axis = 1)
ratings.columns = ['UserId', 'ItemId', 'Rating' ]
# testing cmf
model_no_sideinfo = CMF_explicit(method="als", k=40, lambda_=1e+1)
model_no_sideinfo.fit(ratings)

# testing full cmf
model_with_sideinfo = CMF_explicit(method="als", k=40, lambda_=1e+1, w_main=0.5, w_user=0.25, w_item=0.25)
model_with_sideinfo.fit(X=ratings, U=friends_adjacent, I=df_one_hot_tagged)

model