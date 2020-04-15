import pandas as pd
import matplotlib.pyplot as plt


df_user_artist = pd.read_csv("data/user_artists.dat", sep = '\\t', engine='python')

df_artist = pd.read_csv("data/artists.dat", sep = '\\t', engine='python')

df_user_taggedartist = pd.read_csv("data/user_taggedartists.dat", sep = '\\t', engine='python')

df_tag = pd.read_csv("data/tags.dat", sep = '\\t', engine='python')

# simple data profiling
# 1 to 1 relationship of artistID to name
df_artist.groupby('id').count().sort_values('name', ascending=False)

# Report counts of artist on user and play level
df_user_artist_full = pd.merge(df_user_artist,
                               df_artist,
                               how='left',
                               left_on= 'artistID', right_on='id' )

user_counts = (df_user_artist.
        groupby('artistID', as_index=False).
        count().
        sort_values('userID', ascending=False))

user_counts.describe()

user_counts.plot(kind = 'bar',x = 'artistID', y = 'userID',
          logy = True, xticks = [],
          title = 'User Artist Interaction Counts')
plt.show()

play_counts = (df_user_artist.
        groupby('artistID', as_index=False).
        sum().
        sort_values('weight', ascending=False))

df_user_artist.describe()

play_counts.plot(kind = 'bar',x = 'artistID', y = 'weight',
          logy = True, xticks = [],
          title = 'User Artist Interaction Counts')
plt.show()



df_user_artist_full.groupby('name', as_index=False).count().sort_values('userID', ascending=False, )['userID']