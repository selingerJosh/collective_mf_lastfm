library(tidyverse)
library(Cairo)
library(GGally)
library(network)

df_artist <- read_delim("data/artists.dat", delim = '\t')
df_tag <- read_delim("data/tags.dat",  delim = '\t')
df_user_artist <- read_delim("data/user_artists.dat", delim = '\t')
df_user_taggedartist <- read_delim("data/user_taggedartists.dat",  delim = '\t')
df_user_friends <- read_delim("data/user_friends.dat",  delim = '\t')
df_user_taggedartist_timestamp <- read_delim("data/user_taggedartists-timestamps.dat",  delim = '\t')
#data summaries
glimpse(df_artist)
glimpse(df_tag)
glimpse(df_user_artist)
glimpse(df_user_taggedartist)

#check integrity 
df_artist %>%
  filter(is.na(name) == T)

#not all artists are named
nrow(df_artist)

df_user_artist %>%
  distinct(artistID) %>%
  nrow()

#join artist names to user artists
df_user_artist_names <- df_user_artist %>%
  left_join(df_artist, by = c("artistID" = "id"))

#for labelled artists there is a one to one relationship 
df_user_artist_names %>%
  distinct(artistID, name) %>%
  group_by(artistID) %>%
  count() %>%
  filter(n > 1)

#number of  users no name
df_user_artist_names %>%
  distinct(artistID, name) %>%
  filter(is.na(name) == T) %>%
  nrow()
#1209/17632 

#user and listen counts per artist

count_user_artist <- df_user_artist_names %>%
  group_by(name, artistID) %>%
  count() %>%
  arrange(desc(n))


count_play_artist <- df_user_artist_names %>%
  group_by(name, artistID) %>%
  summarise(totalCounts = sum(weight)) %>%
  arrange(desc(totalCounts))

# relationship between artist user and artist play
# filter top 50 played artists
count_rank_scatter <- count_user_artist %>% ungroup() %>% 
  mutate(userRank = rank(desc(n))) %>% 
  left_join(
    count_play_artist %>% ungroup() %>% mutate(playRank = rank(desc(totalCounts))),
  by = "artistID")

cor.test(x=count_rank_scatter$userRank, y=count_rank_scatter$playRank, method = 'spearman')


count_rank_scatter_plot <- count_rank_scatter %>%
  ggplot(aes(x = playRank, y = userRank )) +
  geom_point(colour= 'steelblue', alpha = 0.7) +
  geom_text(aes(label = ifelse(userRank > 2 * playRank | playRank > 2 * userRank,
                               as.character(name.x), '')), position = position_jitter(),  size = 2.5, hjust=0,vjust=0) +
  geom_abline(intercept = 0, slope = 1, linetype = "dotted", colour= 'steelblue') + 
  coord_cartesian(xlim = c(1:80), ylim = c(1:60)) +
  scale_x_continuous(breaks = c(1,20,40,60,80)) +
  scale_y_continuous(breaks = c(1,20,40,60)) +
  labs(x =  'Rank by count of Listens', 
       y = 'Rank by count of Users', 
       title = 'Relationship between User Count Rank and Listen Count Rank per Artist',
       subtitle = 'Best Rank is 1') 
  
ggsave("plays_users.png", count_rank_scatter_plot, device = "png", dpi = 300, width = 9)


cor.test(x=cars$speed, y=cars$dist, method = 'spearman')


# dataset is extremely skewed but evidence of approximately being log normal 
#count_play_artist %>% summary

count_play_artist_hist <- count_play_artist %>% 
  mutate(transform = 'Level') %>%
  bind_rows(count_play_artist %>% mutate(totalCounts = log(totalCounts), transform = 'Log'))

hist_play_image <- count_play_artist_hist %>%
  ggplot(aes(x = totalCounts)) +
  geom_histogram(fill= 'steelblue') +
  facet_wrap(~transform, scale = 'free') +
  labs(x = 'User Artist Plays', 
       y = 'Counts', 
       title = 'Distribution of User Listen Counts per Artist', 
       subtitle = 'Level Listen Count vs Log Listen Count' ) 

ggsave("hist_plays.png", hist_play_image, device = "png", dpi = 300, width = 12)


#variety and volume of artists per listener 
count_artist_user <- df_user_artist %>%
  distinct(userID, artistID) %>%
  count(userID)

count_artist_user %>% summary

count_artist_user %>%
  group_by(n) %>%
  summarise(artist_count = n()) 
  mutate(artist_perc = artist_count/sum(artist_count)) %>%
  arrange(desc(artist_perc))

#96% 50 artists, mean 41%

count_play_user <- df_user_artist %>% 
  group_by(userID) %>%
  summarise(totalCount = sum(weight))
  
count_play_user %>% summary()

#42964 - 9742

count_play_user %>%
  ggplot(aes(x = totalCount)) +
  geom_histogram() +
  labs(x = 'Total listens per user', y = 'Count', title = 'Histogram of Listens per user')

#artist tags
#join artist tags distinct count with listen count 
count_distinct_tag_artist <- df_user_taggedartist %>%
  left_join(df_tag, by = c("tagID")) %>%
  distinct(artistID, tagID, tagValue) %>%
  count(artistID)

count_distinct_tag_artist %>% summary()

count_distinct_tag_artist %>% 
  ggplot(aes(x = log(n))) +
  geom_histogram()

count_play_artist %>%
  left_join(count_distinct_tag_artist, by = 'artistID') %>%
  ggplot(aes(x = n, y = totalCounts)) +
  geom_jitter(alpha = 0.2) +
  scale_y_continuous(trans = 'log10')


# top tags
count_top_user_tags <- df_user_taggedartist %>%
  left_join(df_tag, by = "tagID") %>%
  group_by(tagID, tagValue) %>%
  count() %>%
  arrange(desc(n))

## some
df_user_taggedartist %>%
  count(year)



#friend ship stats
friend_dist <- df_user_friends %>%
  group_by(userID) %>%
  count() %>%
  ggplot(aes(n)) +
  geom_histogram(fill= 'steelblue') +
  labs(title = "Distribution of User Friend Connection Counts", x = "Number of Friends", y = "Proportions")

ggsave("friend_dist.png", friend_dist, device = "png", dpi = 300, width = 8)

df_user_friends %>%
  group_by(userID) %>%
  count() %>%
  summary()


#plot undirected graph of adjacency list
net<- network(df_user_friends_mainstream, directed = FALSE, matrix.type = 'edgelist')
net
#stratify by listeners of popular music 
social_network <- ggnet2(net, node.alpha = 0.2,  node.color = 'steelblue') + 
  labs(title = "Social Network of Last.FM User Friends") 

ggsave("social_network.png", social_network, device = "png", dpi = 300, width = 8)


