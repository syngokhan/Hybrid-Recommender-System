#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x : "%.3f" %x)
pd.set_option("display.width", 200)


# In[3]:


rating_path = "/Users/gokhanersoz/Desktop/VBO_Dataset/movie_lens_dataset/rating.csv"
movie_path =  "/Users/gokhanersoz/Desktop/VBO_Dataset/movie_lens_dataset/movie.csv"


# In[5]:


rating_ = pd.read_csv(rating_path)
movie_ = pd.read_csv(movie_path)


# In[7]:


rating = rating_.copy()
movie = movie_.copy()


# In[8]:


movie.head()


# In[9]:


rating.head()


# In[10]:


df = movie.merge(right=rating, how = "left", on = "movieId")
df.head()


# In[11]:


######################################
# Creating User Movie Df
######################################


# In[12]:


# Our goal is to create the user movie matrix, let's do the operations here...


# In[13]:


# Let's look at the total number of comments. A user may have commented on more than one movie.

print("Comments : {}".format(df.shape[0]))


# In[14]:


# Let's look at the number of unique movies...

print("Unique Movies Values : {}".format(df.title.nunique()))


# In[15]:


# How many comments were made on which movie...

# By looking at the comments made for the movies from here, 
# we can remove the less watched or the less commented ones...

title_comments = df["title"].value_counts()
title_comments = pd.DataFrame(title_comments)
title_comments


# In[22]:


# Let's separate rare movies or less watched movies

rare_movies = title_comments[title_comments["title"] <= 1000]
rare_movies


# In[23]:


# Now we need to filter this on the main dataframe

# We saw that the number of movies and the number of comments decreased...

rare_movies = rare_movies.index
commen_movies = df[~df["title"].isin(rare_movies)]
commen_movies.head()


# In[31]:


print(f"Comments : {commen_movies.shape[0]}")
      
print(f"Unique Movies Values : {commen_movies.title.nunique()}")


# In[ ]:


# Let's create user_movie_df...
# Here we will take the values title, rating, userid.

# commen_movies.pivot_table(values = "rating" , columns = ["title"], index = ["userId"])
# pd.pivot_table(values = "rating" , columns = "title" , index = "userId", data = commen_movies)


# In[37]:


commen_movies.head()


# In[43]:


print("UserID Unique : {}".format(commen_movies.userId.nunique()))
print("Movies Unique : {}".format(commen_movies.title.nunique()))


# In[40]:


user_movie_df = commen_movies.pivot_table(values = "rating",
                                          columns="title",
                                          index = "userId")

print("User Movie Shape : {}".format(user_movie_df.shape))


# In[44]:


user_movie_df.head()


# In[45]:


######################################
# Making Item-Based Movie Suggestions
######################################


# In[55]:


# Let's shoot a random movie
movie_name = pd.Series(user_movie_df.columns).sample(1 , random_state = 30).values[0]
movie_name


# In[57]:


# Get the complete movie from user_movie_df
# Let's see who gave what rating, we will establish a corr relationship with user_movie_df...

select_movie_df = user_movie_df[movie_name]
select_movie_df[select_movie_df > 0]


# In[59]:


# Let's evaluate all movies on user_movie_df

"""
Compute pairwise correlation.

Pairwise correlation is computed between rows or columns of
DataFrame with rows or columns of Series or DataFrame. DataFrames
are first aligned along both axes before computing the
correlations.

"""

user_movie_df.corrwith(select_movie_df).sort_values(ascending = False).head()


# In[64]:


example = user_movie_df.corrwith(select_movie_df).sort_values(ascending = False).head()
example = pd.DataFrame(example).index.to_list()

print(f"Recommended Movie For '{example[0]}'",end = "\n\n")
print("".center(50,"*"),end = "\n\n")
for i in example[1:]:
    print(i)


# In[65]:


movie_name = "Matrix, The (1999)"
movie_name = user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(5)


# In[77]:


######################################
# Functionalization of Operations
######################################


# In[81]:


def create_user_movie_df():
    import pandas as pd
    rating_path = "/Users/gokhanersoz/Desktop/VBO_Dataset/movie_lens_dataset/rating.csv"
    movie_path =  "/Users/gokhanersoz/Desktop/VBO_Dataset/movie_lens_dataset/movie.csv"
    
    rating = pd.read_csv(rating_path)
    movie = pd.read_csv(movie_path)
    
    df = movie.merge(right = rating , how = "left", on = "movieId")
    comment_count = pd.DataFrame(df.title.value_counts())
    rare_movies = comment_count[comment_count["title"] <= 1000].index
    commen_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = commen_movies.pivot_table(values = "rating", index = "userId", columns = "title")
    
    return user_movie_df


# In[82]:


user_movie_df = create_user_movie_df()


# In[83]:


user_movie_df.head()


# In[89]:


def item_based_recommender(user_movie_df ,movie_name ,count = 5):
    
    movie_name = [word for word in user_movie_df.columns if movie_name in word]
    movie_name = pd.Series(movie_name).sample(1).values[0]
    
    selected_movie_df = user_movie_df[movie_name]
    calculation = user_movie_df.corrwith(selected_movie_df).sort_values(ascending = False).head(count+1)
    results =calculation.index 
    
    print(f"Selected Movie '{results[0]}'" , end = "\n\n")
    print("".center(50,"*"),end = "\n\n")
    
    print(f"Recommended Movies", end = "\n\n")
    for movie in results[1:]:
        print(movie)

    

item_based_recommender(user_movie_df , "Matrix", 2)


# In[87]:


######################################
# BONUS: Storing and Recalling USER-MOVIE DF
######################################


# In[86]:


"""
'r'       open for reading (default)
'w'       open for writing, truncating the file first
'x'       create a new file and open it for writing
'a'       open for writing, appending to the end of the file if it exists
'b'       binary mode
't'       text mode (default)
'+'       open a disk file for updating (reading and writing)
'U'       universal newline mode (deprecated)
"""

# saving user_movie_df
import pickle

pickle.dump(user_movie_df, open("user_movie_df.pkl","wb"))

# install user_movie_df
user_movie_df = pickle.load(open("user_movie_df.pkl","rb"))
user_movie_df.head()


# ## Mission 2:
# 
# * Determine the movies watched by the user to be suggested.

# In[91]:


print("User Movie DataFrame Shape : {}".format(user_movie_df.shape))


# In[93]:


random_user = int(pd.Series(user_movie_df.index).sample(1 , random_state=45).values[0])
random_user


# In[110]:


###########################################
# Determining the Movies Watched by the User to Suggest
###########################################

# We used any() as we wanted (numbers , ).

random_user_df = user_movie_df[user_movie_df.index == random_user]
movies_wathced = random_user_df.columns[random_user_df.notna().any()].tolist()
movies_wathced


# In[112]:


#Don't forget True is turning False....
type(random_user_df.notna()),type(random_user_df.notnull().any())


# In[116]:


# Movies watched by the user 
# Number of movies watched by the user

test = user_movie_df.loc[user_movie_df.index == random_user , movies_wathced]
test


# In[118]:


movies = test.columns.tolist()
movies


# In[119]:


print("Number of Movies : {}".format(len(movies)))


# ## Mission 3:
# 
# * Access data and Ids of other users watching the same movies.

# In[120]:


###########################################
# Accessing Data and Ids of Other Users Watching the Same Movies
###########################################


# In[121]:


# Let's just shoot these movies from user_movie_df
movie_watched_df = user_movie_df[movies_wathced]
print("Movie Watched DataFrame Shape : {}".format(movie_watched_df.shape))


# In[123]:


# Now let's see how many people have watched according to these movies !!!!!
# notnull() --- > Detect existing (non-missing) values.
# notna() ---- > Detect existing (non-missing) values.

movie_watched_df.T.iloc[:5,:5]


# In[124]:


movie_watched_df.T.iloc[:5,:5].notnull()


# In[125]:


movie_watched_df.T.iloc[:5,:5].notnull().sum()


# In[134]:


# When notnull() is called, filled ones are returned as False and those that are not True are added.

user_movie_count = movie_watched_df.T.notnull().sum()
user_movie_count.head()


# In[135]:


user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId","movie_counts"]
user_movie_count.sort_values(by = "movie_counts" , ascending = False).head()


# In[137]:


# Do not forget to include itself in it !!!!!

[col for col in user_movie_count.index.tolist() if col == random_user]


# In[144]:


# perc = len(movies_watched) * 60 / 100
# len(movies_watched) = 33 , perc = 19.8

users_same_movies = user_movie_count[user_movie_count["movie_counts"] > 20]["userId"]
users_same_movies.head()


# In[146]:


len(users_same_movies),users_same_movies.nunique()


# ## Mission 4:
# 
# * Identify the users who are most similar to the user to be suggested.

# In[147]:


###########################################
# Determination of Users with the Most Similar Behaviors to the User to be Suggested
###########################################


# In[153]:


print("Movie Watched DataFrame Shape : {}".format(movie_watched_df.shape))
print("Random User DataFrame Shape : {}".format(random_user_df.shape))


# In[154]:


# Attention here we have our user at the end, we will post this later !!!

final_df = pd.concat([movie_watched_df[movie_watched_df.index.isin(users_same_movies)],
                      random_user_df[movies_wathced]] )

print("Final DataFrame Shape : {}".format(final_df.shape))


# In[156]:


random_user


# In[157]:


final_df.tail()


# In[160]:


final_df.T.corr().iloc[:5,:5]


# In[163]:


final_df.T.corr().iloc[:5,:5].unstack()


# In[165]:


corr_df = final_df.T.corr().unstack().sort_values()
print("Before Duplicates : {}".format(corr_df.shape))


# In[166]:


corr_df = corr_df.drop_duplicates()
print("After Duplicates : {}".format(corr_df.shape))


# In[168]:


corr_df = pd.DataFrame(corr_df , columns = ["corr"])
corr_df.index.names = ["user_id_1","user_id_2"]
corr_df = corr_df.reset_index()
corr_df


# In[170]:


random_user


# In[171]:


corr_df.sort_values(by = "corr", ascending = False).head(5)


# In[179]:


# Users with a correlation of 45 percent or more with the user...
# corr value is the highest 58 for this user, so I entered the value 45...

top_users = corr_df[ (corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][["user_id_2","corr"]]

top_users = top_users.reset_index(drop=True)
top_users = top_users.sort_values("corr", ascending = False)
top_users.head()


# In[180]:


top_users.rename(columns = {"user_id_2" : "userId"}, inplace = True)


# In[181]:


top_users.shape


# In[182]:


# UserId with the most similar behavior with me

print("Number of UserId : {}".format(top_users.shape[0]))


# In[183]:


rating.head()


# In[185]:


movie.head()


# In[186]:


# Here I took a userId that acts in common with me according to the movies I have watched for random_user that I have chosen...
# But they may have watched different movies besides me....
# I think this is the breaking point

top_users_rating = top_users.merge(right = rating[["userId","movieId","rating"]], how = "inner")
print("Top Users Rating Shape : {}".format(top_users_rating.shape))


# In[189]:


top_users_rating.head()


# In[190]:


# We must remove the random user userIds we have chosen from the top users rating...

top_users_rating = top_users_rating[top_users_rating["userId"] != random_user]
print("Top Users Rating Shape : {}".format(top_users_rating.shape))


# ## Task 5:
# 
# * Calculate the Weighted Average Recommendation Score and keep the first 5 movies.

# In[191]:


top_users_rating["weighted_rating"] = top_users_rating["corr"] * top_users_rating["rating"]
top_users_rating.head()


# In[192]:


top_users_rating.groupby("movieId").agg({"weighted_rating" : "mean"}).head()


# In[193]:


recommendation_df = top_users_rating.groupby("movieId").agg({"weighted_rating" : "mean"})
recommendation_df = recommendation_df.reset_index()
recommendation_df.head()


# In[196]:


# We looked at the max values... we need to examine how many intervals it should be...

recommendation_df = recommendation_df.sort_values(by = "weighted_rating" , ascending = False)
recommendation_df.head()


# In[205]:


movie_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].                                                            sort_values(by = "weighted_rating",
                                                                       ascending = False)

print("Movie To Be Recommend Shape : {}".format(movie_to_be_recommend.shape))
print("MovieID Unique : {}".format(movie_to_be_recommend.movieId.nunique()))


# In[206]:


movie_to_be_recommend.head()


# In[207]:


movie.head()


# In[208]:


movie_to_be_recommend = movie_to_be_recommend.merge(right=movie[["movieId","title"]])
print("Last Movie To Be Recommend Shape : {}".format(movie_to_be_recommend.shape))


# In[209]:


movie_to_be_recommend.head()


# In[211]:


# I looked at this here, he won't recommend me the movies I've watched, it's nice, we've already released it, that's a separate issue, but
# I wanted to check it again....

[i for i in movie_to_be_recommend.title if i in movies_wathced]


# ## Görev 6:
# 
# 
# * Kullanıcının izlediği filmlerden en son en yüksek puan verdiği filmin adına göre item-based öneri yapınız.
# 
#     ▪ 5 öneri user-based
# 
#     ▪ 5 öneri item-based
# 
# * olacak şekilde 10 öneri yapınız.

# In[212]:


random_user


# In[218]:


df[df["userId"] == random_user]["title"].head()


# In[220]:


# For User-Based, the following output should be obtained.

user_based = pd.DataFrame(movie_to_be_recommend.title).head(20)
user_based


# In[219]:


item_based_recommender(user_movie_df , "Sabrina" , 5 )


# In[ ]:




