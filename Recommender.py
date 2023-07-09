# Movie Recommender System

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
credit = pd.read_csv('tmdb_5000_credits.csv')
movie =pd.read_csv('tmdb_5000_movies.csv')

# Merge two dataset
movie=movie.merge(credit,on='title')

# Pick those that column that help us to make a tag
movie=movie[['movie_id','title','overview','genres','keywords','cast','crew']]
#movie.info()

# Remove missing data and duplicate data in our dataset
movie.isnull().sum()
movie.dropna(inplace=True)

movie.duplicated().sum()

# Take a important item in a each block of column  

import ast  # use to convert string to integer

def convert(obj):

    l = []

    for i in ast.literal_eval(obj):

        l.append(i['name'])

    return l

movie['keywords']=movie['keywords'].apply(convert)
movie['keywords'].head()

movie['genres']=movie['genres'].apply(convert)
movie['genres'].head()


def convert2(obj):

    l = []
    counter = 0
    for i in ast.literal_eval(obj):
       if counter!=3:
        l.append(i['name'])
        counter+=1
    
    return l
        
movie['cast']=movie['cast'].apply(convert2)
      

def convert3(obj):

    l = []

    for i in ast.literal_eval(obj):
       if i['job'] == 'Director':
        l.append(i['name'])
        break
    return l

movie['crew']=movie['crew'].apply(convert3)
movie['crew'].head()

movie['overview']=movie['overview'].apply(lambda x:x.split())

# After making each column in a list we have to remove space between words in each column

movie['genres']=movie['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movie['keywords']=movie['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movie['crew']=movie['crew'].apply(lambda x:[i.replace(" ","") for i in x])
movie['cast']=movie['cast'].apply(lambda x:[i.replace(" ","") for i in x])
  
#After all Now we have to create a tags in movie dataset

movie['tags']=movie['overview']+movie['genres']+movie['keywords']+movie['cast']+movie['crew']  

movie.head()

new_df=movie[['movie_id','title','tags']]

new_df.head()

# Now we have to convert list into string of tags

new_df['tags']=new_df['tags'].apply(lambda x: " ".join(x))

new_df['tags'][0]

new_df['tags']=new_df['tags'].apply(lambda x: x.lower())
new_df['tags'][0]

# Now we have to use vecotrization

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')
vector=cv.fit_transform(new_df['tags']).toarray()

cv.get_feature_names_out() # this is help us to check there is any sentence that meaning is same (ex:Action ,Actions)

# For this we have to make a vector

from sklearn.metrics.pairwise import cosine_similarity
similarity=cosine_similarity(vector)


def recommend(movie):
    index = new_df[new_df['title'] == movie].index[0]
    distance=similarity[index]
    movie_list = sorted(list(enumerate(distance)),reverse=True,key = lambda x: x[1])[1:7]
    for i in movie_list:
        print(new_df.iloc[i[0]].title)
        
        
        recommend('Avatar')
    
import pickle
pickle.dump(new_df,open('movies.pkl','wb'))        


pickle.dump(new_df.to_dict(),open('movies_dict.pkl','wb'))

pickle.dump(similarity,open('similarity.pkl','wb'))


