# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 11:34:44 2021

@author: Chaitanya
"""

import pandas as pd
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS 

path = 'C:\\Users\\h183096\\Downloads\\ml-25m\\'

movies = pd.read_csv(path+'movies.csv')
movies.shape

tags = pd.read_csv(path+'tags.csv')
tags.shape

ratings = pd.read_csv(path+'ratings.csv')
ratings.shape

links = pd.read_csv(path+'links.csv')
links.shape

genres = ['Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']

genres_rating_list = []

for i in range(len(genres)):
    fil = genres[i]+'_filter'
    mov = genres[i]+'_movies'
    rat = genres[i]+'_ratings'
    rat_mean = rat+'_mean'
    fil = movies['genres'].str.contains(genres[i])
    mov = movies[fil]
    rat = mov.merge(ratings, on='movieId', how='inner')
    rat_mean = round(rat['rating'].mean(), 2)
    #print(genres[i], round(rat_mean,2))
    genres_rating_list.append(rat_mean)
    
df = {'Genre':genres, 'Genres Mean Rating':genres_rating_list}
genres_rating = pd.DataFrame(df)
genres_rating['Genres Standard Deviation'] = genres_rating['Genres Mean Rating'].std()
genres_rating['Mean'] = genres_rating['Genres Mean Rating'].mean()
genres_rating['Zero'] = 0
genres_rating
overall_mean = round(genres_rating['Genres Mean Rating'].mean(), 2)
overall_std = round(genres_rating['Genres Mean Rating'].std(),2)
scifi_rating = genres_rating[genres_rating['Genre'] == 'Sci-Fi']['Genres Mean Rating']
print(overall_mean)
print(overall_std)
print(scifi_rating)
genres_rating['Diff from Mean'] = genres_rating['Genres Mean Rating'] - overall_mean

genre_list = list(genres_rating['Genre'])
genres_rating_list = list(genres_rating['Genres Mean Rating'])
genres_diff_list = list(genres_rating['Diff from Mean'])

plt.figure(figsize=(20, 10))

ax1 = plt.subplot(2,1,1)
x = [x for x in range(0, 18)]
xticks_genre_list = genre_list
y = genres_rating_list
plt.xticks(range(len(x)), xticks_genre_list)
plt.scatter(x,y, color='g')
plt.plot(x, genres_rating['Mean'], color="red")
plt.autoscale(tight=True)
#plt.rcParams["figure.figsize"] = (10,2)
plt.title('Movie ratings by genre')
plt.xlabel('Genre')
plt.ylabel('Rating')
plt.ylim(ymax = 4, ymin = 3)
plt.grid(True)
plt.savefig(r'movie-ratings-by-genre.png')

plt.annotate("Sci-Fi Rating",
            xy=(14.25,3.5), xycoords='data',
            xytext=(14.20, 3.7), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"),
            )

for i,j in enumerate( y ):
    ax1.annotate( j, ( x[i] + 0.03, y[i] + 0.02))

ax2 = plt.subplot(2,1,2)
x = [x for x in range(0, 18)]
xticks_genre_list = genre_list
y = genres_rating['Diff from Mean']
plt.xticks(range(len(x)), xticks_genre_list)
plt.plot(x,y)
plt.plot(x, genres_rating['Zero'])
plt.autoscale(tight=True)
#plt.rcParams["figure.figsize"] = (10,2)
plt.title('Deviation of each genre\'s rating from the overall mean rating')
plt.xlabel('Genre')
plt.ylabel('Deviation from mean rating')
plt.grid(True)
plt.savefig(r'deviation-from-mean-rating.png')

plt.annotate("Sci-Fi Rating",
            xy=(14,-0.13), xycoords='data',
            xytext=(14.00, 0.0), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"),
            )


plt.show()

ratings.describe()
print (ratings['rating'].min())
print (ratings['rating'].max())

drama_movies=movies['genres'].str.contains('Drama')
drama_movies.shape

comedy_movies = movies['genres'].str.contains('Comedy')
comedy_movies.shape

tag_search = tags['tag'].str.contains('dark')

del ratings['timestamp']

movie_data_ratings_data=movies.merge(ratings,on = 'movieId',how = 'inner')

high_rated= movie_data_ratings_data['rating']>4.0

low_rated = movie_data_ratings_data['rating']<4.0

unique_genre=movies['genres'].unique().tolist()
len(unique_genre)

most_rated = movie_data_ratings_data.groupby('title').size().sort_values(ascending=False)[:25]

movies['year'] =movies['title'].str.extract('.*\((.*)\).*',expand = False)

def count_word(df, ref_col, liste):
    keyword_count = dict()
    for s in liste: keyword_count[s] = 0
    for liste_keywords in df[ref_col].str.split('|'):
        if type(liste_keywords) == float and pd.isnull(liste_keywords): continue
        for s in liste_keywords: 
            if pd.notnull(s): keyword_count[s] += 1
    # convert the dictionary in a list to sort the keywords  by frequency
    keyword_occurences = []
    for k,v in keyword_count.items():
        keyword_occurences.append([k,v])
    keyword_occurences.sort(key = lambda x:x[1], reverse = True)
    return keyword_occurences, keyword_count

genre_labels = set()
for s in movies['genres'].str.split('|').values:
    genre_labels = genre_labels.union(set(s))
    
keyword_occurences, dum = count_word(movies, 'genres', genre_labels)
keyword_occurences

def random_color_func(word=None, font_size=None, position=None,
                      orientation=None, font_path=None, random_state=None):
    h = int(360.0 * tone / 255.0)
    s = int(100.0 * 255.0 / 255.0)
    l = int(100.0 * float(random_state.randint(70, 120)) / 255.0)
    return "hsl({}, {}%, {}%)".format(h, s, l)
    
words = dict()
trunc_occurences = keyword_occurences[0:50]
for s in trunc_occurences:
    words[s[0]] = s[1]
tone = 100 # define the color of the words
f, ax = plt.subplots(figsize=(14, 6))
wordcloud = WordCloud(width=550,height=300, background_color='black', 
                      max_words=1628,relative_scaling=0.7,
                      color_func = random_color_func,
                      normalize_plurals=False)
wordcloud.generate_from_frequencies(words)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

fig = plt.figure(1, figsize=(18,13))
ax2 = fig.add_subplot(2,1,2)
y_axis = [i[1] for i in trunc_occurences]
x_axis = [k for k,i in enumerate(trunc_occurences)]
x_label = [i[0] for i in trunc_occurences]
plt.xticks(rotation=85, fontsize = 15)
plt.yticks(fontsize = 15)
plt.xticks(x_axis, x_label)
plt.ylabel("No. of occurences", fontsize = 24, labelpad = 0)
ax2.bar(x_axis, y_axis, align = 'center', color='r')
plt.title("Popularity of Genres",bbox={'facecolor':'k', 'pad':5},color='w',fontsize = 30)
plt.show()






