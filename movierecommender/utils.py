import pandas as pd
import pickle
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import NMF
from fuzzywuzzy import process
import os

package_dir = os.path.dirname(__file__)
print(package_dir)

# put the movieId into the row index!
movies = pd.read_csv(package_dir + '/data/ml-latest-small/movies.csv', index_col=0)  
ratings = pd.read_csv(package_dir + '/data/ml-latest-small/ratings.csv')
item_avg = ratings.groupby(['movieId'])['rating'].sum()
                  

def lookup_movie(search_query, titles):
    """
    given a search query, uses fuzzy string matching to search for similar 
    strings in a pandas series of movie titles

    returns a list of search results. Each result is a tuple that contains 
    the title, the matching score and the movieId.
    """
    matches = process.extractBests(search_query, titles, score_cutoff=90)
    return matches

def get_movie_review(movieIds):
    item_review = ratings.groupby(['movieId'])['rating'].aggregate(['mean','count'])
    return item_review.loc[movieIds]

def get_movie_id(liked_items):
    """
    given a list of liked_items
    return the item_names and liked_item_ids
    one item_name only returns one liked_item_id
    """
    item_names = []
    liked_item_ids=[]
    for item in liked_items:
        item_name = lookup_movie(item, movies['title'])
        item_name = item_name[0][0]
        item_names.append(item_name)
        movie_filter = movies['title']==item_name
        item_id = movies[movie_filter].index[0]
        liked_item_ids.append(item_id)
    return item_names, liked_item_ids

def final_rec(recommendations,liked_item_ids,k):
    '''
    take predicted user_item_matrix
    return top k unseen movies for user
    '''
    item_filter = ~recommendations.index.isin(liked_item_ids) 
    recommendations = recommendations.loc[item_filter]
    recommendations = movies.loc[recommendations.head(k).index]
    return recommendations

def get_user_item_matrix():
    user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')
    user_item_matrix.fillna(2.5, inplace=True)
    return user_item_matrix

def train_nn_model(metric):
    '''
    train model for movie recommender
    Neighborhood-based Collaborative Filtering (NearestNeighbors)
    '''
    user_item = csr_matrix((ratings['rating'], (ratings['userId'], ratings['movieId'])))
    model = NearestNeighbors(metric=metric)
    model.fit(user_item)
    return model

def train_nmf_model(n_components,init, max_iter):
    '''
    train model for movie recommender
    Collaborative Filtering with Matrix Factorization
    the 
    '''
    user_item_matrix = get_user_item_matrix()
    nmf = NMF(n_components=n_components, init=init,max_iter=max_iter, tol=0.001, verbose=True)
    nmf.fit(user_item_matrix)
    print('nmf.reconstruction_err_',nmf.reconstruction_err_)
    return nmf

if __name__ == '__main__':
    '''
    results = process.extractBests('baby yoda', movies['title'])
    # [(title, score, movieId), ...]
    print(results)    

    liked_items = ['star trek', 'star wars', 'toy story','shawshank redemption']
    item_names, liked_item_ids = get_movie_id(liked_items)
    print(item_names)
    print(liked_item_ids)

    model = train_model('cosine')
    with open('./models/movie_recommender_model1.pickle', 'wb') as file:
        pickle.dump(model, file)

    user_item_matrix = get_user_item_matrix()
    print(user_item_matrix)
    

    model = train_nmf_model(n_components=55, init='nndsvd',max_iter=10000)
    with open('./models/movie_recommender_nmf.pickle', 'wb') as file:
        pickle.dump(model, file)
    
    '''
    movieIds = [260, 1196, 1210, 2628, 5378, 1, 3114, 78499]
    print(get_movie_review(movieIds))

