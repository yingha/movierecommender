"""
Contains various recommondation implementations
all algorithms return a list of movieids
"""
import numpy as np
import pandas as pd
from movierecommender.utils import movies,ratings,item_avg,final_rec,get_movie_id,get_user_item_matrix # you have already the data access of movies
import pickle


def recommend_random(liked_items, k=5):
    """
    return k random unseen movies for user 
    """
    # dummy implementation
    return movies.sample(k)

def recommend_most_popular(liked_items, item_avg, k=5):
    """
    return k most popular unseen movies for user
    """
    item_names, liked_item_ids = get_movie_id(liked_items)
    recommendations= item_avg.sort_values(ascending=False)
    item_filter = ~recommendations.index.isin(liked_item_ids) 
    recommendations = final_rec(recommendations,liked_item_ids,k)
    
    return recommendations

def personalized_recommender(liked_items, item_rating, k=5):
    '''
    gets a user list of watched movies, the item averages 
    and returns a list of k movie_ids based on user's preferance
    Neighborhood-based Collaborative Filtering (NearestNeighbors)
    '''
    item_names, liked_item_ids = get_movie_id(liked_items)
    with open('./models/movie_recommender_model1.pickle', 'rb') as file:
        model = pickle.load(file)
    user_vec = np.repeat(0, 193610)
    user_vec[liked_item_ids]=item_rating
    distances, user_ids = model.kneighbors([user_vec], n_neighbors=5)
    neighborhood = ratings.set_index('userId').loc[user_ids[0]]
    recommendations = neighborhood.groupby('movieId')['rating'].sum().sort_values(ascending=False)
    recommendations = final_rec(recommendations,liked_item_ids,k)
    return recommendations

def personalized_recommender_nmf(liked_items, item_rating, k=5):
    '''
    gets a user list of watched movies, the item averages 
    and returns a list of k movie_ids based on user's preferance
    Collaborative Filtering with Matrix Factorization
    '''
    with open('./models/movie_recommender_nmf.pickle', 'rb') as file:
        nmf = pickle.load(file)
    user_item_matrix = get_user_item_matrix()
    Q = nmf.components_ 
    movie_list = list(user_item_matrix.columns)
    item_names, liked_item_ids = get_movie_id(liked_items)
    user_movie_rating = dict(zip(liked_item_ids, item_rating))
    empty_list = [2.5]*len(movie_list)
    movie_dict = dict(zip(movie_list, empty_list))
    for movie, rating in user_movie_rating.items():
        movie_dict[movie]=rating
    movie_df = pd.DataFrame(list(movie_dict.values()), index=movie_dict.keys())
    movie_df = movie_df.T
    P = nmf.transform(movie_df)
    predictions = np.dot(P,Q)
    user_item_prediction = pd.DataFrame(predictions, columns=user_item_matrix.columns)
    recommendations = user_item_prediction[(movie_df == 2.5)].round(1).T
    recommendations.columns = ['predicted_rating']
    recommendations=recommendations['predicted_rating'].sort_values(ascending=False)
    print(recommendations.value_counts())
    recommendations = final_rec(recommendations,liked_item_ids,k)
    return recommendations


if __name__ == '__main__':
    #print(recommend_random([1, 2, 3]))
    liked_items = ['star trek', 'star wars', 'toy story', 'inside out', 'zootopia', 'coco (2017)']
    item_rating = ['4', '4', '2', '4', '4', '4']
    #print(personalized_recommender(liked_items, item_rating, k=5))
    #print(personalized_recommender_nmf(liked_items, item_rating, k=5))
    print(recommend_most_popular(liked_items, item_avg, k=5))





