from movierecommender.recommender import recommend_random


def test_recommend_random():
    liked_items = ['star trek', 'star wars']
    assert len(recommend_random(liked_items)) == 5
