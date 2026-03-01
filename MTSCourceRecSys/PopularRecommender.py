import pandas as pd
from itertools import islice, cycle

class PopularRecommender():
    def __init__(self, max_K=100, days=30, item_column='item_id', dt_column='date'):
        self.max_K = max_K
        self.days = days
        self.item_column = item_column
        self.dt_column = dt_column
        self.recommendations = []
        
    def fit(self, df):
        min_date = df[self.dt_column].max().normalize() - pd.DateOffset(days=self.days)
        dt_mask = df[self.dt_column] > min_date
        self.recommendations = df.loc[dt_mask, self.item_column].value_counts().head(self.max_K).index.values
        
    def recommend(self, users=None, N=10):
        recs = self.recommendations[:N]
        if users is None:
            return recs
        else:
            return list(islice(cycle([recs]), len(users)))