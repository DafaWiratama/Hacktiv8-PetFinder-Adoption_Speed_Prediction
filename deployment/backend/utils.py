from sklearn.base import TransformerMixin


class CharacterCounter(TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x[x.columns[0]].str.len().fillna(0).astype(int).values[:, None]

    def get_params(self, deep=True):
        return {}


class WordCounter(TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x[x.columns[0]].str.split().str.len().fillna(0).astype(int).values[:, None]

    def get_params(self, deep=True):
        return {}