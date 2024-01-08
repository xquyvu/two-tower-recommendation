import pickle
import pathlib

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import (
    LabelEncoder,
    OrdinalEncoder,
    FunctionTransformer,
)
from sklearn.model_selection import train_test_split

import utils

DATA_FOLDER = pathlib.Path('data/ml-1m')
ARTIFACT_FOLDER = pathlib.Path('artifacts')

ARTIFACT_FOLDER.mkdir(exist_ok=True)

ratings = pd.read_csv(DATA_FOLDER / 'ratings.dat', sep='::', names=['userId', 'movieId', 'rating', 'timestamp'])
movies = pd.read_csv(DATA_FOLDER / 'movies.dat', sep='::', names=['movieId', 'title', 'genres'], encoding='ISO-8859-1') # Because this file wasn't UTF encoded
users = pd.read_csv(DATA_FOLDER / 'users.dat', sep='::', names=['userId', 'Gender', 'Age', 'Occupation', 'Zipcode',])

# Join
data = (
    ratings
    .merge(movies, on='movieId')
    .merge(users, on='userId')
)

data['Age'] = data['Age'].replace(utils.load_age_lookup())
data['Occupation'] = data['Occupation'].replace(utils.load_occupation_lookup())

# Numerical encode categorical variables
def encode_genres(data: pd.DataFrame) -> pd.DataFrame:
    genres = data['genres'].apply(lambda x: x.split('|')).explode().drop_duplicates()
    genre_encoder = LabelEncoder().fit(genres)
    return data['genres'].apply(lambda x: genre_encoder.transform(x.split('|')))[:, np.newaxis] # type: ignore[no-any-return]

def decode_genres(data: pd.DataFrame) -> pd.DataFrame:
    # TODO: Implement
    return data


categorical_encoder = make_column_transformer(
    (OrdinalEncoder(), ['Age']),
    (OrdinalEncoder(), ['Occupation']),
    (OrdinalEncoder(), ['Gender']),
    (FunctionTransformer(encode_genres, check_inverse=False, feature_names_out='one-to-one'), ['genres']),
    remainder='passthrough',
    verbose_feature_names_out=False,
)

categorical_encoder = categorical_encoder.fit(data)

# Save the encoder for later use
with open(ARTIFACT_FOLDER / 'categorical_encoder.pkl', 'wb') as f:
    pickle.dump(categorical_encoder, f)

# Create the dataset for modelling
model_dataset = (
    pd.DataFrame(
        categorical_encoder.transform(data),
        columns=categorical_encoder.get_feature_names_out(),
    )
    .drop(columns=['title', 'Zipcode'])
)

# Split into train and test, taking the last 2 ratings per user as test
test_indices = model_dataset.sort_values('timestamp').groupby('userId').tail(2).index

train = model_dataset.drop(index=test_indices)
test = model_dataset.loc[test_indices, :]

features_train = train.drop(columns=['rating'])
labels_train = train['rating']
features_test = test.drop(columns=['rating'])
labels_test = test['rating']

# Save features and labels for modelling
MODEL_DATA_FOLDER = pathlib.Path('data/modelling')
MODEL_DATA_FOLDER.mkdir(exist_ok=True)

for data_split, dataset in zip(
    ['train', 'test'], [(features_train, labels_train), (features_test, labels_test)]
):
    with open(MODEL_DATA_FOLDER / f'{data_split}.pkl', 'wb') as f:
        pickle.dump(dataset, f)
