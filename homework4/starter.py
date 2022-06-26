from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import pandas as pd
import pickle

with open(Path(__file__).parent / 'model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)

categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('year', type=int)
    parser.add_argument('month', type=int)
    ns = parser.parse_args()
    year = ns.year
    month = ns.month

    df = read_data(f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year}-{month:02}.parquet')
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    print(f'mean predicted duration: {np.mean(y_pred):.2f}')
