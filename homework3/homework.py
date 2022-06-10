from datetime import datetime, timedelta
from pathlib import Path
from prefect import flow, get_run_logger, task
from prefect.task_runners import SequentialTaskRunner
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import requests

@task
def get_paths(date):
    given_date = datetime.strptime(date, '%Y-%m-%d') if date is not None else datetime.now()
    one_month_ago = given_date.date().replace(day=1) - timedelta(days=1)
    two_months_ago = one_month_ago.replace(day=1) - timedelta(days=1)
    one, two = one_month_ago.strftime('%Y-%m'), two_months_ago.strftime('%Y-%m')
    path_fmt = './data/fhv_tripdata_{}.parquet'

    return path_fmt.format(one), path_fmt.format(two)

@task
def download_nyc_for_hire_vehicle(save_path: str) -> str:
    save_path = Path(save_path)

    if not save_path.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fname = save_path.name
        r = requests.get(f'https://nyc-tlc.s3.amazonaws.com/trip+data/{fname}')

        with open(save_path, 'wb') as fout:
            for chunk in r.iter_content(chunk_size=1024):
                fout.write(chunk)
    logger = get_run_logger()
    logger.info('Downloaded file to %s', save_path)

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    logger = get_run_logger()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df.duration.values
    logger = get_run_logger()

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger = get_run_logger()
    logger.info(f"The MSE of validation is: {mse}")
    print(f'The MSE of validation is: {mse:.3f}')
    return

@flow(task_runner=SequentialTaskRunner())
def main(date=None):
    train_path, val_path = get_paths(date).result()
    categorical = ['PUlocationID', 'DOlocationID']

    download_nyc_for_hire_vehicle(train_path)
    download_nyc_for_hire_vehicle(val_path)

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)


if __name__ == '__main__':
    main(date='2021-08-15')
