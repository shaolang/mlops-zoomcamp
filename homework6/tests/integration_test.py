from datetime import datetime
from pandas.testing import assert_frame_equal
import batch as b
import boto3
import os
import pandas as pd
import pytest

S3_ENDPOINT_URL = 'http://localhost.localstack.cloud:4566'
S3_BUCKET = 's3://nyc-duration'


def test_persist_mock_data_to_localstack_s3(mock_data_filename):
    client = boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT_URL,
    )
    item = client.list_objects_v2(Bucket='nyc-duration')['Contents'][0]

    assert 3_000 <= item['Size'] <= 4_000


# -- helper functions --

def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)


@pytest.fixture(scope='function')
def os_environ():
    original = os.environ
    patched = os.environ.copy()
    bucket_name = S3_BUCKET
    filename = '{year:04d}-{month:02d}.parquet'
    patched['INPUT_FILE_PATTERN'] =  bucket_name + '/in/' + filename
    patched['OUTPUT_FILE_PATTERN'] =  bucket_name + '/out/' + filename
    os.environ = patched

    yield

    os.environ = original


@pytest.fixture(scope='function')
def mock_data_filename(os_environ):
    options = dict(client_kwargs=dict(endpoint_url=S3_ENDPOINT_URL))
    data = [
        (None, None, dt(1, 2),    dt(1, 10)),
        (1,    1,    dt(1, 2),    dt(1, 10)),
        (1,    1,    dt(1, 2, 0), dt(1, 2, 50)),
        (1,    1,    dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = ['PULocationId', 'DOLocationID', 'pickup_datetime', 'dropOff_datetime']
    categorical = columns[:2]
    df = pd.DataFrame(data, columns=columns)
    df = pd.DataFrame(
        [[-1, -1, dt(1, 2), dt(1, 10), 8.],
         [*data[1], 8.]],
        columns=columns + ['duration']
    )
    df[categorical] = df[categorical].astype(str)
    input_file_pattern = os.getenv('INPUT_FILE_PATTERN')
    input_file = input_file_pattern.format(year=2021, month=1)

    df.to_parquet(
        input_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )

    return input_file
