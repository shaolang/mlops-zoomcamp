from datetime import datetime
from pandas.testing import assert_frame_equal
import batch as b
import pandas as pd


def test_prepare_data():
    data = [
        (None, None, dt(1, 2),    dt(1, 10)),
        (1,    1,    dt(1, 2),    dt(1, 10)),
        (1,    1,    dt(1, 2, 0), dt(1, 2, 50)),
        (1,    1,    dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = ['PULocationId', 'DOLocationID', 'pickup_datetime', 'dropOff_datetime']
    categorical = columns[:2]
    df = pd.DataFrame(data, columns=columns)
    expected = pd.DataFrame(
        [[-1, -1, dt(1, 2), dt(1, 10), 8.],
         [*data[1], 8.]],
        columns=columns + ['duration']
    )
    expected[categorical] = expected[categorical].astype(str)
    actual = b.prepare_data(df, categorical=categorical)

    assert_frame_equal(actual, expected)


# -- helper functions --

def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)
