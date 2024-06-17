import argparse
import pickle
import pandas as pd

# Set up command line arguments
parser = argparse.ArgumentParser(description='Process some data.')
parser.add_argument('--year', type=int, required=True, help='Year of the data')
parser.add_argument('--month', type=int, required=True, help='Month of the data')
parser.add_argument('--input', type=str, required=True, help='Input filename')
parser.add_argument('--output', type=str, required=True, help='Output filename')

args = parser.parse_args()

YEAR = args.year
MONTH = args.month
INPUT_FILE = args.input
OUTPUT_FILE = args.output

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

df = read_data(INPUT_FILE)

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)

df_result = pd.DataFrame(y_pred, columns=['prediction'])
df_result['ride_id'] = f'{YEAR:04d}/{MONTH:02d}_' + df.index.astype('str')

df_result.to_parquet(
    OUTPUT_FILE,
    engine='pyarrow',
    compression=None,
    index=False
)

print('Prediction results: \n', df_result['prediction'].describe())
