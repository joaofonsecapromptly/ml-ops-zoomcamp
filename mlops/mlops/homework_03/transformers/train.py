import mlflow

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "hw3-log-op-model"
MLFLOW_TRACKING_URI = "sqlite:///home/mlflow/mlflow.db"

mlflow.set_tracking_uri("http://mlflow:5000")
#mlflow.sklearn.autolog()
mlflow.set_experiment(EXPERIMENT_NAME)

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(df, *args, **kwargs):
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    target = 'duration'
    y_train = df[target].values

    with mlflow.start_run():
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_train)

        train_rmse = mean_squared_error(y_train, y_pred, squared=False)

        mlflow.sklearn.log_model(model, artifact_path="artifacts")
        mlflow.sklearn.log_model(dv, artifact_path="artifacts")

        print(f'Train RMSE: {train_rmse}')
        mlflow.log_metric("val_rmse", train_rmse)
        print(f'Intercept of the model {model.intercept_}' )

    return dv, model