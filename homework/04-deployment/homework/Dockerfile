FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim
RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY ["/data/yellow_tripdata_2023-05.parquet", "/app/data/"]

COPY ["Pipfile", "Pipfile.lock", "starter.py", "./"]

RUN pipenv install --system --deploy

ENTRYPOINT ["python", "starter.py"]