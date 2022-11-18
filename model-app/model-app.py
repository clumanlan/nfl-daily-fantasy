import mlflow
import pandas as pd
import boto3
import awswrangler as wr
from datetime import date
import json
import sklearn


test = wr.s3.read_parquet(path=["s3://nfl-daily-fantasy/data/2022-11-18.parquet"])

def handler (event, context):

    qb_processed = wr.s3.read_parquet(path=["s3://nfl-daily-fantasy/data/processed/qb/weekly_stats_player_qb_10week_agg_2022-11-15.parquet"])

    keys = ['MLmodel', 'conda.yaml', 'model.pkl', 'python_env.yaml', 'requirements.txt']
    
    client_s3 = boto3.client("s3")
    nfl_bucket = 'nfl-daily-fantasy'
    for key in keys:
        client_s3.download_file(nfl_bucket, 'model/' + key, '/tmp/' + key)

    model = mlflow.pyfunc.load_model('/tmp/')
    qb_preds = model.predict(qb_processed)
    qb_preds = pd.Series(qb_preds, name='predicted_fp')

    qb_preds_df = pd.concat([qb_processed[['player_name', 'team']], qb_preds], axis=1)

 
    session = boto3.Session()
    current_date = date.today().strftime('%Y-%m-%d')

    wr.s3.to_parquet(
        df=qb_preds_df,
        path="s3://nfl-daily-fantasy/data/model-output/qb/fantasy_point_preds_qb_{}.parquet".format(current_date),
        boto3_session=session
        )

handler(None, None)