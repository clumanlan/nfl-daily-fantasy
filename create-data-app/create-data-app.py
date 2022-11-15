import pandas as pd
import numpy as np
import nfl_data_py as nfl
import datetime
import awswrangler as wr
import logging
import json
import boto3 
logging.basicConfig(level=logging.INFO)


def get_secret():

    secret_name = "dkuser_aws_keys"
    region_name = "us-east-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    get_secret_value_response = client.get_secret_value(SecretId=secret_name)

    secret_response = get_secret_value_response['SecretString']

    return json.loads(secret_response)

def load_data():
        
    current_year = datetime.date.today().year
    last_year = current_year - 1
        
    weekly_team_schedule = nfl.import_schedules(range(last_year, current_year))
    weekly_player_stats = nfl.import_weekly_data(range(last_year, current_year), downcast=True)
    weekly_player_depth_chart = nfl.import_depth_charts(range(last_year, current_year))

    return weekly_team_schedule, weekly_player_stats, weekly_player_depth_chart



def create_aggregate_rolling_functions(window_num = 10, window_min = 1):
    ## aggregate rolling functions to create summary stats
    f_min = lambda x: x.rolling(window=window_num, min_periods=window_min).min() 
    f_max = lambda x: x.rolling(window=window_num, min_periods=window_min).max()
    f_mean = lambda x: x.rolling(window=window_num, min_periods=window_min).mean()
    f_std = lambda x: x.rolling(window=window_num, min_periods=window_min).std()
    f_sum = lambda x: x.rolling(window=window_num, min_periods=window_min).sum()

    return f_min, f_max, f_mean, f_std, f_sum

# PROCESS WEEKLY SCHEDULE DF -------------------------------------------------------------

def process_team_schedule(df):

    team_schedule_col_id = [ 'season', 'week', 'game_type', 'gameday', 
        'gametime', 'div_game', 'roof', 'surface', 'temp', 'wind', 'stadium_id']

    team_schedule_col_values_away = ['away_rest',  'away_qb_id', 'away_team', 'away_score']
    team_schedule_col_values_home = ['home_team', 'home_score', 'home_rest', 'home_qb_id']

    df_away = df[team_schedule_col_id + team_schedule_col_values_away]
    df_away.columns = df_away.columns.str.removeprefix('away_')
    df_away['home_away'] = 'away'

    df_home = df[team_schedule_col_id + team_schedule_col_values_home]
    df_home.columns = df_home.columns.str.removeprefix('home_')
    df_home['home_away'] = 'home' 

    df_processed = pd.concat([df_away, df_home])
    df_processed.drop(['qb_id', 'score'], axis=1, inplace=True)
    
    return df_processed


# CREATE WEEKLY PLAYER STATS SPECIFIC FOR QB & POSITION DF -------------------------------------------------

def define_rel_weekly_num_cols():

    weekly_player_cols_num_all = ['completions', 'attempts', 'passing_yards', 'passing_tds', 
        'interceptions', 'sacks', 'sack_yards', 'sack_fumbles',
        'sack_fumbles_lost', 'passing_air_yards', 'passing_yards_after_catch',  
        'passing_first_downs', 'passing_epa', 'passing_2pt_conversions', 'pacr',
        'carries', 'rushing_yards', 'rushing_tds', 'rushing_fumbles',
        'rushing_fumbles_lost', 'rushing_first_downs', 'rushing_epa',
        'rushing_2pt_conversions', 'air_yards_share', 'wopr', 'receptions', 'targets', 'receiving_yards',
        'receiving_tds', 'receiving_fumbles', 'receiving_fumbles_lost',
        'receiving_air_yards', 'receiving_yards_after_catch',
        'receiving_first_downs', 'receiving_epa', 'receiving_2pt_conversions','special_teams_tds',
        'racr', 'target_share', 'fantasy_points', 'fantasy_points_ppr']

    weekly_player_cols_num_qb = ['completions', 'attempts', 'passing_yards', 'passing_tds', 
        'interceptions', 'sacks', 'sack_yards', 'sack_fumbles',
        'sack_fumbles_lost', 'passing_air_yards', 'passing_yards_after_catch',  
        'passing_first_downs', 'passing_epa', 'passing_2pt_conversions', 'pacr',
        'carries', 'rushing_yards', 'rushing_tds', 'rushing_fumbles',
        'rushing_fumbles_lost', 'rushing_first_downs', 'rushing_epa',
        'rushing_2pt_conversions', 'air_yards_share', 'wopr', 'fantasy_points', 'fantasy_points_ppr']

    weekly_player_cols_num_position = ['carries', 'rushing_yards', 'rushing_tds', 'rushing_fumbles',
        'rushing_fumbles_lost', 'rushing_first_downs', 'rushing_epa',
        'rushing_2pt_conversions', 'receptions', 'targets', 'receiving_yards',
        'receiving_tds', 'receiving_fumbles', 'receiving_fumbles_lost',
        'receiving_air_yards', 'receiving_yards_after_catch', 'air_yards_share',
        'receiving_first_downs', 'receiving_epa', 'receiving_2pt_conversions','special_teams_tds',
        'racr', 'target_share', 'wopr', 'fantasy_points', 'fantasy_points_ppr']

    return weekly_player_cols_num_all, weekly_player_cols_num_qb, weekly_player_cols_num_position


def filter_n_separate_weekly_player_stats(df, all_num_col, qb_num_col, pos_num_col):

    weekly_player_stats = df

    ## copy fantasypoints column to create a target variable
    weekly_player_stats['fantasy_points_actual'] = weekly_player_stats['fantasy_points']
    weekly_player_stats.rename({'recent_team':'team'}, axis=1, inplace=True)
    weekly_player_stats = weekly_player_stats.sort_values(['season', 'week'])
    weekly_player_stats['season'] = weekly_player_stats['season'].astype(str)

    weekly_player_cols_cat = ['player_id', 'player_name', 'position', 'position_group',
        'team', 'season', 'week']


    # filter for numeric variables needed for each subgroup since some stats (e.g. completions) don't make sense for other group
   
    weekly_player_stats_qb_n_position =  weekly_player_stats[weekly_player_cols_cat  + all_num_col +  ['fantasy_points_actual']]
    weekly_player_stats_qb = weekly_player_stats[weekly_player_stats['position_group']=="QB"][weekly_player_cols_cat  + qb_num_col + ['fantasy_points_actual']]
    weekly_player_stats_position =  weekly_player_stats[weekly_player_stats['position_group'].isin(["WR", "RB", "TE"])][weekly_player_cols_cat  + pos_num_col + ['fantasy_points_actual']]

    weekly_player_stats_qb_n_position[all_num_col] = weekly_player_stats_qb_n_position[all_num_col].fillna(0)
    weekly_player_stats_qb[qb_num_col] = weekly_player_stats_qb[qb_num_col].fillna(0)
    weekly_player_stats_position[pos_num_col] = weekly_player_stats_position[pos_num_col].fillna(0)

    return weekly_player_stats_qb_n_position, weekly_player_stats_qb, weekly_player_stats_position



def create_weekly_team_stats(weekly_player_stats_qb_n_position, weekly_player_cols_num_all):

    weekly_team_stats = (weekly_player_stats_qb_n_position # create weekly team agg stats
        .sort_values(['season', 'week'])
        .groupby(['team', 'season', 'week'])[weekly_player_cols_num_all]
        .apply(lambda x : x.sum())
        .reset_index()
    )

    weekly_team_stats = ( # lag stats
        weekly_team_stats.assign(**{
        f'team_{col}_lagged': weekly_team_stats.sort_values(['season', 'week']).groupby(['season','team'])[col].shift(1)
        for col in weekly_player_cols_num_all})
    )

    weekly_team_stats.drop(weekly_player_cols_num_all, axis=1, inplace=True)

    function_list = [f_min, f_max, f_mean, f_std, f_sum] # create summary stats from weekly stats
    function_name = ['min', 'max', 'mean', 'std', 'sum']

    for col in weekly_team_stats.columns[weekly_team_stats.columns.str.endswith('_lagged')]:
        print(col)
        for i in range(len(function_list)):
            weekly_team_stats[(col + '_%s' % function_name[i])] = weekly_team_stats.sort_values(['week']).groupby(['team', 'season'], group_keys=False)[col].apply(function_list[i])
            print(function_name[i])

    return weekly_team_stats


def create_weekly_position_stats(df, rel_num_cols):
    
    df = (
        df
        .sort_values(['season', 'week'])
        .groupby(['position_group', 'season', 'week'])[rel_num_cols]
        .apply(lambda x : x.sum())
        .reset_index()
    )

    df = (
        df.assign(**{
        f'qb_position_{col}_lagged': df.sort_values(['season', 'week']).groupby(['season', 'position_group'], group_keys=False)[col].shift(1)
        for col in rel_num_cols})
    )

    df.drop(rel_num_cols, axis=1, inplace=True)

    function_list = [f_min, f_max, f_mean, f_std, f_sum]
    function_name = ['min', 'max', 'mean', 'std', 'sum']

    for col in df.columns[df.columns.str.endswith('_lagged')]:
        print(col)

        for i in range(len(function_list)):
            df[(col + '_%s' % function_name[i])] = df.sort_values(['week']).groupby(['position_group', 'season'], group_keys=False)[col].apply(function_list[i])
            print(function_name[i])

    return df


def create_weekly_team_position_stats(df, rel_num_cols, pos):

    df = (
       df
       .groupby(['team','position_group', 'season', 'week'], group_keys=False)[rel_num_cols]
       .apply(lambda x : x.sum())
       .reset_index()
       .sort_values(['season', 'week'])
    )

    
    df = (
        df.assign(**{
        f'team_position_{col}_lagged_qb': df.sort_values(['season', 'week']).groupby(['team','position_group', 'season'], group_keys=False)[col].shift(1)
        for col in rel_num_cols})
    )

    df.drop(rel_num_cols, axis=1, inplace=True)

    function_list = [f_min, f_max, f_mean, f_std, f_sum]
    function_name = ['min', 'max', 'mean', 'std', 'sum']

    for col in df.columns[df.columns.str.endswith(f"_lagged_{pos}")]:
        print(col)
        for i in range(len(function_list)):
            df[(col + '_%s' % function_name[i])] = df.sort_values(['week']).groupby(['team', 'season'], group_keys=False)[col].apply(function_list[i])
            print(function_name[i])

    return df

def create_weekly_player_stats(df, rel_num_cols):

    df = (
        df.assign(**{
        f'player_{col}_lagged': df.sort_values(['season', 'week']).groupby(['player_id', 'season'], group_keys=False)[col].shift(1)
        for col in rel_num_cols})
        .reset_index(drop=True)
    )

    df.drop(rel_num_cols, axis=1, inplace=True)

    function_list = [f_min, f_max, f_mean, f_std, f_sum]
    function_name = ['min', 'max', 'mean', 'std', 'sum']

    for col in df.columns[df.columns.str.endswith('_lagged')]:
        print(col)
        for i in range(len(function_list)):
            df[(col + '_%s' % function_name[i])] = df.sort_values(['season','week']).groupby(['player_id', 'season'], group_keys=False)[col].apply(function_list[i])
            print(function_name[i])

    return df


def merge_data(weekly_stats_team, weekly_stats_pos, weekly_stats_team_pos, weekly_stats_player, weekly_team_schedule):
    
    processed_df = weekly_stats_player.merge(weekly_stats_team_pos, on=['season', 'week', 'team', 'position_group'])
    processed_df = processed_df.merge(weekly_stats_pos, on=['season', 'week', 'position_group'])
    processed_df = processed_df.merge(weekly_stats_team, on=['season', 'week', 'team'])

    processed_df.dropna(how='any', inplace=True) # we lose a lot of data due to players playing single games and all week 1 (due to lag) and 2 (due to aggregation) stats
    processed_df['season'] = processed_df['season'].astype('int')
    processed_df = processed_df.merge(weekly_team_schedule, on=['team', 'season', 'week'])

    return processed_df

def write_data_to_s3(weekly_stats_player_qb, weekly_stats_player_position):

    current_date = datetime.date.today().strftime('%Y-%m-%d')
    
    secret_dict = get_secret()
    aws_key_id = secret_dict['aws_access_key_id']
    aws_secret = secret_dict['aws_secret_access_key']

    session = boto3.Session(
        aws_access_key_id=aws_key_id,
        aws_secret_access_key=aws_secret)
    
    wr.s3.to_parquet(
            df=weekly_stats_player_qb,
            path="s3://nfl-daily-fantasy/data/processed/qb/weekly_stats_player_qb_{}.parquet".format(current_date),
            boto3_session=session
        )

    wr.s3.to_parquet(
            df=weekly_stats_player_position,
            path="s3://nfl-daily-fantasy/data/processed/position/weekly_stats_player_position_{}.parquet".format(current_date),
            boto3_session=session
        )



weekly_team_schedule, weekly_player_stats, weekly_player_depth_chart = load_data()
f_min, f_max, f_mean, f_std, f_sum = create_aggregate_rolling_functions()

    # DEF REL COLS AND BUILD OUT SEPARATE DFs  ------------------------------
weekly_player_cols_num_all, weekly_player_cols_num_qb, weekly_player_cols_num_position = define_rel_weekly_num_cols()
weekly_player_stats_qb_n_position, weekly_player_stats_qb, weekly_player_stats_position = filter_n_separate_weekly_player_stats(weekly_player_stats, weekly_player_cols_num_all, weekly_player_cols_num_qb, weekly_player_cols_num_position)

    # CREATE WEEKLY STATS AT DIFFERENT LEVELS OF AGGREGATION ------------------------------
weekly_stats_team = create_weekly_team_stats(weekly_player_stats_qb_n_position, weekly_player_cols_num_all)

weekly_stats_qb = create_weekly_position_stats(weekly_player_stats_qb, weekly_player_cols_num_qb)
weekly_stats_position = create_weekly_position_stats(weekly_player_stats_position, weekly_player_cols_num_position)

weekly_stats_team_qb = create_weekly_team_position_stats(weekly_player_stats_qb, weekly_player_cols_num_qb, "qb")
weekly_stats_team_position = create_weekly_team_position_stats(weekly_player_stats_position, weekly_player_cols_num_position, "pos")

weekly_stats_player_qb = create_weekly_player_stats(weekly_player_stats_qb, weekly_player_cols_num_qb)
weekly_stats_player_position = create_weekly_player_stats(weekly_player_stats_position, weekly_player_cols_num_position)

weekly_team_schedule_processed = process_team_schedule(weekly_team_schedule)



write_data_to_s3(weekly_stats_player_qb, weekly_stats_player_position)


def handler (event, context):
    # load data and define functions -----------------
    weekly_team_schedule, weekly_player_stats, weekly_player_depth_chart = load_data()
    f_min, f_max, f_mean, f_std, f_sum = create_aggregate_rolling_functions()

    # DEF REL COLS AND BUILD OUT SEPARATE DFs  ------------------------------
    weekly_player_cols_num_all, weekly_player_cols_num_qb, weekly_player_cols_num_position = define_rel_weekly_num_cols()
    weekly_player_stats_qb_n_position, weekly_player_stats_qb, weekly_player_stats_position = filter_n_separate_weekly_player_stats(weekly_player_stats, weekly_player_cols_num_all, weekly_player_cols_num_qb, weekly_player_cols_num_position)

    # CREATE WEEKLY STATS AT DIFFERENT LEVELS OF AGGREGATION ------------------------------
    weekly_stats_team = create_weekly_team_stats(weekly_player_stats_qb_n_position, weekly_player_cols_num_all)

    weekly_stats_qb = create_weekly_position_stats(weekly_player_stats_qb, weekly_player_cols_num_qb)
    weekly_stats_position = create_weekly_position_stats(weekly_player_stats_position, weekly_player_cols_num_position)

    weekly_stats_team_qb = create_weekly_team_position_stats(weekly_player_stats_qb, weekly_player_cols_num_qb, "qb")
    weekly_stats_team_position = create_weekly_team_position_stats(weekly_player_stats_position, weekly_player_cols_num_position, "pos")

    weekly_stats_player_qb = create_weekly_player_stats(weekly_player_stats_qb, weekly_player_cols_num_qb)
    weekly_stats_player_position = create_weekly_player_stats(weekly_player_stats_position, weekly_player_cols_num_position)

    write_data_to_s3(weekly_stats_player_qb, weekly_stats_player_position)


handler(None, None)

