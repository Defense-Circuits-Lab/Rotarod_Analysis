import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Tuple, List, Dict, Optional
import warnings

def get_df(filename: str, lines: Optional[List]=None, animals_to_exclude: Optional[List] = None, subgroup: Optional[str]=None, verbose: bool = True)->pd.DataFrame:
    """
    slices the dataframe to get a new dataframe containing only the specified lines (or all lines if not specified)
    """
    if filename.endswith(".xlsx"):
        df_raw = pd.read_excel(filename, dtype={"subgroup": str})
    elif filename.endswith(".csv"):
        df_raw = pd.read_csv(filename, dtype={"subgroup": str})
    else:
        raise ValueError ("File type not supported!")
        
    if animals_to_exclude is None:
        animals_to_exclude = []
        
    if subgroup is not None:
        if "subgroup" not in df_raw.keys():
            raise KeyError(f"The dataframe doesn't have a column 'subgroup'. Please set subgroup to None!")
        if subgroup not in df_raw['subgroup'].values:
            raise KeyError(f"The subgroup you specified is not in the subgroups in the dataframe!\n"
                           f"The following subgroups are in the dataframe: {df_raw['subgroup'].unique()}.")
        else:
            df_raw = df_raw.loc[df_raw['subgroup']==subgroup, :]
        
    if lines is None:
        animals = list(
        set(subject_id for subject_id in df_raw["subject_id"] if subject_id not in animals_to_exclude
        ))
    else:
        for line in lines:
            if not any([str(line) in subject_id for subject_id in df_raw["subject_id"]]):
                raise KeyError(f"The specified line {line} was not found in dataframe!")
                    
        animals = list(
        set(subject_id for subject_id in df_raw["subject_id"] if any([str(line) in subject_id for line in lines]) and subject_id not in animals_to_exclude
        ))
        
    df = df_raw.loc[df_raw["subject_id"].isin(animals), :]
    df.reset_index(inplace=True, drop=True)
    if verbose:
        print('Found the following subjects: ', animals)
    return df
    
def create_analysis_dfs(
    df: pd.DataFrame,
    sessions: Optional[List[str]]=None,
    analysis_types: Optional[List]=None,
    keep_dfs: bool =True,
    dropfirst: bool=False,
    baseline_session: Optional[str]=None
) -> pd.DataFrame:
    """
    takes input df and analyses mean, median, max, normalizes, then performs linear regression iterative for each animal and returns concated df
    """
    pd.options.mode.chained_assignment = None #removes dataframe slicing warnings

    animals = list(df["subject_id"].unique())
    
    if analysis_types == None:
        analysis_types = ['mean', 'max', 'median']
        
    if sessions == None:
        sessions = list(df["session_id"].unique())
    else:
        for session in sessions:
            if session not in df["session_id"]:
                raise KeyError(f"Could not find a session {session} in the dataframe!")
    if len(sessions) < 2:
        raise OverflowError("Can't run analysis on only one session!")
        
    if baseline_session is None:
        baseline_session = sessions[0]
        print(f"Since no session to use for baseline was specified, {baseline_session} will be used!")
    else:
        if baseline_session not in df["session_id"]:
            raise KeyError (f"Could not find a baseline session {baseline_session} in the dataframe!")
    
    trials = list(df["trial_id"].unique())
    num_trials = len(trials)
    if dropfirst:
            df = df.drop(df[df['trial_id'] == 1].index, axis = 0)
            num_trials = num_trials - 1
            
    new_size = (len(sessions), num_trials)
    df = df.loc[df['session_id'].isin(sessions), :]
    
    mean_dfs, median_dfs, max_dfs, mean_relative_dfs, median_relative_dfs, max_relative_dfs, mean_absolute_dfs, median_absolute_dfs, max_absolute_dfs  = [], [], [], [], [], [], [], [], []
    
    for animal in animals:
        df_work = df.loc[df['subject_id'] == animal, :]
        df_template = df_work.loc[df_work['trial_id'] == trials[-1], :]
        df_template.drop('trial_id', axis=1, inplace=True)
        df_template.reset_index(inplace=True, drop=True)
        
        if df_work.shape[0]%len(sessions) != 0:
            print(f'Invalid number of values for {animal}!')
            continue
        
        df_mean = df_template.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            df_mean['Data'] = np.nanmean(df_work['Data'].values.reshape(new_size), axis = 1)
        df_mean_relative, df_mean_absolute = perform_analysis(df_mean, baseline_session)
        mean_dfs.append(df_mean)
        mean_relative_dfs.append(df_mean_relative)
        mean_absolute_dfs.append(df_mean_absolute)

        df_median = df_template.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            df_median["Data"] = np.nanmedian(df_work['Data'].values.reshape(new_size), axis = 1)
        df_median_relative, df_median_absolute = perform_analysis(df_median, baseline_session)
        median_dfs.append(df_median)
        median_relative_dfs.append(df_median_relative)
        median_absolute_dfs.append(df_median_absolute)

        df_max = df_template.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            df_max["Data"] = df_work['Data'].values.reshape(new_size).max(axis = 1)
        df_max_relative, df_max_absolute = perform_analysis(df_max, baseline_session)
        max_dfs.append(df_max)
        max_relative_dfs.append(df_max_relative)
        max_absolute_dfs.append(df_max_absolute)

    all_dfs = {}
    if 'mean' in analysis_types:
        all_dfs['mean_raw'] = pd.concat(mean_dfs)
        if keep_dfs:
            all_dfs['mean_relative'] = pd.concat(mean_relative_dfs)
            all_dfs['mean_absolute'] = pd.concat(mean_absolute_dfs)
    if 'median' in analysis_types:
        all_dfs['median_raw'] = pd.concat(median_dfs)
        if keep_dfs:
            all_dfs['median_relative'] = pd.concat(median_relative_dfs)
            all_dfs['median_absolute'] = pd.concat(median_absolute_dfs)
    if 'max' in analysis_types:
        all_dfs['max_raw'] = pd.concat(max_dfs)
        if keep_dfs:
            all_dfs['max_relative'] = pd.concat(max_relative_dfs)
            all_dfs['max_absolute'] = pd.concat(max_absolute_dfs)
    for df in all_dfs.values():
        df.reset_index(inplace=True, drop=True)
    
    return all_dfs
    
    
def create_baseline(df_input: pd.DataFrame, baseline_session : str) -> pd.DataFrame:
    df_baseline = df_input.copy()
    df_baseline["baseline"] = df_input.loc[df_input["session_id"] == baseline_session, "Data"]

    df_baseline["baseline"].interpolate("pad", inplace=True)
    df_baseline.loc[:, "Data"] = (
        df_baseline.loc[:, "Data"] - df_baseline.loc[:, "baseline"]
    )
    return df_baseline


def normalize_df(df_input: pd.DataFrame) -> pd.DataFrame:
    df_normalized = df_input.copy()
    df_normalized["Data"] = ((df_normalized["Data"] / df_normalized["baseline"])*100)+100 # percentage of baseline
    return df_normalized


def perform_analysis(df: pd.DataFrame, baseline_session: str) -> Tuple[pd.DataFrame]:
    baseline_df = create_baseline(df.copy(), baseline_session)
    processed_df = normalize_df(baseline_df.copy())
    try:
        baseline_df.drop('baseline', axis = 1, inplace=True)
    except:
        pass
    try:
        processed_df.drop('baseline', axis = 1, inplace=True)
    except:
        pass
    return processed_df, baseline_df


def linearregression(df_input: pd.DataFrame) -> pd.DataFrame: #use this function with care! It seems to sort the data different than in the input!
    model = LinearRegression()
    scaler = StandardScaler()
    df_input.dropna(inplace=True, axis=0)
    x = df_input[["Data"]]
    x_scaled = scaler.fit_transform(x)
    y = np.array(
        [int(session[5:]) for session in df_input["session_id"].values]
    ).reshape(df_input.shape[0], 1)
    model.fit(x_scaled, y)
    x_pred = model.predict(y)
    coefficient = model.coef_
    r2_Score = r2_score(x_scaled, x_pred)
    df_input["Data"] = x_pred
    return df_input

