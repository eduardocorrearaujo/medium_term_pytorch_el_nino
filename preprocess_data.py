import torch
import numpy as np
import pandas as pd
from epiweeks import Week
from scipy.stats import boxcox as fn_boxcox
THR = 0.1
# this list will be used for multiple functions 

df_pop_region = pd.read_csv('./data/pop_regional.csv')

df_env = pd.read_csv('data/regional_biome.csv.gz')


def load_cases_data(filename = None):
    '''
    Function that load the dataset of cases 
    '''
   
    df = pd.read_csv(filename)

    df.date = pd.to_datetime(df.date)

    df.set_index('date', inplace = True)

    return df 

def load_enso_weekly(filename = 'data/enso_weekly_forecast.csv'): 

    df = pd.read_csv(filename, index_col = 'index')

    df.index = pd.to_datetime(df.index)

    df = add_epiweek_label(df)

    return df

def add_epiweek_label(df_w):
    '''
    This function assumes that the dataframe has a datetime index
    and add the epiweek and year value
    '''

    df_w['epiweek_label'] = [Week.fromdate(x) for x in df_w.index]

    df_w['epiweek_label'] = df_w['epiweek_label'].astype(str)

    df_w = df_w.loc[df_w.epiweek_label.str[-2:].astype(int) != 53]

    df_w['epiweek'] = df_w['epiweek_label'].astype(str).str[-2:].astype(int)
    df_w['year'] = df_w['epiweek_label'].astype(str).str[:4].astype(int)

    return df_w

def aggregate_data(df, geocode = None, column = 'geocode'):
  '''
  Função para agregar os dados a partir de um geocode específico, se o geocode não
  é fornecido os dados são agregados para todo o estado.
  '''

  if geocode is not None:

    df = df.loc[df[column] == geocode]

  df_w = df[['casos']]

  df_w = df_w.resample('W-SUN').sum()

  df_w = add_epiweek_label(df_w)

  return df_w


def aggregate_data_media(df, geocode=None, column='geocode'):
    '''
    Função para agregar os dados a partir de um geocode específico.
    Se o geocode não é fornecido, os dados são agregados para todo o estado.
    Também adiciona uma média móvel de 3 semanas.
    '''

    if geocode is not None:
        df = df.loc[df[column] == geocode]

    df_w = df[['casos']]

    # agregação semanal
    df_w = df_w.resample('W-SUN').sum()

    # média móvel de 3 semanas
    df_w['casos'] = (
        df_w['casos']
        .rolling(window=2, min_periods=1)
        .mean()
    )

    # adiciona label epidemiológica
    df_w = add_epiweek_label(df_w)

    return df_w


def prepare_regional_data(
    df,
    geo,
    columns_to_normalize,
    enso=None,
    boxcox=False,
    media=False
):
    """
    Prepare regional dataframe.
    """

    # -------------------------------------------
    # Aggregate
    # -------------------------------------------
    if media:
        df_w = aggregate_data_media(
            df,
            geo,
            column='regional_geocode'
        )
    else:
        df_w = aggregate_data(
            df,
            geo,
            column='regional_geocode'
        )

    # -------------------------------------------
    # Box-Cox
    # -------------------------------------------
    if boxcox:
        df_w['casos'] = fn_boxcox(
            df_w['casos'] + 1,
            THR
        )

    # -------------------------------------------
    # Extra features
    # -------------------------------------------
    df_w['pop_norm'] = (
        df_pop_region.loc[
            df_pop_region.regional_geocode == geo,
            'pop_norm'
        ].values[0]
    )

    if 'biome' in columns_to_normalize:

        df_w['biome'] = (
            df_env.loc[
                df_env.regional_geocode == geo,
                'biome'
            ].values[0]
        )

    # -------------------------------------------
    # Merge ENSO
    # -------------------------------------------
    if enso is not None:

        df_w = df_w.merge(
            enso[['enso']],
            left_index=True,
            right_index=True,
            how='left'
        )

    return df_w.dropna()

def gen_forecast_dates(year, max_epiweek):
    '''
    Function to gen the date of the forecasted 41-40 weeks.
    '''
    
    dates = []
    for week in np.arange(max_epiweek+1, 53):

        dates.append(Week(year, week).startdate())

    return dates 

def _build_input_window(df, year, max_epiweek, columns):
    """
    Monta a janela temporal de entrada.
    """
    mask = (
        ((df.year < year) & (df.year >= year - 1))
        | ((df.year == year) & (df.epiweek <= max_epiweek / 49.4))
    )

    values = []

    for col in columns:
        arr = (
            df.loc[mask]
            .sort_index()[col]
            .values.reshape(-1, 1)
        )

        values.append(arr)

    return np.concatenate(values, axis=1)


def get_data(
    df_w,
    columns_to_normalize=['casos', 'epiweek', 'enso'],
    max_epiweek=25,
    min_year=None,
    enso=None
):
    """
    Retorna dados de treinamento usando normalização robusta por percentil 95.
    """

    forecast_weeks = 52 - max_epiweek

    df_w = df_w.copy()

    # SUBSTITUÍDO: .max() por .quantile(0.95) robusto
    norm_values = df_w[columns_to_normalize].quantile(0.95)
    # Garante que não haverá divisão por zero ou valores negativos
    norm_values = norm_values.clip(lower=1e-5)

    df_w[columns_to_normalize] = (
        df_w[columns_to_normalize] / norm_values
    )

    norm_enso = None

    if enso is not None:
        enso = enso.copy()

        # SUBSTITUÍDO: .max() por .quantile(0.95) robusto para o ENSO
        norm_enso = enso['enso'].quantile(0.95)
        if norm_enso <= 0:
            norm_enso = 1.0

        enso['enso'] = enso['enso'] / norm_enso

    if min_year is None:
        min_year = df_w.index.year.min() + 2
    else:
        min_year = max(min_year, df_w.index.year.min() + 2)

    columns = columns_to_normalize + ['pop_norm']

    X_train = []
    y_train = []
    X_future = []

    for year in range(min_year, df_w.index.year.max() + 1):

        X = _build_input_window(
            df=df_w,
            year=year,
            max_epiweek=max_epiweek,
            columns=columns
        )

        y = (
            df_w.loc[
                (df_w.year == year)
                & (df_w.epiweek > max_epiweek / 49.4),
                'casos'
            ]
            .values
        )

        if y.sum()*norm_values['casos'] >= 10:

            X_train.append(X)
            y_train.append(y)

            if enso is not None:

                future = (
                    enso.loc[
                        (enso.index.year == year)
                        & (enso.epiweek > max_epiweek),
                        'enso'
                    ]
                    .values
                )

                X_future.append(future)

    X_train = torch.tensor(
        np.array(X_train),
        dtype=torch.float32
    )

    y_train = torch.tensor(
        np.array(y_train),
        dtype=torch.float32
    )

    if enso is not None:

        X_future = torch.tensor(
            np.array(X_future),
            dtype=torch.float32
        )

    else:
        X_future = None

    return X_train, y_train, X_future, norm_values, norm_enso


def get_single_data(
    df_w,
    year,
    norm_values,
    max_epiweek=25,
    columns_to_normalize=['casos', 'epiweek', 'enso'],
    enso=None,
    norm_enso=None
):
    """
    Retorna dados de teste para um único ano.
    """

    forecast_weeks = 52 - max_epiweek

    df_w = df_w.copy()

    df_w[columns_to_normalize] = (
        df_w[columns_to_normalize] / norm_values
    )

    if enso is not None:

        enso = enso.copy()

        enso['enso'] = enso['enso'] / norm_enso

    columns = columns_to_normalize + ['pop_norm']

    X = _build_input_window(
        df=df_w,
        year=year,
        max_epiweek=max_epiweek,
        columns=columns
    )

    y = (
        df_w.loc[
            (df_w.year == year)
            & (df_w.epiweek > max_epiweek / 49.4),
            'casos'
        ]
        .values
    )

    X = torch.tensor(
        X[np.newaxis, :, :],
        dtype=torch.float32
    )

    y = torch.tensor(
        y[np.newaxis, :],
        dtype=torch.float32
    )

    X_future = None

    if enso is not None:

        future = (
            enso.loc[
                (enso.index.year == year)
                & (enso.epiweek > max_epiweek),
                'enso'
            ]
            .values
        )

        X_future = torch.tensor(
            future[np.newaxis, :],
            dtype=torch.float32
        )

    return X, y, X_future


def generate_regional_train_samples(
    df,
    enso,
    test_year,
    max_epiweek=25,
    columns_to_normalize=['casos', 'epiweek', 'enso'],
    min_year=None,
    boxcox=False,
    media=False
):
    """
    Generate train samples from all health regions.
    """

    forecast_weeks = 52 - max_epiweek

    X_train_all = []
    y_train_all = []
    X_future_all = []

    norm_values = {}
    norm_enso = None

    for geo in df.regional_geocode.unique():

        # --------------------------------------------------
        # Aggregate regional data
        # --------------------------------------------------
        if media:
            df_w = aggregate_data_media(
                df,
                geo,
                column='regional_geocode'
            )
        else:
            df_w = aggregate_data(
                df,
                geo,
                column='regional_geocode'
            )

        # --------------------------------------------------
        # Optional Box-Cox transform
        # --------------------------------------------------
        if boxcox:
            df_w['casos'] = fn_boxcox(df_w['casos'] + 1, THR)

        # --------------------------------------------------
        # Additional features
        # --------------------------------------------------
        df_w['pop_norm'] = (
            df_pop_region.loc[
                df_pop_region.regional_geocode == geo,
                'pop_norm'
            ].values[0]
        )

        if 'biome' in columns_to_normalize:

            df_w['biome'] = (
                df_env.loc[
                    df_env.regional_geocode == geo,
                    'biome'
                ].values[0]
            )

        # --------------------------------------------------
        # Merge ENSO if available
        # --------------------------------------------------
        data = df_w.copy()

        if enso is not None:

            data = data.merge(
                enso[['enso']],
                left_index=True,
                right_index=True,
                how='left'
            )

        # --------------------------------------------------
        # Generate train data
        # --------------------------------------------------
        X_train_, y_train_, X_future_, norm_values_, norm_enso_ = get_data(
            df_w=data.loc[data.year < test_year],
            columns_to_normalize=columns_to_normalize,
            max_epiweek=max_epiweek,
            min_year=min_year,
            enso=(
                enso.loc[enso.index.year <= test_year]
                if enso is not None else None
            )
        )

        # --------------------------------------------------
        # Store normalization values
        # --------------------------------------------------
        norm_values[geo] = norm_values_['casos']

        if norm_enso is None:
            norm_enso = norm_enso_

        # --------------------------------------------------
        # Append tensors
        # --------------------------------------------------
        X_train_all.append(X_train_)
        y_train_all.append(y_train_)

        if X_future_ is not None:
            X_future_all.append(X_future_)

    # ------------------------------------------------------
    # Concatenate all regions
    # ------------------------------------------------------
    X_train = torch.cat(X_train_all, dim=0)

    y_train = torch.cat(y_train_all, dim=0)

    X_future = None

    if len(X_future_all) > 0:
        X_future = torch.cat(X_future_all, dim=0)

    return X_train.float(), y_train.float(), X_future, norm_values, norm_enso
