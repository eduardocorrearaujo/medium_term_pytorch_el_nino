import os
import torch 
import numpy as np
import pandas as pd 
import torch.nn as nn
import preprocess_data as prep
import torch.nn.functional as F
from scipy.special import inv_boxcox
from scipy.stats import boxcox as fn_boxcox
from loss_func import WISLossFromDistribution
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset, Subset
import preprocess_data as prep

THR = 0.1

class EarlyStopping:

    def __init__(self, patience=20, min_delta=0.0):

        self.patience = patience
        self.min_delta = min_delta

        self.counter = 0
        self.best_loss = float('inf')

        self.early_stop = False

    def __call__(self, val_loss):

        if val_loss < self.best_loss - self.min_delta:

            self.best_loss = val_loss
            self.counter = 0

        else:

            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True




###########################################################
# SHARED LSTM ENCODER
###########################################################

class LSTMEncoder(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=64,
        dropout=0.2
    ):

        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.lstm2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x, _ = self.lstm1(x)

        x = self.dropout(x)

        x, _ = self.lstm2(x)

        x = self.dropout(x)

        return F.gelu(x[:, -1])


###########################################################
# BASE LOGNORMAL MODEL
###########################################################

class LSTMLogNormalModel(nn.Module):

    def __init__(
        self,
        features,
        predict_n,
        hidden=64,
        dropout=0.2
    ):

        super().__init__()

        self.encoder = LSTMEncoder(
            input_size=features,
            hidden_size=hidden,
            dropout=dropout
        )

        self.fc_mu = nn.Linear(
            hidden,
            predict_n
        )

        self.fc_sigma = nn.Linear(
            hidden,
            predict_n
        )

    def forward(self, x):

        h = self.encoder(x)

        mu = self.fc_mu(h)

        sigma = F.softplus(
            self.fc_sigma(h)
        ) + 1e-6

        return mu, sigma


###########################################################
# FUTURE COVARIATES MODEL
###########################################################

class LSTMWithFutureCovariatesV2(nn.Module):

    def __init__(
        self,
        past_features,
        future_cov_size,
        predict_n,
        hidden=64,
        dropout=0.2
    ):

        super().__init__()

        self.encoder = LSTMEncoder(
            input_size=past_features,
            hidden_size=hidden,
            dropout=dropout
        )

        ###################################################
        # BASE DISTRIBUTION
        ###################################################

        self.fc_mu = nn.Linear(
            hidden,
            predict_n
        )

        self.fc_sigma = nn.Linear(
            hidden,
            predict_n
        )

        ###################################################
        # FUTURE UPDATES
        ###################################################

        self.future_mu = nn.Linear(
            future_cov_size,
            predict_n
        )

        self.future_sigma = nn.Linear(
            future_cov_size,
            predict_n
        )

        ###################################################
        # GATES
        ###################################################

        self.gate_mu = nn.Sequential(
            nn.Linear(
                future_cov_size,
                predict_n
            ),
            nn.Sigmoid()
        )

        self.gate_sigma = nn.Sequential(
            nn.Linear(
                future_cov_size,
                predict_n
            ),
            nn.Sigmoid()
        )

    def forward(
        self,
        x_past,
        x_future
    ):

        ###################################################
        # PAST REPRESENTATION
        ###################################################

        h = self.encoder(x_past)

        ###################################################
        # BASE DISTRIBUTION
        ###################################################

        mu_base = self.fc_mu(h)

        sigma_base = F.softplus(
            self.fc_sigma(h)
        ) + 1e-6

        ###################################################
        # FUTURE UPDATES
        ###################################################

        delta_mu = self.future_mu(x_future)

        delta_sigma = F.softplus(
            self.future_sigma(x_future)
        )

        ###################################################
        # GATED UPDATE
        ###################################################

        mu = mu_base + (
            self.gate_mu(x_future)
            * delta_mu
        )

        sigma = sigma_base + (
            self.gate_sigma(x_future)
            * delta_sigma
        )

        sigma = F.softplus(sigma) + 1e-6

        return mu, sigma
    
# ===============================
# Train model 
# ==============================

def train_model(
    model,

    X_train,
    Y_train,

    X_future=None,

    label='model',

    batch_size=32,
    epochs=50,

    patience=20,
    min_delta=0.0,

    lr=1e-3,

    criterion=WISLossFromDistribution(),

    val_size=0.25,

    verbose=1,

    overwrite=True,
    save=True,

    doenca='dengue',

    device='cuda'
    if torch.cuda.is_available()
    else 'cpu'
):
    """
    Generic training function for:
    - regular models
    - encoder-decoder models

    Parameters
    ----------
    X_future : torch.Tensor or None
        Future covariates for encoder-decoder models.
    """

    # =====================================================
    # DEVICE
    # =====================================================

    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    # =====================================================
    # TRAIN / VAL SPLIT
    # =====================================================

    if X_future is None:

        (
            X_tr,
            X_val,

            y_tr,
            y_val

        ) = train_test_split(

            X_train,
            Y_train,

            test_size=val_size,

            random_state=7
        )

        train_dataset = TensorDataset(
            X_tr,
            y_tr
        )

        val_dataset = TensorDataset(
            X_val,
            y_val
        )

    else:

        (
            X_tr,
            X_val,

            Xf_tr,
            Xf_val,

            y_tr,
            y_val

        ) = train_test_split(

            X_train,
            X_future,
            Y_train,

            test_size=val_size,

            random_state=7
        )

        train_dataset = TensorDataset(
            X_tr,
            Xf_tr,
            y_tr
        )

        val_dataset = TensorDataset(
            X_val,
            Xf_val,
            y_val
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # =====================================================
    # EARLY STOPPING
    # =====================================================

    early_stopping = EarlyStopping(
        patience=patience,
        min_delta=min_delta
    )

    # =====================================================
    # EPOCH FUNCTION
    # =====================================================

    def run_epoch(loader, training=True):

        if training:
            model.train()
        else:
            model.eval()

        total_loss = 0.0

        with torch.set_grad_enabled(training):

            for batch in loader:

                # -----------------------------------------
                # SIMPLE MODEL
                # -----------------------------------------
                if X_future is None:

                    X_batch, y_batch = batch

                    X_batch = (
                        X_batch
                        .float()
                        .to(device)
                    )

                    y_batch = (
                        y_batch
                        .float()
                        .to(device)
                    )

                    inputs = [X_batch]

                # -----------------------------------------
                # ENCODER-DECODER
                # -----------------------------------------
                else:

                    (
                        X_batch,
                        X_future_batch,
                        y_batch
                    ) = batch

                    X_batch = (
                        X_batch
                        .float()
                        .to(device)
                    )

                    X_future_batch = (
                        X_future_batch
                        .float()
                        .to(device)
                    )

                    y_batch = (
                        y_batch
                        .float()
                        .to(device)
                    )

                    inputs = [
                        X_batch,
                        X_future_batch
                    ]

                # -----------------------------------------
                # FORWARD
                # -----------------------------------------
                if training:
                    optimizer.zero_grad()

                mu, sigma = model(*inputs)

                loss = criterion(
                    mu,
                    sigma,
                    y_batch
                )

                # -----------------------------------------
                # BACKPROP
                # -----------------------------------------
                if training:

                    loss.backward()

                    optimizer.step()

                total_loss += loss.item()

        return total_loss / len(loader)

    # =====================================================
    # TRAIN LOOP
    # =====================================================

    history = {
            'train_loss': [],
            'val_loss': []
        }
    
    for epoch in range(epochs):

        train_loss = run_epoch(
            train_loader,
            training=True
        )

        val_loss = run_epoch(
            val_loader,
            training=False
        )

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # -------------------------------------------------
        # VERBOSE
        # -------------------------------------------------
        if verbose:

            print(
                f'Epoch {epoch+1}/{epochs} | '
                f'Train: {train_loss:.4f} | '
                f'Val: {val_loss:.4f}'
            )

        # -------------------------------------------------
        # EARLY STOPPING
        # -------------------------------------------------
        early_stopping(val_loss)

        if early_stopping.early_stop:

            if verbose:

                print(
                    f'Early stopping '
                    f'at epoch {epoch+1}'
                )

            break

    # =====================================================
    # SAVE
    # =====================================================

    if save:

        os.makedirs(
            'saved_models',
            exist_ok=True
        )

        filename = (
            f'saved_models/'
            f'trained_{doenca}_{label}.pt'
        )

        if overwrite or not os.path.exists(filename):

            torch.save(
                model.state_dict(),
                filename
            )

            if verbose:

                print(
                    f'Model saved to {filename}'
                )

    return model, history 

#============================================
# Apply model 
#===========================================

def evaluate_samples(
    model,
    X_past,
    X_future=None,
    n_passes=100,
    mc_dropout=True
):
    """
    Generate predictive samples from a probabilistic model.

    Parameters
    ----------
    model : torch.nn.Module

    X_past : torch.Tensor

    X_future : torch.Tensor or None
        Optional future covariates.

    n_passes : int
        Number of stochastic forward passes.

    mc_dropout : bool
        Enable dropout during inference.

    Returns
    -------
    np.ndarray
        Shape:
            (n_passes, N, predict_n)
        or squeezed to:
            (n_passes, predict_n)
        when N == 1
    """

    device = next(model.parameters()).device

    X_past = X_past.float().to(device)

    inputs = [X_past]

    if X_future is not None:

        X_future = X_future.float().to(device)

        inputs.append(X_future)

    # ---------------------------------------------------
    # Dropout inference
    # ---------------------------------------------------
    if mc_dropout:

        def enable_dropout(m):
            if isinstance(m, torch.nn.Dropout):
                m.train()

        model.eval()
        model.apply(enable_dropout)

    else:
        model.train()

    predictions = []

    # ---------------------------------------------------
    # Sampling
    # ---------------------------------------------------
    with torch.no_grad():

        for _ in range(n_passes):

            mu, sigma = model(*inputs)

            dist = torch.distributions.LogNormal(
                mu,
                sigma
            )

            samples = dist.rsample()

            predictions.append(
                samples.detach().cpu().numpy()
            )

    predicted = np.stack(
        predictions,
        axis=0
    )

    # Remove singleton batch dim
    if predicted.shape[1] == 1:

        predicted = np.squeeze(
            predicted,
            axis=1
        )

    return predicted


def samples_to_quantiles(predicted, dates):
    """
    Convert samples into forecast dataframe.
    """

    df_preds = pd.DataFrame({
        'pred': np.percentile(predicted, 50, axis=0),

        'lower_50': np.percentile(predicted, 25, axis=0),
        'upper_50': np.percentile(predicted, 75, axis=0),

        'lower_80': np.percentile(predicted, 10, axis=0),
        'upper_80': np.percentile(predicted, 90, axis=0),

        'lower_90': np.percentile(predicted, 5, axis=0),
        'upper_90': np.percentile(predicted, 95, axis=0),

        'lower_95': np.percentile(predicted, 2.5, axis=0),
        'upper_95': np.percentile(predicted, 97.5, axis=0),

        'date': pd.to_datetime(dates)
    })

    return df_preds


def sum_regions_predictions(
    model,
    df,
    enso,
    test_year,
    columns_to_normalize,
    max_epiweek=25,
    boxcox=False,
    n_passes=500,
    min_year=None,
    media=False,
    return_samples=False
):
    """
    Apply model to all health regions and sum predictions.

    Parameters
    ----------
    return_samples : bool
        If True returns raw samples.
        If False returns quantile dataframe.
    """

    forecast_weeks = 52 - max_epiweek

    predicted = np.zeros((n_passes, forecast_weeks))

    # ---------------------------------------------------
    # Iterate regions
    # ---------------------------------------------------
    for geo in df.regional_geocode.unique():

        # -----------------------------------------------
        # Prepare data
        # -----------------------------------------------
        data = prep.prepare_regional_data(
            df=df,
            geo=geo,
            columns_to_normalize=columns_to_normalize,
            enso=enso,
            boxcox=boxcox,
            media=media
        )

        # -----------------------------------------------
        # Train normalization
        # -----------------------------------------------
        X_train, y_train, X_future_train, norm_values, norm_enso = prep.get_data(
            df_w=data.loc[data.year < test_year],
            columns_to_normalize=columns_to_normalize,
            max_epiweek=max_epiweek,
            min_year=min_year,
            enso=(
                enso.loc[enso.index.year <= test_year]
                if enso is not None else None
            )
        )

        # -----------------------------------------------
        # Test data
        # -----------------------------------------------
        X_test, y_test, X_future = prep.get_single_data(
            df_w=data,
            year=test_year,
            norm_values=norm_values,
            max_epiweek=max_epiweek,
            columns_to_normalize=columns_to_normalize,
            enso=enso,
            norm_enso=norm_enso
        )

        # -----------------------------------------------
        # Predict
        # -----------------------------------------------
        pred = evaluate_samples(
            model,
            X_test,
            X_future,
            n_passes=n_passes
        )

        if pred.ndim == 3 and pred.shape[1] == 1:
    
            pred = pred.squeeze(1)

        # -----------------------------------------------
        # Undo normalization
        # -----------------------------------------------
        pred = pred * norm_values['casos']

        if boxcox:
            pred = inv_boxcox(pred, THR) - 1

        predicted += pred

    # ---------------------------------------------------
    # Return raw samples
    # ---------------------------------------------------
    if return_samples:
        return predicted

    # ---------------------------------------------------
    # Convert to quantiles
    # ---------------------------------------------------
    dates = prep.gen_forecast_dates(
        test_year,
        max_epiweek=max_epiweek
    )

    return samples_to_quantiles(predicted, dates)

def get_total_cases(preds, region, model_name, region_col = 'regional_geocode'): 

    sum_preds = preds.sum(axis =1)

    df_preds = pd.DataFrame()

    df_preds[region_col] = [region]

    df_preds['pred'] = [np.percentile(sum_preds, q=50, axis=0)]

    df_preds['lower_50'] = [np.percentile(sum_preds, q=25, axis=0)]
    df_preds['upper_50'] = [np.percentile(sum_preds, q=75, axis=0)]

    df_preds['lower_80'] = [np.percentile(sum_preds, q=10, axis=0)]
    df_preds['upper_80'] = [np.percentile(sum_preds, q=90, axis=0)]

    df_preds['lower_90'] = [np.percentile(sum_preds, q=5, axis=0)]
    df_preds['upper_90'] = [np.percentile(sum_preds, q=95, axis=0)]

    df_preds['lower_95'] = [np.percentile(sum_preds, q=2.5, axis=0)]
    df_preds['upper_95'] = [np.percentile(sum_preds, q=97.5, axis=0)]

    if model_name is not None: 

        df_preds['model'] = [model_name]

    return df_preds 


def regional_predictions(
    model,
    df,
    enso,
    test_year,
    columns_to_normalize,
    max_epiweek=25,
    boxcox=False,
    n_passes=500,
    min_year=None,
    media=False,
):
    """
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]

    df_weekly:
        Weekly forecasts by regional_geocode.

    df_total:
        Accumulated forecast cases by regional_geocode.
    """

    dates = prep.gen_forecast_dates(
        test_year,
        max_epiweek=max_epiweek
    )

    weekly_forecasts = []
    total_forecasts = []

    for geo in df.regional_geocode.unique():

        # preparação dos dados ...
        data = prep.prepare_regional_data(
            df=df,
            geo=geo,
            columns_to_normalize=columns_to_normalize,
            enso=enso,
            boxcox=boxcox,
            media=media,
        )

        X_train, y_train, X_future_train, norm_values, norm_enso = prep.get_data(
            df_w=data.loc[data.year < test_year],
            columns_to_normalize=columns_to_normalize,
            max_epiweek=max_epiweek,
            min_year=min_year,
            enso=(
                enso.loc[enso.index.year <= test_year]
                if enso is not None else None
            ),
        )

        X_test, y_test, X_future = prep.get_single_data(
            df_w=data,
            year=test_year,
            norm_values=norm_values,
            max_epiweek=max_epiweek,
            columns_to_normalize=columns_to_normalize,
            enso=enso,
            norm_enso=norm_enso,
        )

        pred = evaluate_samples(
            model,
            X_test,
            X_future,
            n_passes=n_passes,
        )

        if pred.ndim == 3 and pred.shape[1] == 1:
            pred = pred.squeeze(1)

        pred = pred * norm_values["casos"]

        if boxcox:
            pred = inv_boxcox(pred, THR) - 1

        # previsões semanais
        df_weekly = samples_to_quantiles(pred, dates)
        df_weekly["regional_geocode"] = geo
        weekly_forecasts.append(df_weekly)

        # casos acumulados
        total_forecasts.append(
            get_total_cases(pred, geo, None, 'regional_geocode')
        )

    return (
        pd.concat(weekly_forecasts, ignore_index=True),
        pd.concat(total_forecasts, ignore_index=True),
    )


def build_model(
    region,
    TEST_YEAR,
    doenca,
    model_name,
    predict_n,
    max_epiweek,
    base_model,
    enso_model
):
    
    if "enso" in model_name:

        return LSTMWithFutureCovariatesV2(
            hidden=64,
            past_features=5,
            future_cov_size=predict_n,
            predict_n=predict_n,
            dropout=0.2,
        )

    elif "base" in model_name:

        return LSTMLogNormalModel(
            hidden=64,
            features=4,
            predict_n=52 - max_epiweek,
        )

    elif "mix" in model_name:

        base_model = load_model(
            region,
            TEST_YEAR,
            doenca,
            base_model,
            predict_n,
            max_epiweek,
            device="cpu",
        )

        future_model = load_model(
            region,
            TEST_YEAR,
            doenca,
            enso_model,
            predict_n,
            max_epiweek,
            device="cpu",
        )

        for p in base_model.parameters():
            p.requires_grad = False

        for p in future_model.parameters():
            p.requires_grad = False

        return MixtureOfExperts(
            model1=base_model,
            model2=future_model,
            hidden_size=64,
            future_cov_size=predict_n,
            predict_n=predict_n,
            gate_hidden=32,
        )
    
def load_model(
    region,
    TEST_YEAR,
    doenca,
    model_name,
    predict_n,
    max_epiweek,
    device,
    base_model = '',
    enso_model = ''
):

    model = build_model(
        region,
        TEST_YEAR,
        doenca,
        model_name,
        predict_n,
        max_epiweek,
        base_model = base_model,
        enso_model = enso_model
    )

    model_path = (
        f"./saved_models/"
        f"trained_{doenca}_{region}_{TEST_YEAR-1}_{model_name}.pt"
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True,
        )
    )

    model.to(device)

    return model


class MixtureOfExperts(nn.Module):

    def __init__(
        self,
        model1,
        model2,
        hidden_size,
        future_cov_size,
        predict_n,
        gate_hidden=32,
    ):
        super().__init__()

        self.model1 = model1
        self.model2 = model2

        ###################################################
        # GATE NETWORK
        ###################################################

        gate_input_size = (
            hidden_size +
            future_cov_size
        )

        self.gate = nn.Sequential(

            nn.Linear(
                gate_input_size,
                gate_hidden
            ),

            nn.ReLU(),

            nn.Linear(
                gate_hidden,
                predict_n
            ),

            nn.Sigmoid()
        )




    def forward(
        self,
        x_past,
        x_future
    ):

        x_past_base = torch.cat(
                        [x_past[:, :, :3], x_past[:, :, 4:]],
                        dim=2
                    )
        x_past_future = x_past
        ###################################################
        # MODEL 1
        ###################################################

        mu1, sigma1 = self.model1(x_past_base)

        ###################################################
        # MODEL 2
        ###################################################

        mu2, sigma2 = self.model2(
            x_past_future,
            x_future
        )

        ###################################################
        # SHARED ENCODER STATE
        ###################################################

        h = self.model1.encoder(
            x_past_base
        )

        ###################################################
        # GATE
        ###################################################

        gate_input = torch.cat(
            [h, x_future],
            dim=1
        )

        w = self.gate(
            gate_input
        )

        ###################################################
        # MIXTURE
        ###################################################

        #mu = (
        #    w * mu1 +
        #    (1 - w) * mu2
        #)

        #sigma = (
        #    w * sigma1 +
        #    (1 - w) * sigma2
        #)

        ###################################################
        # MIXTURE (LOG-NORMAL MOMENT MATCHING)
        ###################################################

        # 1. Calcular o primeiro momento (Média no espaço real) para cada especialista
        m1 = torch.exp(mu1 + 0.5 * (sigma1 ** 2))
        m2 = torch.exp(mu2 + 0.5 * (sigma2 ** 2))

        # 2. Calcular o segundo momento (E[X^2]) para cada especialista
        m1_sq = torch.exp(2 * mu1 + 2 * (sigma1 ** 2))
        m2_sq = torch.exp(2 * mu2 + 2 * (sigma2 ** 2))

        # 3. Misturar os momentos usando as probabilidades do Gate (w)
        E_Y = w * m1 + (1 - w) * m2
        E_Y2 = w * m1_sq + (1 - w) * m2_sq

        # 4. Converter os momentos misturados de volta para mu_mix e sigma_mix
        # A razão E[Y^2] / (E[Y])^2 deve ser sempre > 1. Usamos clamp para estabilidade numérica.
        ratio = torch.clamp(E_Y2 / (E_Y ** 2), min=1.0 + 1e-6)
        
        var_mix = torch.log(ratio)
        
        # Parâmetros finais para a Log-Normal da Mistura
        sigma = torch.sqrt(var_mix)
        mu = torch.log(E_Y) - 0.5 * var_mix

        return mu, sigma