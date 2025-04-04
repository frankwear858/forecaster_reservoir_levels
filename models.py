##
import os.path

import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from settings import usern, passw, db_name, host_machine, port
import warnings


# pandas options to allow pd dataframes to be easier to read
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.max_info_columns', 200)
pd.set_option('display.width', 2000)  # Set to the desired display width

# config parameters
schema = 'reservoirs'
res_data_dir = 'res_data'

# creating sqlalchemy engine
engine = create_engine(f'postgresql://{usern}:{passw}@{host_machine}:{port}/{db_name}')


class ReservoirCNN(nn.Module):
    def __init__(self, input_dim):
        super(ReservoirCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
        self.fc1 = nn.Linear(32 * (input_dim - 4), 64)  # input_dim - 4 because of kernel size reduction
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension: (batch_size, 1, input_dim)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


input_features = ['cs', 'precip', 'avg_temp']
def create_lag_features(df):
    """
        Adds lag and rolling window features to the input DataFrame for specified input features.

        For each feature in `input_features`, the function generates lagged versions (up to 13 lags)
        and rolling statistics (mean and standard deviation) for various window sizes. Additionally,
        it creates a target feature by shifting the 'cs' column by one time step.

        Args:
            df (pd.DataFrame): Input DataFrame containing time series data.

        Returns:
            pd.DataFrame: DataFrame with added lag, rolling window, and target features.
        """
    # Create lag features
    for feature in input_features:
        for i in range(1, 14):
            df[f'{feature}_lag_{i}'] = df[feature].shift(i)

        # Add rolling window features
        for window in [2, 3, 5, 7, 10, 14]:  # Example windows: 7 days, 14 days, 30 days
            df[f'{feature}_rolling_mean_{window}'] = df[feature].rolling(window=window).mean()
            df[f'{feature}_rolling_std_{window}'] = df[feature].rolling(window=window).std()

    # target feature
    df['cs_target'] = df['cs'].shift(-1)

    return df


def load_reservoir_data():
    '''loads reservoir_daily_data'''
    reservoir_data = pd.read_sql(f'SELECT * FROM reservoir_daily_data', engine)
    # print(reservoir_data)
    return reservoir_data

##
def load_metadata():
    '''loads tx_res_meta data'''
    meta_data = pd.read_sql(f'SELECT * FROM tx_res_meta', engine)
    # print(meta_data)
    return meta_data
##

target_feature = 'cs_target'
def load_and_prepare_reservoir_training_data():
    # load reservoir daily data
    reservoir_data = load_reservoir_data()

    # # # #
    # # joining metadata made models less accurate. here for reference.
    # metadata = pd.read_sql(f'SELECT * FROM {schema}.tx_res_meta', engine)
    # print(metadata)
    #
    # metadata['stn_id'] = metadata['stn_id'].astype(str).apply(lambda x: '0' + x)
    # metadata.rename(columns={"stn_id": "cs_id"}, inplace=True)
    # print(metadata)
    #
    # # #
    # meta_cols_to_join = ['cs_id', 'lat', 'lon', 'fp', 'dp', 'pf']
    # reservoir_data = pd.merge(reservoir_data, metadata[meta_cols_to_join], on='cs_id', how='left')
    # print(reservoir_data)

    # creating lag
    reservoir_data = create_lag_features(reservoir_data)
    reservoir_data.dropna(inplace=True)

    reservoir_data['date'] = pd.to_datetime(reservoir_data['date'])
    reservoir_data['month'] = reservoir_data['date'].dt.month
    # reservoir_data['day_of_year'] = reservoir_data['date'].dt.dayofyear
    reservoir_data['day_of_month'] = reservoir_data['date'].dt.day
    # reservoir_data['day_of_week'] = reservoir_data['date'].dt.dayofweek

    # getting unique reservoir names
    reservoirs = reservoir_data['reservoir_name'].unique()

    # declaring input features
    exclude_columns = ['date', 'reservoir_name', 'cs_id'] + [target_feature]
    # Get the list of feature column names
    features = reservoir_data.drop(columns=exclude_columns).columns.tolist()
    return reservoir_data, features, reservoirs


def train_scikit_linear_regression():
    print('\ntraining reservoir sklearn linear regression model...')
    ##
    # load and prepare reservoir training data. also gets features and list of reservoirs.
    reservoir_data, features, reservoirs = load_and_prepare_reservoir_training_data()

    # splitting data into training and test sets
    train_test_split_date = '2020-01-01'
    train_data = reservoir_data[reservoir_data['date'] < train_test_split_date]
    print(train_data)
    test_data = reservoir_data[reservoir_data['date'] >= train_test_split_date]
    print(test_data)


    ##
    # training scikit-learn linear regression model for each reservoir
    for reservoir in reservoirs:
        print(f'\ntraining scikit-learn linear regression model for reservoir: {reservoir}')
        train_res = train_data[train_data['reservoir_name'] == reservoir]
        # print(train_res)
        test_res = test_data[test_data['reservoir_name'] == reservoir]
        # print(test_res)

        X_train, y_train = train_res[features], train_res[target_feature]
        X_test, y_test = test_res[features], test_res[target_feature]

        lin_regression_model = LinearRegression()
        lin_regression_model.fit(X_train, y_train)

        y_pred = lin_regression_model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f'{reservoir} - MAE: {mae}, RMSE: {rmse}')

        # Save model
        print(f'saving linear regression model for {reservoir}...')
        model_path = os.path.join('models', f'{reservoir}_linear_regression.pkl')
        joblib.dump(lin_regression_model, model_path)
        print(f'saved {model_path}!')


##
# training pytorch CNN model
def train_reservoir_CNN():
    print('\ntraining reservoir CNN model...')
    reservoir_data, features, reservoirs = load_and_prepare_reservoir_training_data()

    # splitting data into training and test sets
    train_test_split_date = '2020-01-01'
    train_data = reservoir_data[reservoir_data['date'] < train_test_split_date]
    print(train_data)
    test_data = reservoir_data[reservoir_data['date'] >= train_test_split_date]
    print(test_data)


    # configuring GPU. checking GPU status
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print("PyTorch version:", torch.__version__)
    print("Is CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("Number of GPUs available:", torch.cuda.device_count())
    print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")


    n_epochs_options = [5, 10, 25, 50, 75, 100, 150, 200, 500]  # Different numbers of epochs to try

    # class ReservoirCNN(nn.Module):
    #     def __init__(self, input_dim):
    #         super(ReservoirCNN, self).__init__()
    #         self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
    #         self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
    #         self.fc1 = nn.Linear(32 * (input_dim - 4), 64)  # input_dim - 4 because of kernel size reduction
    #         self.fc2 = nn.Linear(64, 1)
    #
    #     def forward(self, x):
    #         x = x.unsqueeze(1)  # Add a channel dimension: (batch_size, 1, input_dim)
    #         x = torch.relu(self.conv1(x))
    #         x = torch.relu(self.conv2(x))
    #         x = x.view(x.size(0), -1)  # Flatten for fully connected layer
    #         x = torch.relu(self.fc1(x))
    #         x = self.fc2(x)
    #         return x

    # Convert data to PyTorch tensors
    def to_tensor(data, features, target):
        X = torch.tensor(data[features].values, dtype=torch.float32)
        y = torch.tensor(data[target].values, dtype=torch.float32).unsqueeze(1)
        return X, y


    # Check if GPU is available and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rates = [
        1.0, 0.5, 0.25, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001,
        0.05, 0.005, 0.0005, 0.00005, 0.000005, 0.002, 0.02, 0.2, 0.0025
    ]

    # Dictionary to store best results for each reservoir (good for debug and model research)
    best_results = {}
    # training a pytorch CNN model for each reservoir (will automatically find the best learning rate and # of epochs)
    for reservoir in reservoirs:
        best_lr = None
        best_epochs = None
        best_mae = float('inf')
        best_rmse = float('inf')
        best_model = None

        print(f'\nTraining PyTorch CNN model for reservoir: {reservoir}')
        train_res = train_data[train_data['reservoir_name'] == reservoir]
        test_res = test_data[test_data['reservoir_name'] == reservoir]

        X_train, y_train = to_tensor(train_res, features, target_feature)
        X_test, y_test = to_tensor(test_res, features, target_feature)

        # Move tensors to GPU
        X_train, y_train = X_train.to(device), y_train.to(device)
        X_test, y_test = X_test.to(device), y_test.to(device)

        # Loop through learning rates
        for lr in learning_rates:
            for n_epochs in n_epochs_options:
                # print(f'\nTesting learning rate: {lr}')
                # Initialize model and move it to GPU
                CNN_model = ReservoirCNN(input_dim=len(features)).to(device)
                criterion = nn.MSELoss().to(device)
                # learning_rate = learning_rates.get(reservoir, 0.0001)  # Default to 0.001 if not specified
                optimizer = optim.Adam(CNN_model.parameters(), lr=lr)

                # Training loop
                CNN_model.train()
                for epoch in range(n_epochs):
                    optimizer.zero_grad()
                    outputs = CNN_model(X_train)
                    loss = criterion(outputs, y_train)
                    loss.backward()
                    optimizer.step()

                # Evaluation
                CNN_model.eval()
                with torch.no_grad():
                    y_pred = CNN_model(X_test).cpu().numpy()
                    y_test_np = y_test.cpu().numpy()

                mae = mean_absolute_error(y_test_np, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test_np, y_pred))

                # print(f'{reservoir} - MAE: {mae}, RMSE: {rmse}')
                # print(f'{reservoir} Learning Rate: {lr} - MAE: {mae:.4f}, RMSE: {rmse:.4f}')

                # Check if this is the best learning rate so far
                if mae < best_mae:
                    best_mae = mae
                    best_rmse = rmse
                    best_lr = lr
                    best_epochs = n_epochs
                    best_model = CNN_model.state_dict()

        print(f'\nBest learning rate and epochs for {reservoir}: LR: {best_lr}, Epochs: {best_epochs} with MAE: {best_mae:.4f} and RMSE: {best_rmse:.4f}')
        # print(f'\nBest learning rate so far: {best_lr} with MAE: {best_mae:.4f} and RMSE: {best_rmse:.4f}')
        # print(f'\nBest learning rate for {reservoir}: {best_lr} with MAE: {best_mae:.4f} and RMSE: {best_rmse:.4f}')

        # Save the best results for the current reservoir
        best_results[reservoir] = {
            'best_lr': best_lr,
            'best_mae': best_mae,
            'best_rmse': best_rmse,
            'best_model': best_model
        }
        print(f'saving CNN model for {reservoir}...')
        model_path = os.path.join('models', f'{reservoir}_CNN.pth')
        torch.save(best_model, model_path)
        print(f'saved {model_path}!')


##


##
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")
def iterative_forecast_with_lag(model, initial_data, features, n_days=7, model_type='pytorch'):
    """
    Iteratively forecasts future values using a trained model, updating input data with each prediction.

    Parameters:
    -----------
    model : object
        Trained model (PyTorch or scikit-learn) for predictions.

    initial_data : pandas.DataFrame
        DataFrame containing the latest data points for prediction. It will be updated iteratively.

    features : list of str
        List of feature column names used for prediction.

    n_days : int, optional (default=7)
        Number of future days to forecast.

    model_type : str, optional (default='pytorch')
        Type of model used ('pytorch' or 'sklearn').

    Returns:
    --------
    predictions : list of float
        Predicted values for the next `n_days` days.
    """
    # Make a deep copy of the initial data to avoid modifying the original dataset
    predictions = []
    forecast_data = initial_data.copy()

    for _ in range(n_days):
        # print(f'\npredicting day {_ + 1}...')
        # Generate lag and rolling features
        forecast_data = create_lag_features(forecast_data)

        # Ensure we use the latest available data for prediction
        X_last = forecast_data[features].iloc[-1:].copy()  # Get the last row with all features
        # print('X_last:', X_last)

        if model_type == 'pytorch':
            X_last = X_last.apply(pd.to_numeric, errors='coerce').fillna(0.0)
            X_last_tensor = torch.tensor(X_last.values, dtype=torch.float32).to(next(model.parameters()).device)
            prediction = model(X_last_tensor).item()
        elif model_type == 'sklearn':
            prediction = model.predict(X_last)[0]

        else:
            raise ValueError("Invalid model_type. Choose 'pytorch' or 'sklearn'.")

        # print('prediction:', prediction)

        # Append the prediction to the results
        predictions.append(prediction)

        # Update the forecast_data with the new prediction to create the next lag feature
        new_row = forecast_data.iloc[-1].copy()
        # Update with the prediction
        new_row['cs'] = prediction

        # Calculate the next day
        last_date = pd.Timestamp(new_row['date'])  # Ensure it's a Timestamp
        next_day = last_date + pd.Timedelta(days=1)
        # update with next date
        new_row['date'] = next_day

        # print('new_row:')
        # print(new_row)

        # transpose row for pd.concat
        new_row_T = new_row.to_frame().T
        # forecast_data = forecast_data.append(new_row, ignore_index=True)
        if len(predictions) < n_days:
            new_row_T = new_row_T.apply(pd.to_numeric, errors='coerce').fillna(0.0)
            forecast_data = pd.concat([forecast_data, new_row_T], ignore_index=True)
            # print('updated forecast data:')
            # print(forecast_data)

    return predictions


def load_model(reservoir_name, model_type, input_dim=84):
    """Load the appropriate model based on the reservoir name and model type."""
    if model_type == 'sklearn':
        model_path = os.path.join('models', f'{reservoir_name}_linear_regression.pkl')
        model = joblib.load(model_path)
    elif model_type == 'pytorch':
        model_path = os.path.join('models', f'{reservoir_name}_CNN.pth')
        model = ReservoirCNN(input_dim=input_dim)  # Adjust input_dim based on your features
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        raise ValueError("Unsupported model type. Choose 'sklearn' or 'pytorch'.")
    return model


def make_predictions(reservoir_name, model_type, date):
    """
        Generate a 7-day forecast for reservoir storage using a specified model.

        Parameters:
        -----------
        reservoir_name : str
            The name of the reservoir for which to make predictions.

        model_type : str
            The type of model to use ('pytorch' or 'sklearn').

        date : str or pd.Timestamp
            The reference date for starting the forecast.

        Returns:
        --------
        predictions : list of float
            Predicted reservoir storage values for the next 7 days.
        """
    reservoir_data, features, reservoirs = load_and_prepare_reservoir_training_data()
    # filtering for reservoir
    reservoir_data = reservoir_data[reservoir_data['reservoir_name'] == reservoir_name]

    # getting last 14-day window of reservoir_data
    target_date = pd.Timestamp(date)
    reservoir_input_14days = reservoir_data[(reservoir_data['date'] <= target_date) & (reservoir_data['date'] > target_date - pd.Timedelta(days=14))]

    model = load_model(reservoir_name, model_type, len(features))

    predictions = iterative_forecast_with_lag(model, reservoir_input_14days, features, n_days=7, model_type=model_type)
    return predictions

##
# end product. this works once models are trained and saved to models directory
# forecast = make_predictions('lakegeorgetown', model_type='pytorch', date='2021-04-26')
# print(forecast)

##

def main():
    train_scikit_linear_regression()
    train_reservoir_CNN()


if __name__=='__main__':
    main()
