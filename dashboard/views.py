# from django.shortcuts import render
# from .models import ReservoirDailyData, TxResMeta
# from django.db.models import Avg

from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from dashboard.models import ReservoirDailyData, ReservoirMetaData
from django.utils.text import slugify
from models import make_predictions, load_reservoir_data, load_metadata
import pandas as pd
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error


def home(request):
    # return render(request, 'dashboard/home.html')  # Render the home page template
    reservoirs = ReservoirMetaData.objects.all()
    for reservoir in reservoirs:
        print(reservoir.name, reservoir.lat, reservoir.lon)  # Add this line to check the data
    return render(request, 'dashboard/home.html', {'reservoirs': reservoirs})

#
def reservoir_detail(request, stn_id):

    # load reservoir metadata
    reservoir_meta = load_metadata()
    reservoir_meta = reservoir_meta[reservoir_meta['stn_id'] == int(stn_id)]

    # loading reservoir daily data
    formatted_stn_id = '0' + stn_id
    data = load_reservoir_data()
    data = data[data['cs_id'] == formatted_stn_id]


    # Get the latest date in the DataFrame
    # Calculate the date 3 years prior to the latest date
    latest_date = data['date'].max()
    three_years_ago = latest_date - pd.DateOffset(years=3)
    data = data[data['date'] >= three_years_ago]

    # Group by month and calculate average monthly storage
    data['month'] = data['date'].dt.to_period('M')
    monthly_avg = data.groupby('month').agg({'cs': 'mean'}).reset_index()
    #  Calculate as percent of total capacity
    monthly_avg['cs_percent'] = ((monthly_avg['cs'] / 1000) / reservoir_meta['fp'].iloc[0]) * 100
    print('monthly avg')
    print(monthly_avg)

    # Create labels for the chart
    monthly_avg['month'] = monthly_avg['month'].dt.strftime('%Y-%m')  # Convert Period to string format
    labels = monthly_avg['month'].tolist()
    cs_percent = monthly_avg['cs_percent'].tolist()

    reservoir_name = str(reservoir_meta['name'].iloc[0])
    # print('reservoir_name:', reservoir_name)
    # print('Labels:', labels)  # Debugging line
    # print('CS Percent:', cs_percent)  # Debugging line
    context = {
        'reservoir': reservoir_name,
        'labels': json.dumps(labels),
        'cs_percent': json.dumps(cs_percent),
    }
    return render(request, 'dashboard/reservoir_detail.html', context)


def reservoir_predictions(request):
    # Logic to prepare data for the template
    reservoirs = ReservoirMetaData.objects.all()  # Fetch all reservoir data
    return render(request, 'dashboard/reservoir_prediction.html', {'reservoirs': reservoirs})


def reservoir_data_api(request, reservoir_id, model_type='pytorch'):
    date = request.GET.get('date')
    # Fetch actual and predicted values based on reservoir_id and date
    res_data = load_reservoir_data()
    res_data = res_data[res_data['cs_id'] == '0' + str(reservoir_id)]
    reservoir_name = res_data['reservoir_name'].iloc[0]

    # get the next 7 days
    specific_date = pd.Timestamp(date)
    start_date = specific_date + pd.Timedelta(days=1)  # Start the range from the day after
    end_date = start_date + pd.Timedelta(days=6)  # End the range 6 days after the start date
    actual_values = res_data[(res_data['date'] >= start_date) & (res_data['date'] <= end_date)]['cs']

    dates = res_data[(res_data['date'] >= start_date) & (res_data['date'] <= end_date)]['date'].dt.strftime('%Y-%m-%d').tolist()

    # predicting next 7 cs values
    # print('date pre formatted:', date, type(date))
    # date =
    predicted_values_pytorch = make_predictions(reservoir_name, 'pytorch', date)
    predicted_values_sklearn = make_predictions(reservoir_name, 'sklearn', date)
    # calculating mae
    model_maes_pytorch = mean_absolute_error(actual_values, predicted_values_pytorch)
    model_maes_sklearn = mean_absolute_error(actual_values, predicted_values_sklearn)


    # print('model mae:', model_maes)
    # print('dates:', dates)
    # print('predicted_values', type(predicted_values))
    actual_values_list = actual_values.tolist()
    # print('actual values:', actual_values_list)
    json_resp = JsonResponse({
        'dates': dates,
        'actual': actual_values_list,
        'predicted_pytorch': predicted_values_pytorch,
        'predicted_sklearn': predicted_values_sklearn,
        'mae_pytorch': model_maes_pytorch,
        'mae_sklearn': model_maes_sklearn
    })

    # print('json_resp')
    # print(json_resp)
    return json_resp
