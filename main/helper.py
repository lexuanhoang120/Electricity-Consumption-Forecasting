import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import STL
from prophet import Prophet
import pandas as pd


# from pmdarima import auto_arima


def measure_simility(arr1, arr2):
    """
    description: measure the simility of 2 series
    :param seri 2:
    :param seri 2:
    :return score:
    """
    arr1 = arr1.reshape((-1,))
    arr2 = arr2.reshape((-1,))
    score = np.corrcoef(arr1, arr2)[0, 1]
    return score


def pick_data_train(vt, arrs, n):
    """
    description: pick a seri from many series which has maxinum nth score (simility)
    :param a vector:
    :param a array has many vector:
    :param get n vector from array having maximum simility score:
    :return a value of vectors having maxinum nth score and its score
    """
    score = []
    for arr in arrs:
        score.append(measure_simility(vt, arr))
    score = np.array(score)
    return np.argsort(score)[-n:], score[np.argsort(score)[-n:]]


def create_data_total(df, month, year, lag, shift):
    """
    description: create essential data for forecasting the total of SLUONG
    :param df:
    :param month:
    :param year:
    :param lag:
    :param shift:
    :return: x_knowns , y_known, x_news
    """
    x_knowns = df[df.index < year].iloc[:, 11 + month - lag - shift: 11 + month - shift].values.reshape((-1, lag))
    # y_known = df[df.index == year].iloc[0,11 + month-lag: 11 +month].values
    y_known = df[df.index == year].iloc[0, 11 + month - lag: 11 + month].values
    # y_known = df.iloc[-1, 11 + month - lag: 11 + month].values
    x_news = df[df.index < year].iloc[:, 11 + month - shift].values
    # print(month,year,y_known)
    return x_knowns, y_known, x_news


def create_data_cumsum(df, month, year, lag, shift):
    """
    description: create essential data for forecasting the cumsum of SLUONG
    :param df:
    :param month:
    :param year:
    :param lag:
    :param shift:
    :return: x_knowns, y_known, x_news
    """
    x_knowns = df[df.index < year].iloc[:, 11 + month - lag - shift: 11 + month - shift].cumsum(axis=1).values.reshape(
        (-1, lag))
    y_known = df[df.index == year].iloc[:, 11 + month - lag: 11 + month].cumsum(axis=1).iloc[0, :].values
    # y_known = df.iloc[[-1], 11 + month - lag: 11 + month].cumsum(axis=1).iloc[0, :].values
    x_news = df[df.index < year].iloc[:, 11 + month - lag - shift: 12 + month - shift].cumsum(axis=1).iloc[:, -1].values
    return x_knowns, y_known, x_news


def create_data_rolling(df, month, year, lag, shift, n_rolling):
    """
    ignored
    :param df:
    :param month:
    :param year:
    :param lag:
    :param shift:
    :param n_rolling:
    :return:
    """
    y_old = df[df.index == year].iloc[0, 11 + month - 1 - n_rolling + 1:11 + month - 1].sum()
    df = df.rolling(n_rolling, axis=1).mean()
    x_knowns = df[df.index < year].iloc[:, 11 + month - lag - shift: 11 + month - shift].values.reshape((-1, lag))
    y_known = df[df.index == year].iloc[0, 11 + month - lag: 11 + month].values
    x_news = df[df.index < year].iloc[:, 11 + month - shift].values
    return x_knowns, y_known, x_news, y_old


def predict_total(known_y, known_x, new_x, order):
    """
    description: fit a line having order to points of data. To predict the electrics total of the next month
    :param known_y:
    :param known_x:
    :param new_x:
    :param order:
    :return a forecasting value
    """
    # known_y = known_y.reshape((-1,))
    # known_x = known_x.reshape((-1,))
    coefficients = np.polyfit(known_x, known_y, order)
    estimated_func = np.poly1d(coefficients)
    return float(estimated_func(new_x))


def predict_cumsum(known_y, known_x, new_x, order):
    """
    description: fit a line having order to points of data. To predict the electrics cumsum total of the next month
    :param known_y:
    :param known_x:
    :param new_x:
    :param order:
    :return a forecasting value
    """
    # known_y = known_y.reshape((-1,))
    # known_x = known_x.reshape((-1,))
    coefficients = np.polyfit(known_x, known_y, order)
    estimated_func = np.poly1d(coefficients)
    return float(estimated_func(new_x))


def predict(df_total, year, month: int, lag: int, order1: int, order2: int, shift: int, n_max):
    """
    description: forecasting the value in the future follow the way of statistics (using to forecasting the next point of season or others)
    :param df_total: dataframe
    :param year: month is forecasted
    :param month: month is forecasted
    :param lag: 6 (the number of month is back to history to compute the simility score, and to predict)
    :param order1: 1 (ignored)
    :param order2: 1 (ignored)
    :param shift: 0 (ignored)
    :param n_max: the number of year in the history to estimate the recent year
    :return: forecasting values and simility score
    """
    x_knowns, y_known, x_news = create_data_total(df_total, month, year, lag, shift)
    max_scores, coeffs = pick_data_train(y_known, x_knowns, n_max)
    pre = []
    for max_score in max_scores:
        x_knowns, y_known, x_news = create_data_total(df_total, month, year, lag, shift)
        x_known = x_knowns[max_score]
        x_new = x_news[max_score]

        pred1 = predict_total(known_y=y_known,
                              known_x=x_known,
                              new_x=x_new, order=order1)
        pre.append(pred1)
        shift = 0
        x_knowns, y_known, x_news = create_data_cumsum(df_total, month, year, lag, shift)
        x_known = x_knowns[max_score]
        x_new = x_news[max_score]
        pred2 = predict_cumsum(known_y=y_known,
                               known_x=x_known,
                               new_x=x_new, order=order2) - y_known[-1]

        pre.append(pred2)
    return np.mean(pre), coeffs.mean()


def predict_master(data: pd.DataFrame, d_date: pd.DataFrame, month, year, lag, p, d, q, n_max, type_decomp,
                   methods_season, method_resid, method_trend, order1, order2, name_province):
    """
    description: main forecasting function
    :param data: dataframe
    :param d_date: dataframe datetime to concat with series
    :param month: forecasted month
    :param year: forecasted year
    :param lag: 6 ((ignored))
    :param p: 1 (ignored)
    :param d: 0 (ignored)
    :param q: 1 (ignored)
    :param n_max: the number of year in the history to estimate the future
    :param type_decomp: type to decompose the time series into trend, season, residual
    :param methods_season: method to forecast the next point of seasonal series
    :param method_resid: method to forecast the next point of residual series
    :param method_trend: method to forecast the next point of trending series
    :param order1: 1 (ignored)
    :param order2: 1 (ignored)
    :param name_province: name of province
    :return: the next values of trend, season and residual
    """
    date = f"{year}-{month}-1"
    # print(date)
    # d_train = data[data.index < f"{year}-1-1"]
    d_train = data[data.index < date]
    d_train = pd.concat([d_train, d_date], axis=1).dropna(axis=0)

    d_residuals, d_trend, d_seasonal = detrend(d_train, type=type_decomp)
    # y_residual=0
    y_residual = forecast_residual(d_residuals.values, p, d, q, method_resid=method_resid)
    # except:
    #     y_residual = 0

    d_seasonal.name = name_province
    # d_seasonal = pd.concat([d_seasonal, d_date], axis=1)
    d_seasonal = pd.concat([d_seasonal, d_date], axis=1).fillna(0)
    d_seasonal = d_seasonal.pivot_table(columns='month', index='year', values=name_province)
    y_seasonal = forecast_seasonal(d_seasonal, month, year, lag, n_max, order1, order2, methods_season)

    d_trend.name = name_province
    d_trend = pd.concat([d_trend, d_date], axis=1).dropna()
    y_trend = forecast_trend(d_trend.iloc[-2, 0], d_trend.iloc[-1, 0], method_trend=method_trend)

    # d_residuals = pd.concat([d_residuals, d_date], axis=1).dropna()
    # y_re3 = d_seasonal.pivot_table(columns='month', index='year', values=name_province).dropna(axis=0).mean(axis=0)[
    #     month]
    # d_residuals = d_residuals.pivot_table(columns='month', index='year', values=name_province)
    # d_trend = d_trend.pivot_table(columns='month', index='year', values=name_province)
    # y_re2 = forecast_seasonal(d_trend, month, year, lag, 2)
    # return y_lr + (y_re1 +y_re2+y_re)/3,y_re
    # return (y_re3 + y_re1) / 2 + y_re2
    return y_trend, y_seasonal, y_residual


def detrend(df: pd.DataFrame, type='STL'):
    """
    description: decompose a series, using prophet and STL, assumption: the data is seasonal with period = 12
    :param df: series
    :param type: type of decomposition
    :return: trend, season and residual
    """
    if type == 'STL':
        data = df.copy()
        data = data.iloc[:, 0]
        stl = STL(data, period=12)
        res = stl.fit()
        trend = res.trend
        resid = res.resid
        season = res.seasonal
        return resid, trend, season
    if type == 'prophet':
        data = df.copy()
        data = data.iloc[:, [0]]
        data = data.reset_index()
        data.columns = ['ds', 'y']
        model = Prophet(yearly_seasonality=12)
        model.fit(data)
        forecaster = model.predict(model.make_future_dataframe(periods=0, freq='MS'))
        forecaster['y'] = data['y']
        forecaster['resid'] = forecaster.apply(lambda row: (row.y - (row.yearly + row.trend)), axis=1)
        forecaster = forecaster.set_index('ds')
        df_resid = forecaster['resid']
        df_trend = forecaster['trend']
        df_season = forecaster['yearly']
        df_season.name = 'seasonal'
        return df_resid, df_trend, df_season
    else:
        return None, None, None


def compute_error(data, month, year, y_pr):
    """
    description: compute the error of data on month in year with its y_predict
    :param data:
    :param month:
    :param year:
    :param y_pr:
    :return: error value
    """
    y_tr = data[data.index == f'{year}-{month}-1'].values[0]
    return (y_pr - y_tr) / y_tr, y_tr


def forecast_seasonal(data, month, year, lag, n_max, order1, order2, method='statistics'):
    """
    description: forecasting the next point of season
    :param data:
    :param month:
    :param year:
    :param lag:
    :param n_max:
    :param order1:
    :param order2:
    :param method:
    :return: value of the next point of input data
    """
    if method == 'statistics':
        """
        method: statistics, using the history data to forecast the future (month in the recent year), choosing amount 
        of year in the history to estimate the future
        """
        df = data.copy()
        df = pd.concat([df.shift(1, axis=0), df, df.shift(-1, axis=0)], axis=1).iloc[1:, :] + 10e10
        # lag = int(params[month - 1][1])
        # shift = int(params[month - 1][2])
        # order1 = int(params[month - 1][3])
        # order2 = int(params[month - 1][4])
        # lag = 6
        shift = 0
        # order1 = 1
        # order2 = 1
        y, _ = predict(df, year, month, lag, order1, order2, shift, n_max)
        return y - 10e10
    if method == 'average':
        """
        method: average, using all the history data to forecast the future, take the average of each month and forecast
         the next month by average this month of all year in the history 
         """
        df = data.copy()
        # df = df[df.index<year]
        # print(df.dropna(axis=0).mean(axis=0))
        y = df.dropna(axis=0).mean(axis=0)[month]
        # print(y)
        return y
    if method == 'latest':
        """
        method: latest, forecast the next month by the value of this month in the previous year 
        """
        df = data.copy()
        # df = df[df.index<year]
        # print(df.dropna(axis=0).mean(axis=0))
        y = df.dropna(axis=0).iloc[-1, :][month]
        # print(y)
        return y
    else:
        raise "Not specify the season method"


def forecast_trend(y_1, y_t, method_trend='average'):
    """
    forecast the next point of trend data, included average and latest method, using 2 latest point of trend
    :param y_1: y_t-1
    :param y_t: y_t
    :param method_trend:
    :return: value of the next point
    """
    if method_trend == 'average':
        """
        take the average of 2 point to forecast the next point
        """
        y_pr_trend = y_t + (y_t - y_1)
        # y_pr_trend = y_t
        return y_pr_trend
    if method_trend == 'latest':
        """
        take the latest point to forecast the next point
        """
        return y_t
    else:
        raise "Not specify the trend method"


def forecast_residual(series, p, d, q, method_resid='arima'):
    """
    forecast the next point of residual data. Residual data is quite the same of white noise, so
    forecast it difficult and cautious. Here use arima to forecast. With residual data is highly variant,
    the forecasted next point is 0
    :param series:
    :param p: p,d,q is parameter of arima model and value is estimated from experience
    :param d:
    :param q:
    :param method_resid:
    :return: the next point
    """
    if method_resid == 'arima':
        try:
            model = ARIMA(series, order=(p, d, q))
            model_fit = model.fit()
            return model_fit.forecast(freq='MS')[0]
        except:
            return 0
    # if method_resid == 'autoarima':
    #     model = auto_arima(series, seasonal=False, trace=False)
    #     return model.predict(n_periods=1, freq='MS')[0]
    else:
        return 0


def get_ytr(df, month, year):
    """
    description: get the y truth from data frame on month in year
    :param df:
    :param month:
    :param year:
    :return: value or None if month in year is not existed yet
    """
    try:
        filter = f'{year}-{month}-01'
        return df[df.index == filter].values[0][0]
    except:
        return None
