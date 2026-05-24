from init import *
from helper import *
import argparse
from tqdm import trange
import datetime

# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-f", "--from_date", help="datetime: month to forecasting firstly", default=0)
parser.add_argument("-t", "--to_date", help="datetime: month to forecasting lastly", default=0)
parser.add_argument("-p", "--path_params", help="path: it include dataframe which occupy parameters to forecast",
                    default="/path/to/company_data/user/<user>/cpc/data/parameter/final_params_13province_u2022.parquet")
parser.add_argument("-i", "--input", help="path: it must include data of 13 province",
                    default="/path/to/company_data/user/<user>/cpc/data/sample_sum_data.parquet")
parser.add_argument("-o", "--output", help="path: to save the forecasting result",
                    default="/path/to/company_data/user/<user>/cpc/data/fct_12t_sum_13province_test_2024.parquet")

# Read arguments from command line
args = parser.parse_args()
try:
    if args.from_date:
        from_date = args.from_date
    if args.to_date:
        to_date = args.to_date
    if args.path_params:
        path_params = args.path_params
    if args.input:
        path_input = args.input
    if args.output:
        path_output = args.output
except Exception as e:
    print(e)

# read input data from parquet file
df_total = pd.read_parquet(path_input, filesystem=hdfs).sort_index(
    ascending=False)  # df_total included 15 columns (13 columns of provinces and month and year, all has 15 columns)

# Padding short-time in the future
df_total_padding = pd.DataFrame({col: None for col in df_total.columns},
                                index=pd.date_range(start=df_total.index.max() + datetime.timedelta(days=32),
                                                    end=df_total.index.max() + datetime.timedelta(days=400), freq='MS'))
df_total_final = pd.concat([df_total, df_total_padding], axis=0)

# create datetime dataframe
date_range = pd.date_range(start=df_total.index.min(), end=df_total.index.max() + datetime.timedelta(days=400),
                           freq='MS')
df_datetime = pd.DataFrame({'stt': [i for i in range(date_range.shape[0])]}, index=date_range)
df_datetime['month'] = df_datetime.index.month
df_datetime['year'] = df_datetime.index.year

# read parameters from file
dfp = pd.read_parquet(path_params, filesystem=hdfs)

# df_result: to saving the forecasing result, lst_pr: name of provinces
lst_pr = df_total_final.columns
df_result = pd.DataFrame()
# years = [2022,2023]

# transform input to month and year in order to forecast
try:
    date_range = pd.date_range(start=from_date, end=to_date, freq='MS')
except:
    print('day is out of for month')
    date_range = pd.date_range(start=f"{df_total.index.max().year}-{df_total.index.max().month + 1}",
                               end=f"{df_total.index.max().year}-{df_total.index.max().month + 1}", freq='MS')

for n_province in trange(13):
    # for n_province in [1]:
    # df = df_total.iloc[:, [n_province]]
    name_province = lst_pr[n_province]
    try:
        # get parameters of each province to forecasting
        dp = dfp[dfp.province == name_province]
        lag = 6
        n_max = dp.n_max.values[0]
        # type_decomp = dp.type_decomp.values[0]
        # method_season =dp.method_season.values[0]
        method_trend = dp.method_trend.values[0]
        method_resid = dp.method_resid.values[0]
        # n_max = 5
        type_decomp = 'STL'
        method_season = 'statistics'
        p = 1
        d = 0
        q = 1
    except:
        lag = 6
        n_max = 5
        method_trend = 'average'
        method_season = 'statistics'
        type_decomp = 'STL'
        method_resid = 'arima'
        p = 1
        d = 0
        q = 1
    # loop through months and years to forecasting
    for year_month in date_range:
        year, month = (year_month.year, year_month.month)
        # for year in years.strip('][').split(','):
        #     year=int(year)
        #     for month in months.strip('][').split(','):
        #         month=int(month)
        if pd.to_datetime(f"{year}-{month}") > pd.to_datetime(
                f"{df_total.index.max().year}-{df_total.index.max().month + 1}"):
            print('OUT OF RANGE TO FORECASTING')
            break
        lst_ypr = []
        lst_ytr = []
        lst_trend = []
        lst_resid = []
        lst_season = []
        df = df_total_final.copy().iloc[:, [n_province]]
        df = df.sort_index()
        for t in range(12):
            # try:
            # print(t)
            if (month + t > 12):
                year = year + 1
                month = month - 12
                # print(month)

                y_trend, y_seasonal, y_residual = predict_master(
                    data=df,
                    d_date=df_datetime,
                    month=month + t, year=year,
                    lag=lag, p=p, d=d, q=q,
                    n_max=n_max,
                    type_decomp=type_decomp,
                    methods_season=method_season, method_resid=method_resid,
                    method_trend=method_trend, order1=1,
                    order2=1, name_province=name_province)
                y_tr = get_ytr(df, month + t, year)
                y_pr = y_trend + y_seasonal + y_residual

                df.at[f"{year}-{month + t}-1", name_province] = y_pr
                year = year - 1
                month = month + 12
            else:
                y_trend, y_seasonal, y_residual = predict_master(
                    data=df,
                    d_date=df_datetime,
                    month=month + t, year=year,
                    lag=lag, p=p, d=d, q=q,
                    n_max=n_max,
                    type_decomp=type_decomp,
                    methods_season=method_season, method_resid=method_resid,
                    method_trend=method_trend, order1=1,
                    order2=1, name_province=name_province)

                y_tr = get_ytr(df, month + t, year)
                y_pr = y_trend + y_seasonal + y_residual

                df.at[f"{year}-{month + t}-1", name_province] = y_pr

            lst_ypr.append(y_pr)
            lst_ytr.append(y_tr)
            lst_trend.append(y_trend)
            lst_season.append(y_seasonal)
            lst_resid.append(y_residual)

        # except:
        #
        #     print(year,month,t)
        #     break
        df_result = pd.concat([df_result, pd.DataFrame({
            'province': [name_province],
            'year': [year],
            'month': [month],
            'trend': [lst_trend],
            'season': [lst_season],
            'resid': [lst_resid],
            'y_pr': [lst_ypr],
            'y_tr': [lst_ytr],
            'n_max': [n_max],
        })])
    # except:break


def expand_ypr(row):
    row = row['y_pr']
    return row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11]


def expand_ytr(row):
    row = row['y_tr']
    return row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11]


df_result[['y_pr1', 'y_pr2', 'y_pr3', 'y_pr4', 'y_pr5', 'y_pr6', 'y_pr7', 'y_pr8', 'y_pr9', 'y_pr10', 'y_pr11',
           'y_pr12', ]] = df_result.apply(expand_ypr, axis=1, result_type='expand')
df_result[['y_tr1', 'y_tr2', 'y_tr3', 'y_tr4', 'y_tr5', 'y_tr6', 'y_tr7', 'y_tr8', 'y_tr9', 'y_tr10', 'y_tr11',
           'y_tr12', ]] = df_result.apply(expand_ytr, axis=1, result_type='expand')

# df_result['date'] = df_result.apply(lambda x:pd.to_datetime(f"{x.year}-{x.month}-01"),axis=1)
df_result.to_parquet(path_output, filesystem=hdfs)
