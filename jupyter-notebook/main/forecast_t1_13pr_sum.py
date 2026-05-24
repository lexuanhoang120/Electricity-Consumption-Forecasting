from init import *
from helper import *
import argparse
from tqdm import trange
import datetime

# Initialize parser
parser = argparse.ArgumentParser()
# adding argument from commandline
parser.add_argument("-f", "--from_date", help="datetime: month to forecasting firstly", default=0)
parser.add_argument("-t", "--to_date", help="datetime: month to forecasting lastly", default=0)
parser.add_argument("-p", "--path_params", help="path: it include dataframe which occupy parameters to forecast",
                    default="/path/to/company_data/user/<user>/cpc/data/parameter/final_params_13province_u2022.parquet")
parser.add_argument("-i", "--input", help="path: it must include data of 13 province",
                    default="/path/to/company_data/sa/cpc/store/data/SUM_TOTAL/total_sl_13_provinces.parquet")
parser.add_argument("-o", "--output", help="path: to save the forecasting result",
                    default="/path/to/company_data/user/<user>/cpc/result/202309/fct_1t_sum_13province_test_2023.parquet")

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
# df_total included 15 columns (13 columns of provinces and month and year, all has 15 columns)
df_total = pd.read_parquet(path_input,
                           filesystem=hdfs)

# create datetime dataframe
df_datetime = pd.DataFrame({'stt': [i for i in range(df_total.shape[0])]},
                           index=df_total.sort_index(ascending=True).index)
df_datetime['month'] = df_datetime.index.month
df_datetime['year'] = df_datetime.index.year

# read parameters from file
dfp = pd.read_parquet(path_params, filesystem=hdfs)

# create dataframe to save the forecasting result
df_result = pd.DataFrame()
# name of provinces
lst_pr = df_total.columns

# get month and year from input to forecasting
try:
    date_range = pd.date_range(start=from_date, end=to_date, freq='MS')
except:
    print('forecast the next month in the data')
    date_range = pd.date_range(start=f"{df_total.index.max().year}-{df_total.index.max().month + 1}",
                               end=f"{df_total.index.max().year}-{df_total.index.max().month + 1}", freq='MS')


for n_province in trange(13):
    df = df_total.iloc[:, [n_province]]
    name_province = lst_pr[n_province]
    df = df.sort_index()
    dp = dfp[dfp.province == name_province]
    lag = 6
    n_max = dp.n_max.values[0]
    type_decomp = dp.type_decomp.values[0]
    method_season = dp.method_season.values[0]
    method_trend = dp.method_trend.values[0]
    method_resid = dp.method_resid.values[0]
    p = 1
    d = 0
    q = 1
    for year_month in date_range:
        year, month = (year_month.year, year_month.month)
        # for year in years.strip('][').split(','):
        #     year = int(year)
        #     for month in months.strip('][').split(','):
        #         month= int(month)
        #         # print(month)
        y_trend, y_seasonal, y_residual = predict_master(
            data=df,
            d_date=df_datetime,
            month=month, year=year,
            lag=lag, p=1, d=0, q=1,
            n_max=n_max,
            type_decomp=type_decomp,
            methods_season=method_season, method_resid=method_resid,
            method_trend=method_trend, order1=1,
            order2=1, name_province=name_province)

        y_pr = y_trend + y_seasonal + y_residual
        try:
            error, y_tr = compute_error(df, month, year, y_pr)
            df_result = pd.concat([df_result, pd.DataFrame(
                {'n_max': n_max,
                 'province': name_province,
                 'year': year,
                 'month': month,
                 'y_pr': y_pr,
                 'y_tr': y_tr,
                 'error': error,
                 'error_abs': np.abs(error),
                 'y_trend': y_trend,
                 'y_seasonal': y_seasonal,
                 'y_residual': y_residual,
                 'lag': lag,
                 'p': p, 'd': d, 'q': q,
                 'type_decomp': type_decomp,
                 'method_season': method_season,
                 'method_resid': method_resid,
                 'method_trend': method_trend,
                 })])
        except:
            df_result = pd.concat([df_result, pd.DataFrame(
                {'n_max': n_max,
                 'province': name_province,
                 'year': year,
                 'month': month,
                 'y_pr': y_pr,
                 'y_tr': [None],
                 'error': [None],
                 'error_abs': [None],
                 'y_trend': y_trend,
                 'y_seasonal': y_seasonal,
                 'y_residual': y_residual,
                 'lag': lag,
                 'p': p, 'd': d, 'q': q,
                 'type_decomp': type_decomp,
                 'method_season': method_season,
                 'method_resid': method_resid,
                 'method_trend': method_trend,
                 })])
            break

# df_result['date'] = df_result.apply(lambda x:pd.to_datetime(f"{x.year}-{x.month}-01"),axis=1)
df_result.to_parquet(path_output, filesystem=hdfs)
