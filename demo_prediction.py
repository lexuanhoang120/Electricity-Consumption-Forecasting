import importlib.util
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

helper_path = Path(__file__).resolve().parent / 'jupyter-notebook' / 'main' / 'helper.py'
if not helper_path.exists():
    raise FileNotFoundError(f'Cannot find helper module at {helper_path}')
spec = importlib.util.spec_from_file_location('forecast_helper', helper_path)
forecast_helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(forecast_helper)
predict_master = forecast_helper.predict_master


def build_synthetic_data():
    index = pd.date_range(start='2014-01-01', end='2024-12-01', freq='MS')
    provinces = [
        'BDINH', 'DANANG', 'DLAC', 'DNONG', 'GLAI', 'KHOA', 'KTUM', 'PYEN',
        'QBINH', 'QNGAI', 'QNAM', 'QTRI', 'TTHUE'
    ]
    df = pd.DataFrame(index=index)
    for i, province in enumerate(provinces):
        seasonal = 20 + 10 * np.sin(2 * np.pi * (index.month - 1) / 12)
        trend = 0.5 * np.arange(len(index))
        noise = np.random.normal(scale=3.0, size=len(index))
        df[province] = 200 + 5 * i + seasonal + trend + noise
    return df


def build_params(provinces):
    return pd.DataFrame({
        'province': provinces,
        'n_max': [5] * len(provinces),
        'type_decomp': ['STL'] * len(provinces),
        'method_season': ['statistics'] * len(provinces),
        'method_trend': ['average'] * len(provinces),
        'method_resid': ['arima'] * len(provinces),
    })


def build_datetime_index(df, extra_months=12):
    start = df.index.min()
    end = df.index.max() + pd.DateOffset(months=extra_months)
    date_range = pd.date_range(start=start, end=end, freq='MS')
    df_datetime = pd.DataFrame({'stt': np.arange(len(date_range))}, index=date_range)
    df_datetime['month'] = df_datetime.index.month
    df_datetime['year'] = df_datetime.index.year
    return df_datetime


def predict_next_month(df, df_datetime, params):
    next_date = df.index.max() + pd.DateOffset(months=1)
    results = []
    for province in df.columns:
        row = params[params.province == province].iloc[0]
        y_trend, y_season, y_resid = predict_master(
            data=df[[province]],
            d_date=df_datetime,
            month=next_date.month,
            year=next_date.year,
            lag=6,
            p=1,
            d=0,
            q=1,
            n_max=row.n_max,
            type_decomp=row.type_decomp,
            methods_season=row.method_season,
            method_resid=row.method_resid,
            method_trend=row.method_trend,
            order1=1,
            order2=1,
            name_province=province,
        )
        results.append({
            'province': province,
            'year': next_date.year,
            'month': next_date.month,
            'y_pr': y_trend + y_season + y_resid,
            'trend': y_trend,
            'season': y_season,
            'resid': y_resid,
        })
    return pd.DataFrame(results)


def predict_12_months(df, df_datetime, params):
    forecast = []
    df_full = df.copy()
    start_year = df_full.index.max().year
    start_month = df_full.index.max().month + 1
    year = start_year
    month = start_month
    for step in range(12):
        if month > 12:
            month -= 12
            year += 1
        next_index = pd.Timestamp(f'{year}-{month:02d}-01')
        month_results = []
        for province in df.columns:
            row = params[params.province == province].iloc[0]
            y_trend, y_season, y_resid = predict_master(
                data=df_full[[province]],
                d_date=df_datetime,
                month=month,
                year=year,
                lag=6,
                p=1,
                d=0,
                q=1,
                n_max=row.n_max,
                type_decomp=row.type_decomp,
                methods_season=row.method_season,
                method_resid=row.method_resid,
                method_trend=row.method_trend,
                order1=1,
                order2=1,
                name_province=province,
            )
            y_pr = y_trend + y_season + y_resid
            df_full.at[next_index, province] = y_pr
            month_results.append({
                'province': province,
                'year': year,
                'month': month,
                'y_pr': y_pr,
                'trend': y_trend,
                'season': y_season,
                'resid': y_resid,
            })
        forecast.extend(month_results)
        month += 1
    return pd.DataFrame(forecast)


def ensure_docs_dir():
    docs_dir = Path(__file__).resolve().parent / 'docs'
    docs_dir.mkdir(parents=True, exist_ok=True)
    return docs_dir


def add_date_column(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df.assign(day=1)[['year', 'month', 'day']])
    return df


def plot_history_and_forecast(df, forecast_12, next_month, provinces, docs_dir):
    forecast_12 = add_date_column(forecast_12)
    next_month = add_date_column(next_month)
    for province in provinces:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(df.index, df[province], label='history', color='blue')
        ax.plot(
            forecast_12.loc[forecast_12.province == province, 'date'],
            forecast_12.loc[forecast_12.province == province, 'y_pr'],
            label='forecast T+12',
            color='red',
            marker='o',
            linestyle='--',
        )
        nm = next_month.loc[next_month.province == province]
        if not nm.empty:
            ax.scatter(nm.date.iloc[0], nm.y_pr.iloc[0], color='green', s=80, label='forecast T+1', zorder=4)
        ax.set_title(f'{province} historical data and forecasts')
        ax.set_xlabel('Date')
        ax.set_ylabel('Synthetic consumption')
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(docs_dir / f'forecast_{province}.png')
        plt.close(fig)

    sample_provinces = provinces[:3]
    fig, axs = plt.subplots(len(sample_provinces), 1, figsize=(10, 12), sharex=True)
    for ax, province in zip(axs, sample_provinces):
        ax.plot(df.index, df[province], label='history', color='blue')
        ax.plot(
            forecast_12.loc[forecast_12.province == province, 'date'],
            forecast_12.loc[forecast_12.province == province, 'y_pr'],
            label='forecast T+12',
            color='red',
            marker='o',
            linestyle='--',
        )
        nm = next_month.loc[next_month.province == province]
        if not nm.empty:
            ax.scatter(nm.date.iloc[0], nm.y_pr.iloc[0], color='green', s=80, label='forecast T+1', zorder=4)
        ax.set_title(f'{province}')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(alpha=0.3)
    axs[-1].set_xlabel('Date')
    fig.tight_layout()
    fig.savefig(docs_dir / 'forecast_sample.png')
    plt.close(fig)


def plot_bdin_h_components(df, forecast_12, next_month, docs_dir, province='BDINH'):
    forecast_12 = add_date_column(forecast_12)
    next_month = add_date_column(next_month)
    forecast_proc = forecast_12[forecast_12.province == province].copy()
    nm = next_month[next_month.province == province].copy()

    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    axs[0].plot(df.index, df[province], label='history', color='blue')
    axs[0].plot(
        forecast_proc['date'],
        forecast_proc['y_pr'],
        label='T+12 total forecast',
        color='red',
        marker='o',
        linestyle='--',
    )
    if not nm.empty:
        axs[0].scatter(nm['date'].iloc[0], nm['y_pr'].iloc[0], label='T+1 total forecast', color='green', s=100, zorder=4)
    axs[0].set_title(f'{province} total prediction vs historical values')
    axs[0].set_ylabel('Value')
    axs[0].legend()
    axs[0].grid(alpha=0.3)

    axs[1].plot(forecast_proc['date'], forecast_proc['trend'], label='trend forecast', color='orange', marker='o')
    axs[1].plot(forecast_proc['date'], forecast_proc['season'], label='season forecast', color='purple', marker='o')
    axs[1].plot(forecast_proc['date'], forecast_proc['resid'], label='residual forecast', color='brown', marker='o')
    if not nm.empty:
        axs[1].scatter(nm['date'].iloc[0], nm['trend'].iloc[0], label='T+1 trend', color='orange', edgecolors='black', s=80)
        axs[1].scatter(nm['date'].iloc[0], nm['season'].iloc[0], label='T+1 season', color='purple', edgecolors='black', s=80)
        axs[1].scatter(nm['date'].iloc[0], nm['resid'].iloc[0], label='T+1 residual', color='brown', edgecolors='black', s=80)
    axs[1].set_title(f'{province} component predictions')
    axs[1].set_ylabel('Component value')
    axs[1].legend()
    axs[1].grid(alpha=0.3)

    axs[-1].set_xlabel('Date')
    fig.tight_layout()
    fig.savefig(docs_dir / f'forecast_{province}_components.png')
    plt.close(fig)


def main():
    df = build_synthetic_data()
    provinces = list(df.columns)
    params = build_params(provinces)
    df_datetime = build_datetime_index(df, extra_months=24)

    print('=== Synthetic data overview ===')
    print(df.tail(3))

    next_month = predict_next_month(df, df_datetime, params)
    print('\n=== Next month forecast (T+1) ===')
    print(next_month[['province', 'year', 'month', 'y_pr']].to_string(index=False))

    forecast_12 = predict_12_months(df, df_datetime, params)
    print('\n=== 12-month forecast (T+12) for first 5 provinces ===')
    print(forecast_12.groupby('province').head(3)[['province', 'year', 'month', 'y_pr']].to_string(index=False))

    docs_dir = ensure_docs_dir()
    forecast_12.to_csv(docs_dir / 'demo_forecast_12_months.csv', index=False)
    next_month.to_csv(docs_dir / 'demo_forecast_next_month.csv', index=False)
    plot_history_and_forecast(df, forecast_12, next_month, provinces[:4], docs_dir)
    plot_bdin_h_components(df, forecast_12, next_month, docs_dir, province='BDINH')

    print(f'\nSaved demo outputs to {docs_dir}')
    print(f'CSV files: {docs_dir / "demo_forecast_next_month.csv"}, {docs_dir / "demo_forecast_12_months.csv"}')
    print(f'Plot files: {docs_dir / "forecast_sample.png"}, {docs_dir / "forecast_BDINH_components.png"}, and forecast_*.png')


if __name__ == '__main__':
    main()
