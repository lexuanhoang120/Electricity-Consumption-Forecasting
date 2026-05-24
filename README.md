# CPC Prediction Project

This project forecasts monthly electricity consumption (CPC) for 13 provinces in Vietnam using historical electricity consumption data.

> **Note:** This repository is revised from original internal project code from a previous company in 2023.
> Private information, internal infrastructure paths, and confidential business details have been removed.
> This demo does not use real electricity consumption data; it is generated for demonstration purposes only.
>
It supports two forecasting modes:

- **T+1 Forecasting**: predict electricity consumption for the next month.
- **T+12 Forecasting**: predict electricity consumption for the next 12 months.

---

## Demo

A quick demo is included to illustrate the forecasting pipeline using synthetic historical data. This demo does not use real electricity consumption data; it is generated for demonstration purposes only. The demo generates predicted values for each forecast component:

- `trend` — long-term trend forecast
- `season` — seasonal pattern forecast
- `resid` — residual forecast
- `y_pr` — combined final prediction

It also saves visualizations of the forecast results.

Run the demo with:

```bash
python3 demo_prediction.py
```

The demo produces the following files in `docs/`:

- `docs/demo_forecast_next_month.csv` — T+1 forecast with component breakdown and final prediction
- `docs/demo_forecast_12_months.csv` — recursive T+12 forecast with component breakdown and final prediction
- `docs/forecast_sample.png` — combined sample plot showing history and predicted values for example provinces
- `docs/forecast_<province>.png` — individual province plots
- `docs/forecast_BDINH.png` — BDINH province history and total forecast plot
- `docs/forecast_BDINH_total.png` — BDINH total forecast vs historical consumption
- `docs/forecast_BDINH_components.png` — BDINH component forecast showing trend, seasonal, and residual predictions
- `docs/forecast_BDINH_components_three_plots.png` — BDINH component forecasts in three separate charts

![Forecast sample](docs/forecast_sample.png)

#### BDINH component forecast

![BDINH total forecast](docs/forecast_BDINH_total.png)

![BDINH components three plots](docs/forecast_BDINH_components_three_plots.png)

This specific BDINH demo visualizes:

- the historical BDINH series and total forecast
- the forecasted trend, seasonal, and residual components separately
- a dedicated BDINH province forecast plot
- the component predictions shown as three separate charts for clarity

---

## Project Overview

The goal of this project is to forecast monthly electricity consumption at province level using historical data from 2014 onward.

The core forecasting method is based on time-series decomposition. Each monthly consumption series is decomposed into three components:

1. **Trend**
2. **Seasonality**
3. **Residual**

Each component is predicted using a dedicated model, then combined to produce the final forecast.

```text
Final Prediction = Trend Prediction + Seasonal Prediction + Residual Prediction
```

---

## Forecasting Method

### 1. Trend Prediction

The trend component captures the long-term movement of electricity consumption.

A **linear model** is used to forecast the trend component. This helps the forecast follow the general growth, decline, or stable movement of each province's consumption pattern.

### 2. Seasonal Prediction

The seasonal component captures repeated monthly and yearly consumption behavior.

A **statistical model** is used to forecast seasonality. The model uses historical electricity consumption and cumulative consumption patterns.

The statistical model is controlled by several parameters:

| Parameter | Description |
|---|---|
| `lag` | Number of historical months used for estimation |
| `order1` | Function order used to estimate monthly consumption |
| `order2` | Function order used to estimate cumulative consumption |
| `shift` | Number of steps to shift historical data backward |

A parameter set is selected for each month from January to December.

### 3. Residual Prediction

The residual component represents the remaining variation after removing trend and seasonality.

An **ARIMA model** is used to forecast the residual component. This helps capture short-term patterns that are not explained by the trend or seasonal model.

### 4. Multi-Month Forecasting

For forecasts beyond one month, the project uses a **recursive strategy**.

The model first predicts `T+1`, then uses that prediction as part of the input to predict the next step.

```text
Predict T+1 -> update input -> predict T+2 -> update input -> ... -> predict T+12
```

This allows the one-step forecasting logic to be reused for longer forecast horizons. However, forecast error may increase for farther months because errors can propagate through recursive steps.

---

## Results at Development Time

The model was evaluated on monthly electricity consumption data for 13 provinces.

Main reported results:

- Evaluation period: **January 2022 to July 2023**
- Metric: **MAPE**
- Average MAPE below **6%** for all 13 provinces
- Average MAPE below **4%** for 6 out of 13 provinces
- January 2023 showed over-forecasting across all provinces
- DNONG had the most unstable forecast error among the provinces
- For long-horizon forecasting, error generally increased as the forecast horizon became farther

---

## Achievement Summary

Developed a decomposition-based electricity consumption forecasting pipeline for 13 provinces. The method combines a linear model for trend prediction, a statistical parameter-based model for seasonal prediction, and ARIMA for residual prediction. It also supports recursive multi-step forecasting for T+12 prediction. At development time, the model achieved average MAPE below 6% across all 13 provinces, with 6 provinces below 4% MAPE.

---

## Project Structure

```text
.
├── cpc/
│   └── environment.yml
│
├── jupyter-notebook/
│   └── CPC_Forecast.ipynb
│
└── main/
    ├── forecast_t1_13pr_sum.py
    ├── forecast_t12_13pr_sum.py
    ├── helper.py
    └── requirements.txt
```

| Folder / File | Description |
|---|---|
| `cpc/environment.yml` | Conda environment setup file |
| `jupyter-notebook/CPC_Forecast.ipynb` | Notebook demonstrating the forecasting pipeline |
| `main/forecast_t1_13pr_sum.py` | Script for T+1 forecasting |
| `main/forecast_t12_13pr_sum.py` | Script for T+12 forecasting |
| `main/helper.py` | Helper functions |
| `main/requirements.txt` | Required Python packages |

---

## Getting Started

### Prerequisites

- Python 3.x
- Virtual environment is recommended

### Install Dependencies

Using pip:

```bash
pip install -r main/requirements.txt
```

Or using Conda:

```bash
conda env create -f cpc/environment.yml
```

---

## Data Preparation

Prepare historical electricity consumption data for the 13 provinces.

The input data should include at least the following columns:

| Column | Description |
|---|---|
| `Date` | Date or month of the observation. `YYYY-MM-DD` is recommended |
| `Province` | Province name |
| `Consumption` | Electricity consumption value |

Supported file formats may include CSV or Parquet, depending on the script configuration.

---

## Update Data Paths

Before running the scripts, update the input and output paths in:

```text
main/forecast_t1_13pr_sum.py
main/forecast_t12_13pr_sum.py
```

Update the following paths or pass them through command-line arguments:

| Path | Description |
|---|---|
| `path_parameter` | Path to the parameter/configuration file |
| `input_data` | Path to the prepared historical data |
| `output_path` | Path where prediction results will be saved |

---

## Running Predictions

### T+1 Forecast

```bash
python main/forecast_t1_13pr_sum.py \
  -f <From_date> \
  -t <To_date> \
  -p <Path_to_params> \
  -i <Path_to_input> \
  -o <Path_to_output>
```

### T+12 Forecast

```bash
python main/forecast_t12_13pr_sum.py \
  -f <From_date> \
  -t <To_date> \
  -p <Path_to_params> \
  -i <Path_to_input> \
  -o <Path_to_output>
```

### Parameters

| Parameter | Description |
|---|---|
| `-f`, `From_date` | Start month for forecast, format `MM/YYYY` |
| `-t`, `To_date` | End month for forecast, format `MM/YYYY` |
| `-p`, `Path_to_params` | Path to the parameter file for 13 provinces |
| `-i`, `Path_to_input` | Path to input electricity data |
| `-o`, `Path_to_output` | Path to save forecast results |


### Demo preview

![Forecast sample](docs/forecast_sample.png)

### Example

```bash
python main/forecast_t1_13pr_sum.py \
  -f 01/2023 \
  -t 03/2023 \
  -p /path/to/params \
  -i /path/to/input.parquet \
  -o /path/to/output_t1.parquet
```

```bash
python main/forecast_t12_13pr_sum.py \
  -f 01/2023 \
  -t 03/2023 \
  -p /path/to/params \
  -i /path/to/input.parquet \
  -o /path/to/output_t12.parquet
```

---

## Output

Prediction results are saved in Parquet format.

### T+1 Output

Main columns:

| Column | Description |
|---|---|
| `Province` | Province name |
| `Year` | Forecast year |
| `Month` | Forecast month |
| `y_tr` | Actual value, if available |
| `y_pr` | Predicted value |

### T+12 Output

Main columns:

| Column | Description |
|---|---|
| `Province` | Province name |
| `Year` | Forecast year |
| `Month` | Forecast month |
| `y_tr1` to `y_tr12` | Actual values for each forecast horizon, if available |
| `y_pr1` to `y_pr12` | Predicted values for each forecast horizon |

The `y_pr1` output from T+12 can also be used as a T+1 forecast.

---

## Notes

- Update the input data path each month before running a new forecast.
- T+12 forecasting uses recursive prediction, so long-horizon errors may increase.
- This repository is revised from original internal project code from a previous company in 2023.
- Private information, internal infrastructure paths, and confidential business details have been removed.