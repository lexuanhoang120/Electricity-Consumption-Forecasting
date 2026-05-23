# CPC Prediction Project

This project forecasts electricity consumption (CPC) for 13 provinces in Vietnam using historical data.  It provides two prediction approaches:

- **T+1:** Predicts consumption for the next month.
- **T+12:** Predicts consumption for the next 12 months.

## Project Structure

- **cpc:** Contains the virtual environment setup (`environment.yml`).
- **jupyter-notebook:** Jupyter notebook (`CPC_Forecast.ipynb`) demonstrating the pipeline.
- **main:** Core Python scripts for predictions:
    - `forecast_t1_13pr_sum.py`:  Runs T+1 prediction.
    - `forecast_t12_13pr_sum.py`: Runs T+12 prediction.
    - `helper.py`: Helper functions.
    - `requirements.txt`: Required Python packages.

## Getting Started

### Prerequisites

1. **Python 3.x:**  Ensure you have Python 3 installed.
2. **Virtual Environment (Recommended):** Create a virtual environment and activate it.
3. **Install Dependencies:** 
   ```bash
   pip install -r requirements.txt
## Data Preparation

Prepare your data as follows:

- **Data Format:** Historical electricity consumption data for the 13 provinces.
- **Format:** Ensure data is in a suitable format (e.g., CSV, Parquet).
- **Important Columns:** Your data should include at least:
  - `Date`: In a format the code can understand (YYYY-MM-DD recommended).
  - `Province`: The name of the province.
  - `Consumption`: The electricity consumption value for that province and date.

## Update Data Paths

1. Open `main/forecast_t1_13pr_sum.py` and `main/forecast_t12_13pr_sum.py`.
2. Modify the following paths to point to your actual data files:
   - `path_parameter`: Path to the parameter file (default is provided, but you might need to adjust).
   - `input_data`: Path to your prepared historical data file.
   - `output_path`: Directory where you want to save the prediction results.

## Running Predictions

1. **Open Terminal/Command Prompt:** Navigate to the project directory.

2. **Run the Script:**
   ```bash
   python main/forecast_t1_13pr_sum.py -f <From_date> -t <To_date> -p <Path_to_params> -i <Path_to_input> -o <Path_to_output>
   python main/forecast_t12_13pr_sum.py -f <From_date> -t <To_date> -p <Path_to_params> -i <Path_to_input> -o <Path_to_output>
   ```
Parameters
- From_date: Start month for the forecast (format: MM/YYYY).
- To_date: End month for the forecast (format: MM/YYYY). If not provided, the forecast will be for the next month based on the input data.
- Path_to_params: Path to the configuration file for the 13 provinces.
- Path_to_input: Path to the input file containing electricity data for the 13 provinces from 2014 onwards.
- Path_to_output: Path where the forecast results will be saved.

Replace <From_date> and <To_date> as needed for the prediction period in MM-YYYY format.

Example:

   ```bash
   python3 forecast_t1_13pr_sum.py -f 01/2023 -t 03/2023 -o /path/to/output.parquet
   python3 forecast_t12_13pr_sum.py -f 01/2023 -t 03/2023 -o /path/to/output.parquet

   ```
## Understanding the Output

T+1 Output: Columns include province, year, month, y_pr, y_tr.

T+12 Output: Columns include province, year, month, y_tr1 to y_tr12, y_pr1 to y_pr12.

The output files will likely contain the following columns:

- Province: The name of the province.
- Year: The year of the prediction.
- Month: The month of the prediction.
- y_tr (Target): The actual historical value (if available).
- y_pr (Prediction): The predicted consumption value.
- Other Columns: Depending on the script and data, additional columns may include decomposed components (e.g., trend, seasonality), etc.

## Notes
- Update the input file path each month to forecast the next month.
- The T+12 forecast can be used to generate T+1 results using the y_pr1 output.
- The prediction results will be saved in Parquet format in the directory specified in output_path. 

This repos was revised from original code in the internal project from previous company in 2023.
