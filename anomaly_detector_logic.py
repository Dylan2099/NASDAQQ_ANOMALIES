# ==============================================================================
# END-TO-END SOLUTION: MARKET ANOMALY DETECTOR - LOGIC
# Dylan Saenz - 06/2025
# ==============================================================================
import yfinance as yf
import pandas as pd
from sklearn.ensemble import IsolationForest
from datetime import datetime
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- DEFAULT MODEL CONFIGURATION ---
DEFAULT_MODEL_CONFIG = {
    "name": "IsolationForest_configurable",
    "params": {
        "contamination": 0.02, # This will be configurable from the UI
        "random_state": 42
    }
}

# ==============================================================================
# POINTS 4 and 5: AGENT ARCHITECTURE AND MODEL CONTEXT PROTOCOL (MCP)
# ==============================================================================

def initialize_context(ticker, period, model_config_params):
    """(MCP) Initializes the context object that will flow between agents."""
    current_model_config = DEFAULT_MODEL_CONFIG.copy()
    current_model_config["params"].update(model_config_params) # Update with params from UI

    return {
        "run_id": f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "status": "Initialized",
        "input_params": {"ticker": ticker, "period": period},
        "model_config": current_model_config,
        "pipeline_log": ["Context initialized."],
        "data": {},
        "results": {}
    }

def get_data_agent(context, progress_callback=None):
    """(Agent 1) Downloads the required historical data."""
    ticker, period = context["input_params"]["ticker"], context["input_params"]["period"]
    if progress_callback: progress_callback(f"[DATA AGENT] Downloading data for {ticker} ({period})...")
    
    # POINT 1: Nasdaq ETF price over the past X years
    data = yf.download(ticker, period=period, auto_adjust=False, progress=False) # progress=False to avoid yfinance prints
    
    if data.empty:
        context["status"] = "Error: Data Download Failed"
        context["pipeline_log"].append(f"Error: Could not download data for {ticker}.")
        raise ValueError(f"Could not download data for {ticker}.")
    
    # Handle MultiIndex if it exists (common in yfinance for some calls)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns.values]
        # If generic names like 'Adj Close_' still exist after joining, find the correct one
        adj_close_col = next((col for col in data.columns if 'Adj Close' in col), 'Adj Close')
        volume_col = next((col for col in data.columns if 'Volume' in col), 'Volume')
        
        # Ensure the columns we need exist
        required_cols = {adj_close_col: 'Adj Close', volume_col: 'Volume'}
        if not all(col in data.columns for col in required_cols.keys()):
             context["status"] = "Error: Missing required columns after download"
             context["pipeline_log"].append(f"Error: Columns {adj_close_col} or {volume_col} not found.")
             raise ValueError(f"Required columns not found in downloaded data for {ticker}.")
        data = data.rename(columns=required_cols)

    elif not all(col in data.columns for col in ['Adj Close', 'Volume']):
        context["status"] = "Error: Missing Adj Close or Volume"
        context["pipeline_log"].append(f"Error: Adj Close or Volume column not found for {ticker}.")
        # Try common names if it fails
        if 'Close' in data.columns and 'Adj Close' not in data.columns:
            data.rename(columns={'Close':'Adj Close'}, inplace=True)
        if not all(col in data.columns for col in ['Adj Close', 'Volume']):
             raise ValueError(f"Adj Close or Volume column not found in downloaded data for {ticker}.")

    context['data']['raw_dataframe'] = data
    context['status'] = "Data Downloaded"
    context["pipeline_log"].append("[DATA AGENT] Download complete.")
    if progress_callback: progress_callback("[DATA AGENT] Download complete.")
    return context

def preprocess_data_agent(context, progress_callback=None):
    """(Agent 2) Prepares data for the model, creating features."""
    if progress_callback: progress_callback("[PREPROCESSING AGENT] Creating features...")
    raw_data = context['data']['raw_dataframe']
    
    # POINT 2: Time Series Data (Date, PRICE) - We use 'Adj Close' as PRICE
    price_data = raw_data[['Adj Close', 'Volume']].copy()
    price_data.rename(columns={'Adj Close': 'price'}, inplace=True) # Renaming here simplifies
    
    # Feature Engineering
    w_vol, w_trend = 7, 21 # Could also be parameters
    price_data['volatility'] = price_data['price'].rolling(window=w_vol).std()
    price_data['change_pct'] = price_data['price'].pct_change() * 100
    price_data['detrended'] = price_data['price'] - price_data['price'].rolling(window=w_trend).mean()
    price_data['relative_volume'] = price_data['Volume'] / price_data['Volume'].rolling(window=w_trend).mean()
    price_data.dropna(inplace=True)
    
    if price_data.empty:
        context["status"] = "Error: Preprocessing resulted in empty data"
        context["pipeline_log"].append("Error: No data left after preprocessing (check rolling windows and data length).")
        raise ValueError("No data left after preprocessing. The period might be too short for the rolling windows.")

    context['data']['processed_dataframe'] = price_data
    context['status'] = "Data Preprocessed"
    context["pipeline_log"].append("[PREPROCESSING AGENT] Data ready for model.")
    if progress_callback: progress_callback("[PREPROCESSING AGENT] Data ready for model.")
    return context

def anomaly_detection_agent(context, progress_callback=None):
    """(Agent 3) Builds, trains, and runs the anomaly detection model."""
    if progress_callback: progress_callback("[MODELING AGENT] Applying AI model (Isolation Forest)...")
    
    price_data = context['data']['processed_dataframe']
    model_config = context['model_config']
    features_to_use = ['price', 'volatility', 'change_pct', 'detrended','relative_volume']
    
    # Ensure all features exist
    missing_features = [f for f in features_to_use if f not in price_data.columns]
    if missing_features:
        context["status"] = f"Error: Missing features for model: {', '.join(missing_features)}"
        context["pipeline_log"].append(context["status"])
        raise ValueError(context["status"])

    X = price_data[features_to_use].values
    
    model = IsolationForest(**model_config['params'])
    price_data['anomaly_score'] = model.fit_predict(X)
    
    anomalies = price_data[price_data['anomaly_score'] == -1]
    
    context['results']['anomalies_found'] = anomalies
    context['status'] = "Anomalies Detected"
    context["pipeline_log"].append(f"[MODELING AGENT] {len(anomalies)} anomalies detected.")
    if progress_callback: progress_callback(f"[MODELING AGENT] {len(anomalies)} anomalies detected.")
    return context

def reporting_agent(context, progress_callback=None):
    """(Agent 4) Prepares data for the report and visualization."""
    if progress_callback: progress_callback("[REPORTING AGENT] Preparing results...")

    if context['status'] != "Anomalies Detected":
        context["pipeline_log"].append("Pipeline did not complete successfully to generate report.")
        # We don't raise an error here, status already indicates it
        return context, None, None # Returns context, None for df, None for fig

    anomalies = context['results']['anomalies_found']
    
    # POINT 6: output: date and price of the anomalies
    final_output_df = anomalies[['price']].copy()
    final_output_df.index = final_output_df.index.strftime('%Y-%m-%d')
    final_output_df.rename(columns={'price': 'Anomalous_Price_USD'}, inplace=True) # Keeping column name as is, as it's somewhat standard
    
    context['results']['anomalies_report_df'] = final_output_df
    
    # --- Preparation for Visualization ---
    all_data = context['data']['processed_dataframe']
    ticker = context['input_params']['ticker']
    
    plt.style.use('seaborn-v0_8-whitegrid') # Updated style
    fig, ax = plt.subplots(figsize=(15, 7)) # Adjusted size for Streamlit

    ax.plot(all_data.index, all_data['price'], color='royalblue', lw=1.5, label=f'Historical Price {ticker}')
    if not anomalies.empty:
        ax.scatter(anomalies.index, anomalies['price'], color='crimson', s=80, ec='black', lw=1.5, zorder=5, label=f'Anomalies Detected ({len(anomalies)})')
    
    ax.set_title(f'Market Anomaly Detector - {ticker}', fontsize=18, pad=15)
    ax.set_ylabel('Adjusted Close Price (USD)', fontsize=14)
    ax.set_xlabel('Date', fontsize=14)
    ax.legend(loc='upper left', fontsize=12, fancybox=True, frameon=True, shadow=True)
    ax.grid(True, which='major', linestyle='--', linewidth=0.7)
    
    # Format dates on X-axis for better readability
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    fig.tight_layout()
    
    context["pipeline_log"].append("[REPORTING AGENT] Visualization and report ready.")
    if progress_callback: progress_callback("[REPORTING AGENT] Visualization and report ready.")
    
    return context, final_output_df, fig


# --- MAIN ORCHESTRATOR (For local testing, not used by Streamlit directly) ---
if __name__ == "__main__":
    
    # Execution Configuration
    TICKER_TO_ANALYZE = "QQQ"
    TIME_PERIOD = "2y"
    MODEL_PARAMS_FROM_UI = {"contamination": 0.04} # UI Example
    
    print(f"Starting analysis for {TICKER_TO_ANALYZE}, period {TIME_PERIOD}, contamination {MODEL_PARAMS_FROM_UI['contamination']}")

    def simple_progress_printer(message):
        print(message)

    try:
        # 1. Initialize Model Context Protocol
        current_context = initialize_context(TICKER_TO_ANALYZE, TIME_PERIOD, MODEL_PARAMS_FROM_UI)
        
        # 2. Execute agent sequence
        current_context = get_data_agent(current_context, simple_progress_printer)
        current_context = preprocess_data_agent(current_context, simple_progress_printer)
        current_context = anomaly_detection_agent(current_context, simple_progress_printer)
        
        # 3. Generate final report
        current_context, df_anomalies, fig_anomalies = reporting_agent(current_context, simple_progress_printer)
        
        print("\n" + "="*50)
        print("ANOMALY DETECTION DEMO - FINAL RESULT")
        print("="*50)

        if df_anomalies is not None and not df_anomalies.empty:
            print("\n--- ANOMALIES DETECTED (Required Output) ---")
            print(df_anomalies.to_string())
        elif df_anomalies is not None and df_anomalies.empty:
            print("\n--- NO ANOMALIES DETECTED ---")
        else:
            print("\n--- ERROR GENERATING ANOMALY REPORT ---")
            
        if fig_anomalies:
            print("\n[REPORTING AGENT] Showing visualization...")
            plt.show()
        
        # (Optional) Display final MCP object to demonstrate compliance
        context_to_print = current_context.copy()
        if 'data' in context_to_print and isinstance(context_to_print['data'], dict): # Check if 'data' key exists and is a dict
            if 'raw_dataframe' in context_to_print['data']:
                del context_to_print['data']['raw_dataframe'] # Clean up to avoid printing long dataframes
            if 'processed_dataframe' in context_to_print['data']:
                del context_to_print['data']['processed_dataframe']

        print("\n--- Final MCP Object (Traceability) ---")
        print(json.dumps(context_to_print, indent=2, default=str)) # default=str to handle datetime
        
    except Exception as e:
        print(f"\n\n--- FATAL PIPELINE ERROR ---")
        print(f"Error: {e}")
        # If context exists, print it for debug
        if 'current_context' in locals():
            print("\n--- MCP Context at time of error ---")
            context_to_print_on_error = current_context.copy()
            if 'data' in context_to_print_on_error and isinstance(context_to_print_on_error['data'], dict):
                if 'raw_dataframe' in context_to_print_on_error['data']: del context_to_print_on_error['data']['raw_dataframe']
                if 'processed_dataframe' in context_to_print_on_error['data']: del context_to_print_on_error['data']['processed_dataframe']
            print(json.dumps(context_to_print_on_error, indent=2, default=str))
