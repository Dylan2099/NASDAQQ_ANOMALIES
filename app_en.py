# app_en.py
import streamlit as st
import pandas as pd
from datetime import datetime
import json

# Import the anomaly detector logic
import anomaly_detector_logic as adl 

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Market Anomaly Detector", layout="wide")

st.title("📈 Market Anomaly Detector")
st.markdown("""
This application uses an **Isolation Forest** model to detect anomalies in the adjusted closing prices of ETFs.
Data is sourced from Yahoo Finance. An agent-based architecture and a "Model Context Protocol" (MCP)
are implemented for process traceability.
""")

# --- Sidebar for Inputs ---
st.sidebar.header("Analysis Parameters")

# ETF Ticker
ticker_default = "QQQ" # Nasdaq 100 ETF
etf_options = ["QQQ", "SPY", "DIA", "IWM", "GLD", "SLV", "AAPL", "MSFT", "GOOGL", "TSLA"] # Common options #new feature :)
selected_ticker = st.sidebar.selectbox(
    "Select ETF/Stock Ticker:",
    options=etf_options,
    index=etf_options.index(ticker_default),
    help="Examples: QQQ (Nasdaq 100), SPY (S&P 500), AAPL (Apple Inc.)"
)
custom_ticker = st.sidebar.text_input("Or enter a custom Ticker (e.g., VOO):", "")
ticker_to_analyze = custom_ticker if custom_ticker else selected_ticker


# Time Period
period_options_en = { #  keys for the display
    "Last year": "1y",
    "Last 2 years": "2y",
    "Last 5 years": "5y",
    "Last 10 years": "10y",
    "Maximum available": "max"
}
selected_period_label_en = st.sidebar.selectbox(
    "Select Time Period:",
    options=list(period_options_en.keys()),
    index=1 # Default to "Last 2 years"
)
period_to_analyze = period_options_en[selected_period_label_en]

# Model Parameters (Contamination)
contamination_rate = st.sidebar.slider(
    "Model Sensitivity (Contamination):",
    min_value=0.005,
    max_value=0.10,
    value=0.02, # Default value same as BEST_MODEL_CONFIG
    step=0.005,
    help="Expected proportion of anomalies in the data. Higher values detect more anomalies."
)

model_params_from_ui = {"contamination": contamination_rate}

# Button to start analysis
run_button = st.sidebar.button("🚀 Detect Anomalies")

# --- Main Area for Results ---
if run_button:
    # Use English label for the period in the header
    st.header(f"Results for {ticker_to_analyze} ({selected_period_label_en})")


    # Placeholder for progress messages
    progress_area = st.empty()
    def streamlit_progress_callback(message):
        # If messages from adl are in Spanish, translate them here or modify adl
        # Example of a simple check (not robust):
        if "Inicializando contexto..." in message:
            message = "Initializing context..."
        elif "Descargando datos para" in message:
            message = message.replace("Descargando datos para", "Downloading data for").replace("Descarga completada.", "Download complete.")
        elif "Creando características..." in message:
            message = "Preprocessing: Creating features..."
        elif "Datos listos para el modelo." in message:
            message = "Preprocessing: Data ready for model."
        elif "Aplicando modelo IA (Isolation Forest)..." in message:
            message = "Modeling: Applying AI model (Isolation Forest)..."
        elif "Se han detectado" in message and "anomalías" in message:
            parts = message.split(" ")
            num_anomalies = parts[3] # Assuming "Se han detectado X anomalías."
            message = f"Modeling: {num_anomalies} anomalies detected."
        elif "Preparando resultados..." in message:
            message = "Reporting: Preparing results..."
        elif "Visualización y reporte listos." in message:
            message = "Reporting: Visualization and report ready."
        progress_area.info(message)


    try:
        # 1. Initialize Model Context Protocol
        # For simplicity, the first message is hardcoded here.
        # If adl.initialize_context prints, that would be harder to control from here.
        streamlit_progress_callback("Initializing context...") # Keep this in sync with the callback
        contexto = adl.initialize_context(ticker_to_analyze, period_to_analyze, model_params_from_ui)
        
        # 2. Execute agent sequence
        with st.spinner("Processing... This may take a few seconds."):
            contexto = adl.get_data_agent(contexto, streamlit_progress_callback)
            contexto = adl.preprocess_data_agent(contexto, streamlit_progress_callback)
            contexto = adl.anomaly_detection_agent(contexto, streamlit_progress_callback)
            
            # 3. Generate final report
            # If reporting_agent generates plot titles/labels in Spanish, they will appear in Spanish.
            # This would require passing language preference to adl.reporting_agent.
            contexto, df_anomalies, fig_anomalies = adl.reporting_agent(contexto, streamlit_progress_callback)

        progress_area.success("Analysis complete!")

        # Display results
        st.subheader("🗓️ Anomalies Detected")
        if df_anomalies is not None and not df_anomalies.empty:
            st.dataframe(df_anomalies) # Column names like 'Anomalous_Price_USD' are from adl
            st.markdown(f"Found **{len(df_anomalies)}** anomalies.")
        elif df_anomalies is not None and df_anomalies.empty:
            st.info("✅ No anomalies detected with the current parameters.")
        else:
            st.warning("Could not generate the anomaly report (check pipeline log).")


        st.subheader("📊 Price and Anomaly Visualization")
        if fig_anomalies:
            st.pyplot(fig_anomalies) # Plot titles/labels from adl.reporting_agent
        else:
            st.warning("Could not generate visualization.")

        # Display final MCP object
        with st.expander("🔍 View Process Details (Model Context Protocol - MCP)"):
            context_to_display = contexto.copy()
            if 'data' in context_to_display and isinstance(context_to_display['data'], dict):
                context_to_display['data'].pop('raw_dataframe', None)
                context_to_display['data'].pop('processed_dataframe', None)
            
            def datetime_converter_for_json(o):
                if isinstance(o, datetime):
                    return o.isoformat()

            try:
                json_output_mcp = json.dumps(context_to_display, indent=2, default=datetime_converter_for_json)
                st.json(json_output_mcp)
            except TypeError as te:
                st.error(f"Error serializing MCP to JSON: {te}")
                st.text("There might be unhandled data types in the context.")
                st.text(str(context_to_display))

    except ValueError as ve:
        st.error(f"Value Error during processing: {ve}")
        if 'contexto' in locals():
             with st.expander("MCP Details at the time of error"):
                # Using default=str as a fallback for datetime conversion if specific converter fails or is not defined here
                st.json(json.dumps(contexto, indent=2, default=str))
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.error("Please check the parameters or try a different Ticker/Period.")
        if 'contexto' in locals():
            with st.expander("MCP Details at the time of error"):
                st.json(json.dumps(contexto, indent=2, default=str))

else:
    st.info("Configure parameters in the sidebar and press 'Detect Anomalies'.")

st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Dylan Saenz")
st.sidebar.markdown("https://www.linkedin.com/in/dylan-nicol%C3%A1s-s%C3%A1enz-chavarro-a087a218b/")
