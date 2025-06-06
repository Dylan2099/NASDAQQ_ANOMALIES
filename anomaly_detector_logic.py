# ==============================================================================
# SOLUCIÓN END-TO-END: DETECTOR DE ANOMALÍAS DE MERCADO - LÓGICA
# Dylan Saenz - 06/2025
# ==============================================================================
import yfinance as yf
import pandas as pd
from sklearn.ensemble import IsolationForest
from datetime import datetime
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- CONFIGURACIÓN DEL MODELO POR DEFECTO ---
DEFAULT_MODEL_CONFIG = {
    "name": "IsolationForest_configurable",
    "params": {
        "contamination": 0.02, # Este será configurable desde el UI
        "random_state": 42
    }
}

# ==============================================================================
# PUNTO 4 y 5: ARQUITECTURA DE AGENTES Y MODEL CONTEXT PROTOCOL (MCP)
# ==============================================================================

def initialize_context(ticker, period, model_config_params):
    """(MCP) Inicia el objeto de contexto que fluirá entre los agentes."""
    current_model_config = DEFAULT_MODEL_CONFIG.copy()
    current_model_config["params"].update(model_config_params) # Actualiza con params del UI

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
    """(Agente 1) Descarga los datos históricos requeridos."""
    ticker, period = context["input_params"]["ticker"], context["input_params"]["period"]
    if progress_callback: progress_callback(f"[AGENTE DE DATOS] Descargando datos para {ticker} ({period})...")
    
    # PUNTO 1: Nasdaq ETF price over the past X years
    data = yf.download(ticker, period=period, auto_adjust=False, progress=False) # progress=False para evitar prints de yfinance
    
    if data.empty:
        context["status"] = "Error: Data Download Failed"
        context["pipeline_log"].append(f"Error: No se pudieron descargar datos para {ticker}.")
        raise ValueError(f"No se pudieron descargar datos para {ticker}.")
    
    # Manejar MultiIndex si existe (común en yfinance para algunas llamadas)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns.values]
        # Si después de joinear aún quedan nombres genéricos como 'Adj Close_', buscar el correcto
        adj_close_col = next((col for col in data.columns if 'Adj Close' in col), 'Adj Close')
        volume_col = next((col for col in data.columns if 'Volume' in col), 'Volume')
        
        # Asegurarse que las columnas que necesitamos existen
        required_cols = {adj_close_col: 'Adj Close', volume_col: 'Volume'}
        if not all(col in data.columns for col in required_cols.keys()):
             context["status"] = "Error: Missing required columns after download"
             context["pipeline_log"].append(f"Error: Columns {adj_close_col} or {volume_col} not found.")
             raise ValueError(f"Required columns not found in downloaded data for {ticker}.")
        data = data.rename(columns=required_cols)


    elif not all(col in data.columns for col in ['Adj Close', 'Volume']):
        context["status"] = "Error: Missing Adj Close or Volume"
        context["pipeline_log"].append(f"Error: Adj Close or Volume column not found for {ticker}.")
        # Intentar con nombres comunes si falla
        if 'Close' in data.columns and 'Adj Close' not in data.columns:
            data.rename(columns={'Close':'Adj Close'}, inplace=True)
        if not all(col in data.columns for col in ['Adj Close', 'Volume']):
             raise ValueError(f"Adj Close or Volume column not found in downloaded data for {ticker}.")


    context['data']['raw_dataframe'] = data
    context['status'] = "Data Downloaded"
    context["pipeline_log"].append("[AGENTE DE DATOS] Descarga completada.")
    if progress_callback: progress_callback("[AGENTE DE DATOS] Descarga completada.")
    return context

def preprocess_data_agent(context, progress_callback=None):
    """(Agente 2) Prepara los datos para el modelo, creando características."""
    if progress_callback: progress_callback("[AGENTE DE PREPROCESAMIENTO] Creando características...")
    raw_data = context['data']['raw_dataframe']
    
    # PUNTO 2: Time Series Data (Date, PRICE) - Usamos 'Adj Close' como PRICE
    price_data = raw_data[['Adj Close', 'Volume']].copy()
    price_data.rename(columns={'Adj Close': 'price'}, inplace=True) # Renombrar aquí simplifica
    
    # Ingeniería de Características
    w_vol, w_trend = 7, 21 # Podrían ser parámetros también
    price_data['volatility'] = price_data['price'].rolling(window=w_vol).std()
    price_data['change_pct'] = price_data['price'].pct_change() * 100
    price_data['detrended'] = price_data['price'] - price_data['price'].rolling(window=w_trend).mean()
    price_data['relative_volume'] = price_data['Volume'] / price_data['Volume'].rolling(window=w_trend).mean()
    price_data.dropna(inplace=True)
    
    if price_data.empty:
        context["status"] = "Error: Preprocessing resulted in empty data"
        context["pipeline_log"].append("Error: No data left after preprocessing (check rolling windows and data length).")
        raise ValueError("No data left after preprocessing. El periodo podría ser muy corto para las ventanas de rolling.")

    context['data']['processed_dataframe'] = price_data
    context['status'] = "Data Preprocessed"
    context["pipeline_log"].append("[AGENTE DE PREPROCESAMIENTO] Datos listos para el modelo.")
    if progress_callback: progress_callback("[AGENTE DE PREPROCESAMIENTO] Datos listos para el modelo.")
    return context

def anomaly_detection_agent(context, progress_callback=None):
    """(Agente 3) Construye, entrena y ejecuta el modelo de detección de anomalías."""
    if progress_callback: progress_callback("[AGENTE DE MODELADO] Aplicando modelo IA (Isolation Forest)...")
    
    price_data = context['data']['processed_dataframe']
    model_config = context['model_config']
    features_to_use = ['price', 'volatility', 'change_pct', 'detrended','relative_volume']
    
    # Asegurar que todas las features existen
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
    context["pipeline_log"].append(f"[AGENTE DE MODELADO] Se han detectado {len(anomalies)} anomalías.")
    if progress_callback: progress_callback(f"[AGENTE DE MODELADO] Se han detectado {len(anomalies)} anomalías.")
    return context

def reporting_agent(context, progress_callback=None):
    """(Agente 4) Prepara datos para el reporte y la visualización."""
    if progress_callback: progress_callback("[AGENTE DE REPORTE] Preparando resultados...")

    if context['status'] != "Anomalies Detected":
        context["pipeline_log"].append("El pipeline no se completó exitosamente para generar reporte.")
        # No generamos error aquí, el status ya lo indica
        return context, None, None # Devuelve context, None para df, None para fig

    anomalies = context['results']['anomalies_found']
    
    # PUNTO 6: output: date and price of the anomalies
    final_output_df = anomalies[['price']].copy()
    final_output_df.index = final_output_df.index.strftime('%Y-%m-%d')
    final_output_df.rename(columns={'price': 'Anomalous_Price_USD'}, inplace=True)
    
    context['results']['anomalies_report_df'] = final_output_df
    
    # --- Preparación para Visualización ---
    all_data = context['data']['processed_dataframe']
    ticker = context['input_params']['ticker']
    
    plt.style.use('seaborn-v0_8-whitegrid') # Estilo actualizado
    fig, ax = plt.subplots(figsize=(15, 7)) # Ajustado tamaño para Streamlit

    ax.plot(all_data.index, all_data['price'], color='royalblue', lw=1.5, label=f'Precio Histórico {ticker}')
    if not anomalies.empty:
        ax.scatter(anomalies.index, anomalies['price'], color='crimson', s=80, ec='black', lw=1.5, zorder=5, label=f'Anomalías Detectadas ({len(anomalies)})')
    
    ax.set_title(f'Detector de Anomalías de Mercado - {ticker}', fontsize=18, pad=15)
    ax.set_ylabel('Precio de Cierre Ajustado (USD)', fontsize=14)
    ax.set_xlabel('Fecha', fontsize=14)
    ax.legend(loc='upper left', fontsize=12, fancybox=True, frameon=True, shadow=True)
    ax.grid(True, which='major', linestyle='--', linewidth=0.7)
    
    # Formatear fechas en el eje X para mejor lectura
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    fig.tight_layout()
    
    context["pipeline_log"].append("[AGENTE DE REPORTE] Visualización y reporte listos.")
    if progress_callback: progress_callback("[AGENTE DE REPORTE] Visualización y reporte listos.")
    
    return context, final_output_df, fig


# --- ORQUESTADOR PRINCIPAL (Para pruebas locales, no usado por Streamlit directamente) ---
if __name__ == "__main__":
    
    # Configuración de la ejecución
    TICKER_A_ANALIZAR = "QQQ"
    PERIODO_DE_TIEMPO = "2y"
    MODEL_PARAMS_FROM_UI = {"contamination": 0.04} # Ejemplo de UI
    
    print(f"Iniciando análisis para {TICKER_A_ANALIZAR}, periodo {PERIODO_DE_TIEMPO}, contaminación {MODEL_PARAMS_FROM_UI['contamination']}")

    def simple_progress_printer(message):
        print(message)

    try:
        # 1. Iniciar el Model Context Protocol
        contexto = initialize_context(TICKER_A_ANALIZAR, PERIODO_DE_TIEMPO, MODEL_PARAMS_FROM_UI)
        
        # 2. Ejecutar la secuencia de agentes
        contexto = get_data_agent(contexto, simple_progress_printer)
        contexto = preprocess_data_agent(contexto, simple_progress_printer)
        contexto = anomaly_detection_agent(contexto, simple_progress_printer)
        
        # 3. Generar el reporte final
        contexto, df_anomalies, fig_anomalies = reporting_agent(contexto, simple_progress_printer)
        
        print("\n" + "="*50)
        print("DEMO DE DETECCIÓN DE ANOMALÍAS - RESULTADO FINAL")
        print("="*50)

        if df_anomalies is not None and not df_anomalies.empty:
            print("\n--- ANOMALÍAS DETECTADAS (Output Requerido) ---")
            print(df_anomalies.to_string())
        elif df_anomalies is not None and df_anomalies.empty:
            print("\n--- NO SE DETECTARON ANOMALÍAS ---")
        else:
            print("\n--- ERROR AL GENERAR REPORTE DE ANOMALÍAS ---")
            
        if fig_anomalies:
            print("\n[AGENTE DE REPORTE] Mostrando visualización...")
            plt.show()
        
        # (Opcional) Mostrar el objeto MCP final para demostrar el cumplimiento
        context_to_print = contexto.copy()
        if 'raw_dataframe' in context_to_print['data']:
            del context_to_print['data']['raw_dataframe'] # No imprimir dataframes largos
        if 'processed_dataframe' in context_to_print['data']:
            del context_to_print['data']['processed_dataframe']

        print("\n--- Objeto MCP Final (Trazabilidad) ---")
        print(json.dumps(context_to_print, indent=2, default=str)) # default=str para manejar datetime
        
    except Exception as e:
        print(f"\n\n--- ERROR FATAL EN EL PIPELINE ---")
        print(f"Error: {e}")
        # Si el contexto existe, imprimirlo para debug
        if 'contexto' in locals():
            print("\n--- Contexto MCP en el momento del error ---")
            context_to_print = contexto.copy()
            if 'raw_dataframe' in context_to_print['data']: del context_to_print['data']['raw_dataframe']
            if 'processed_dataframe' in context_to_print['data']: del context_to_print['data']['processed_dataframe']
            print(json.dumps(context_to_print, indent=2, default=str))