import streamlit as st
import pandas as pd
from datetime import datetime
import json

# Importar la lógica del detector de anomalías
import anomaly_detector_logic as adl

# --- Configuración de la Página de Streamlit ---
st.set_page_config(page_title="Detector de Anomalías de Mercado", layout="wide")

st.title("📈 Detector de Anomalías de Mercado")
st.markdown("""
Esta aplicación utiliza un modelo de **Isolation Forest** para detectar anomalías en los precios de cierre ajustados de ETFs.
Los datos se obtienen de Yahoo Finance. Se implementa una arquitectura de agentes y un "Model Context Protocol" (MCP)
para la trazabilidad del proceso.
""")

# --- Panel Lateral de Entradas (Sidebar) ---
st.sidebar.header("Parámetros de Análisis")

# Ticker del ETF
ticker_default = "QQQ" # Nasdaq 100 ETF
etf_options = ["QQQ", "SPY", "DIA", "IWM", "GLD", "SLV", "AAPL", "MSFT", "GOOGL", "TSLA"] # Algunas opciones comunes
selected_ticker = st.sidebar.selectbox(
    "Seleccione el Ticker del ETF/Acción:",
    options=etf_options,
    index=etf_options.index(ticker_default),
    help="Ejemplos: QQQ (Nasdaq 100), SPY (S&P 500), AAPL (Apple Inc.)"
)
custom_ticker = st.sidebar.text_input("O ingrese un Ticker personalizado (ej. VOO):", "")
ticker_to_analyze = custom_ticker if custom_ticker else selected_ticker


# Periodo de tiempo
period_options = {
    "Último año": "1y",
    "Últimos 2 años": "2y",
    "Últimos 5 años": "5y",
    "Últimos 10 años": "10y",
    "Máximo disponible": "max"
}
selected_period_label = st.sidebar.selectbox(
    "Seleccione el Periodo de Tiempo:",
    options=list(period_options.keys()),
    index=1 # Default a "Últimos 2 años"
)
period_to_analyze = period_options[selected_period_label]

# Parámetros del Modelo (Contaminación)
contamination_rate = st.sidebar.slider(
    "Sensibilidad del Modelo (Contaminación):",
    min_value=0.005,
    max_value=0.10,
    value=0.02, # Valor por defecto igual al BEST_MODEL_CONFIG
    step=0.005,
    help="Proporción esperada de anomalías en los datos. Valores más altos detectan más anomalías."
)

model_params_from_ui = {"contamination": contamination_rate}

# Botón para iniciar el análisis
run_button = st.sidebar.button("🚀 Detectar Anomalías")

# --- Área Principal para Resultados ---
if run_button:
    st.header(f"Resultados para {ticker_to_analyze} ({selected_period_label})")

    # Placeholder para mensajes de progreso
    progress_area = st.empty()
    def streamlit_progress_callback(message):
        progress_area.info(message)

    try:
        # 1. Iniciar el Model Context Protocol
        streamlit_progress_callback("Inicializando contexto...")
        contexto = adl.initialize_context(ticker_to_analyze, period_to_analyze, model_params_from_ui)
        
        # Cache para la descarga de datos
        # Nota: yf.download puede ser lento, idealmente se cachea si los parámetros son los mismos.
        # Streamlit tiene @st.cache_data para esto. Para simplificar ahora, lo llamamos directo.
        # Si se usa @st.cache_data, la función 'get_data_agent' debería ser "pura" y retornar solo los datos.
        # Aquí, como el agente modifica 'contexto', no es directamente cacheable de esa forma.
        # Una alternativa es cachear la llamada a yf.download dentro del agente.

        # 2. Ejecutar la secuencia de agentes
        with st.spinner("Procesando... Esto puede tardar unos segundos."):
            contexto = adl.get_data_agent(contexto, streamlit_progress_callback)
            contexto = adl.preprocess_data_agent(contexto, streamlit_progress_callback)
            contexto = adl.anomaly_detection_agent(contexto, streamlit_progress_callback)
            
            # 3. Generar el reporte final
            contexto, df_anomalies, fig_anomalies = adl.reporting_agent(contexto, streamlit_progress_callback)

        progress_area.success("¡Análisis completado!")

        # Mostrar resultados
        st.subheader("🗓️ Anomalías Detectadas")
        if df_anomalies is not None and not df_anomalies.empty:
            st.dataframe(df_anomalies)
            st.markdown(f"Se encontraron **{len(df_anomalies)}** anomalías.")
        elif df_anomalies is not None and df_anomalies.empty:
            st.info("✅ No se detectaron anomalías con los parámetros actuales.")
        else:
            st.warning("No se pudo generar el reporte de anomalías (verifique el log del pipeline).")


        st.subheader("📊 Visualización de Precios y Anomalías")
        if fig_anomalies:
            st.pyplot(fig_anomalies)
        else:
            st.warning("No se pudo generar la visualización.")

        # Mostrar el objeto MCP final
        with st.expander("🔍 Ver Detalles del Proceso (Model Context Protocol - MCP)"):
            # Limpiar dataframes del contexto para no mostrar datos masivos en JSON
            context_to_display = contexto.copy()
            if 'data' in context_to_display and isinstance(context_to_display['data'], dict):
                context_to_display['data'].pop('raw_dataframe', None)
                context_to_display['data'].pop('processed_dataframe', None)
            
            # ---------- INICIO DE LA CORRECCIÓN ----------
            # Definir la función convertidora ANTES de usarla
            def datetime_converter_for_json(o): # Renombré ligeramente para evitar confusión con un posible nombre de variable
                if isinstance(o, datetime):
                    return o.isoformat() # Usar isoformat() es más estándar para JSON
                # Puedes añadir más conversiones aquí si es necesario para otros tipos no serializables
                # raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

            try:
                # Pasar la REFERENCIA a la función (su nombre)
                json_output_mcp = json.dumps(context_to_display, indent=2, default=datetime_converter_for_json)
                st.json(json_output_mcp)
            except TypeError as te:
                st.error(f"Error al serializar el MCP a JSON: {te}")
                st.text("Puede haber tipos de datos no manejados en el contexto.")
                # Imprimir una versión más simple del contexto si falla la serialización completa
                st.text(str(context_to_display))
            # ---------- FIN DE LA CORRECCIÓN ----------

    except ValueError as ve:
        st.error(f"Error de Valor durante el proceso: {ve}")
        if 'contexto' in locals(): # Si el contexto se inicializó
             with st.expander("Detalles del MCP en el momento del error"):
                st.json(json.dumps(contexto, indent=2, default=str)) # default=str para datetime
    except Exception as e:
        st.error(f"Ocurrió un error inesperado: {e}")
        st.error("Revise los parámetros o intente con otro Ticker/Periodo.")
        if 'contexto' in locals():
            with st.expander("Detalles del MCP en el momento del error"):
                st.json(json.dumps(contexto, indent=2, default=str))

else:
    st.info("Configure los parámetros en el panel lateral y presione 'Detectar Anomalías'.")

st.sidebar.markdown("---")
st.sidebar.markdown("Desarrollado por Dylan Saenz") 
st.sidebar.markdown("https://www.linkedin.com/in/dylan-nicol%C3%A1s-s%C3%A1enz-chavarro-a087a218b/") # Opcional
