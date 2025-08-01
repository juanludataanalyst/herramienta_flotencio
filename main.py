import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import warnings

# Suprimir warnings no críticos de pandas/yfinance (quitar si necesitas depurar)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Lista de tickers de empresas españolas (IBEX 35 y Mercado Continuo)
tickers = [
    "TEF.MC",  # Telefónica
    "BBVA.MC", # BBVA
    "IBE.MC",  # Iberdrola
    "REP.MC",  # Repsol
    "ITX.MC",  # Inditex
    "SAN.MC",  # Banco Santander
    "CLNX.MC", # Cellnex Telecom
    "TLGO.MC", # Talgo
    "AENA.MC", # Aena
    "FER.MC"   # Ferrovial
]

# Función para obtener precios históricos y métricas fundamentales
@st.cache_data
def get_historical_prices(ticker):
    try:
        stock = yf.Ticker(ticker)
        # Obtener datos desde 2022 hasta fin de 2024
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2024, 12, 31)
        history = stock.history(start=start_date, end=end_date)
        
        if history.empty:
            print(f"No hay datos para {ticker}")
            return None, None, None, None
        
        # Limpiar datos: ordenar índice y eliminar duplicados
        history = history.sort_index()
        history = history[~history.index.duplicated(keep='last')]
        
        # Años a procesar
        years = [2022, 2023, 2024]
        
        # Calcular máximos y mínimos por año
        annual_data = {}
        max_dates = {}
        min_dates = {}
        for year in years:
            yearly_data = history[history.index.year == year]
            if not yearly_data.empty:
                annual_data[f"Máximo {year}"] = yearly_data["Close"].max()
                annual_data[f"Mínimo {year}"] = yearly_data["Close"].min()
                # Guardar fechas de máximos y mínimos
                max_date = yearly_data[yearly_data["Close"] == yearly_data["Close"].max()].index[-1]
                min_date = yearly_data[yearly_data["Close"] == yearly_data["Close"].min()].index[-1]
                max_dates[year] = max_date
                min_dates[year] = min_date
            else:
                annual_data[f"Máximo {year}"] = None
                annual_data[f"Mínimo {year}"] = None
                max_dates[year] = None
                min_dates[year] = None
        
        # Precio actual (último cierre disponible)
        current_price = history["Close"][-1] if not history.empty else None
        
        # Obtener métricas fundamentales
        info = stock.info
        fundamentals = {
            "PER": info.get("trailingPE", None),
            "BPA": info.get("trailingEps", None),
            "Rentabilidad Dividendo (%)": info.get("dividendYield", 0) * 100,
            "EBITDA": info.get("ebitda", None),
            "FCF": info.get("freeCashflow", None)
        }
        
        # Crear diccionario con resultados
        data = {
            "Ticker": ticker,
            "Nombre": stock.info.get("longName", "N/A"),
            "Precio Actual": current_price,
            "Sector": stock.info.get("sector", "N/A"),
            **annual_data,
            **fundamentals
        }
        return data, history, max_dates, min_dates
    except Exception as e:
        print(f"Error al obtener datos de {ticker}: {e}")
        return None, None, None, None

# Recolectar datos para todos los tickers
results = []
all_histories = {}
for ticker in tickers:
    data, history, max_dates, min_dates = get_historical_prices(ticker)
    if data:
        results.append(data)
        all_histories[ticker] = (history, max_dates, min_dates)

# Crear DataFrame con resultados
df_results = pd.DataFrame(results)

# Guardar resultados en CSV
df_results.to_csv("spanish_stocks_summary.csv", index=False)
st.write("Resumen guardado en 'spanish_stocks_summary.csv'")

# Guardar históricos de precios en un CSV separado
for ticker, (history, _, _) in all_histories.items():
    if history is not None:
        history.to_csv(f"historical_prices_{ticker}.csv")
st.write("Precios históricos guardados en archivos individuales")

# Interfaz de Streamlit
st.title("Herramienta de Inversión - Valores Españoles")

# Mostrar tabla con resumen
st.subheader("Resumen de Cotizadas Españolas")
st.dataframe(df_results)

# Seleccionar empresa para visualización
selected_ticker = st.selectbox("Seleccionar Empresa", tickers)
data, history, max_dates, min_dates = get_historical_prices(selected_ticker)

if data and history is not None:
    # Preparar datos para el canal (máximos y mínimos)
    years = [2022, 2023, 2024]
    max_points = [(max_dates[year], data[f"Máximo {year}"]) for year in years if data[f"Máximo {year}"] is not None]
    min_points = [(min_dates[year], data[f"Mínimo {year}"]) for year in years if data[f"Mínimo {year}"] is not None]
    
    # Calcular líneas de tendencia para el canal
    if max_points and min_points:
        # Convertir fechas a valores numéricos para regresión
        max_dates_num = [pd.Timestamp(date).timestamp() for date, _ in max_points]
        max_values = [value for _, value in max_points]
        min_dates_num = [pd.Timestamp(date).timestamp() for date, _ in min_points]
        min_values = [value for _, value in min_points]
        
        # Ajustar regresión lineal
        max_slope, max_intercept = np.polyfit(max_dates_num, max_values, 1)
        min_slope, min_intercept = np.polyfit(min_dates_num, min_values, 1)
        
        # Usar la pendiente promedio para un canal paralelo
        avg_slope = (max_slope + min_slope) / 2
        
        # Recalcular interceptos para mantener el canal paralelo
        max_intercept = np.mean(max_values) - avg_slope * np.mean(max_dates_num)
        min_intercept = np.mean(min_values) - avg_slope * np.mean(min_dates_num)
        
        # Generar puntos para las líneas del canal
        canal_dates = [history.index[0], history.index[-1]]
        canal_dates_num = [pd.Timestamp(d).timestamp() for d in canal_dates]
        max_canal = [avg_slope * x + max_intercept for x in canal_dates_num]
        min_canal = [avg_slope * x + min_intercept for x in canal_dates_num]
    
    # Crear gráfico
    fig = go.Figure()
    
    # Precios históricos
    fig.add_trace(go.Scatter(
        x=history.index,
        y=history["Close"],
        name="Precio",
        line=dict(color="blue")
    ))
    
    # Puntos de máximos anuales
    if max_points:
        max_dates, max_values = zip(*max_points)
        fig.add_trace(go.Scatter(
            x=max_dates,
            y=max_values,
            name="Máximos Anuales",
            mode="markers",
            marker=dict(color="red", size=10)
        ))
    
    # Puntos de mínimos anuales
    if min_points:
        min_dates, min_values = zip(*min_points)
        fig.add_trace(go.Scatter(
            x=min_dates,
            y=min_values,
            name="Mínimos Anuales",
            mode="markers",
            marker=dict(color="green", size=10)
        ))
    
    # Líneas del canal (tendencia)
    if max_points and min_points:
        fig.add_trace(go.Scatter(
            x=canal_dates,
            y=max_canal,
            name="Canal Superior",
            line=dict(color="red", dash="dash")
        ))
        fig.add_trace(go.Scatter(
            x=canal_dates,
            y=min_canal,
            name="Canal Inferior",
            line=dict(color="green", dash="dash")
        ))
    
    # Configurar diseño
    fig.update_layout(
        title=f"Canal de Precios - {selected_ticker} ({data['Nombre']})",
        xaxis_title="Fecha",
        yaxis_title="Precio (€)",
        showlegend=True
    )
    
    # Mostrar gráfico
    st.plotly_chart(fig)
    
    # Mostrar posición relativa respecto al canal más reciente
    if max_points and min_points:
        latest_max = max_points[-1][1]
        latest_min = min_points[-1][1]
        current_price = data["Precio Actual"]
        relative_position = (current_price - latest_min) / (latest_max - latest_min) * 100 if latest_max != latest_min else 0
        st.write(f"Posición relativa del precio actual (2024): {relative_position:.2f}% (0% = Mínimo, 100% = Máximo)")
    
    # Mostrar métricas fundamentales
    st.subheader("Métricas Fundamentales")
    st.write(f"PER: {data['PER'] if data['PER'] is not None else 'N/A'}")
    st.write(f"BPA: {data['BPA'] if data['BPA'] is not None else 'N/A'}")
    st.write(f"Rentabilidad Dividendo: {data['Rentabilidad Dividendo (%)']:.2f}%")
    st.write(f"EBITDA: {data['EBITDA'] if data['EBITDA'] is not None else 'N/A'}")
    st.write(f"FCF: {data['FCF'] if data['FCF'] is not None else 'N/A'}")
else:
    st.error(f"No hay datos disponibles para {selected_ticker}")