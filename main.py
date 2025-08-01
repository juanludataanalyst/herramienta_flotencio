import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings

# Suprimir warnings no críticos de pandas/yfinance (quitar si necesitas depurar el warning específico)
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

# Función para obtener precios históricos y calcular máximos/mínimos anuales
@st.cache_data
def get_historical_prices(ticker):
    try:
        stock = yf.Ticker(ticker)
        # Obtener datos desde 2019 hasta fin de 2023
        start_date = datetime(2019, 1, 1)
        end_date = datetime(2023, 12, 31)
        history = stock.history(start=start_date, end=end_date)
        
        if history.empty:
            print(f"No hay datos para {ticker}")
            return None, None
        
        # Limpiar datos: ordenar índice y eliminar duplicados
        history = history.sort_index()
        history = history[~history.index.duplicated(keep='last')]
        
        # Años a procesar
        years = [2019, 2020, 2021, 2022, 2023]
        
        # Calcular máximos y mínimos por año
        annual_data = {}
        for year in years:
            yearly_data = history[history.index.year == year]
            if not yearly_data.empty:
                annual_data[f"Máximo {year}"] = yearly_data["Close"].max()
                annual_data[f"Mínimo {year}"] = yearly_data["Close"].min()
            else:
                annual_data[f"Máximo {year}"] = None
                annual_data[f"Mínimo {year}"] = None
        
        # Precio actual (último cierre disponible en 2023)
        current_price = history["Close"][-1] if not history.empty else None
        
        # Crear diccionario con resultados
        data = {
            "Ticker": ticker,
            "Nombre": stock.info.get("longName", "N/A"),
            "Precio Actual": current_price,
            "Sector": stock.info.get("sector", "N/A"),
            **annual_data
        }
        return data, history
    except Exception as e:
        print(f"Error al obtener datos de {ticker}: {e}")
        return None, None

# Recolectar datos para todos los tickers
results = []
all_histories = {}
for ticker in tickers:
    data, history = get_historical_prices(ticker)
    if data:
        results.append(data)
        all_histories[ticker] = history

# Crear DataFrame con resultados
df_results = pd.DataFrame(results)

# Guardar resultados en CSV
df_results.to_csv("spanish_stocks_summary.csv", index=False)
st.write("Resumen guardado en 'spanish_stocks_summary.csv'")

# Guardar históricos de precios en un CSV separado
for ticker, history in all_histories.items():
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
data, history = get_historical_prices(selected_ticker)

if data and history is not None:
    # Preparar datos para el canal (máximos y mínimos conectados)
    years = [2019, 2020, 2021, 2022, 2023]
    max_points = []
    min_points = []
    for year in years:
        if data[f"Máximo {year}"] is not None:
            # Usar la última fecha del año para el punto
            year_data = history[history.index.year == year]
            max_date = year_data[year_data["Close"] == data[f"Máximo {year}"]].index[-1]
            max_points.append((max_date, data[f"Máximo {year}"]))
        if data[f"Mínimo {year}"] is not None:
            min_date = year_data[year_data["Close"] == data[f"Mínimo {year}"]].index[-1]
            min_points.append((min_date, data[f"Mínimo {year}"]))

    # Crear gráfico
    fig = go.Figure()
    
    # Precios históricos
    fig.add_trace(go.Scatter(
        x=history.index,
        y=history["Close"],
        name="Precio",
        line=dict(color="blue")
    ))
    
    # Máximos anuales conectados
    if max_points:
        max_dates, max_values = zip(*max_points)
        fig.add_trace(go.Scatter(
            x=max_dates,
            y=max_values,
            name="Máximos Anuales",
            line=dict(color="red", dash="dash"),
            mode="lines+markers"
        ))
    
    # Mínimos anuales conectados
    if min_points:
        min_dates, min_values = zip(*min_points)
        fig.add_trace(go.Scatter(
            x=min_dates,
            y=min_values,
            name="Mínimos Anuales",
            line=dict(color="green", dash="dash"),
            mode="lines+markers"
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
        st.write(f"Posición relativa del precio actual (2023): {relative_position:.2f}% (0% = Mínimo, 100% = Máximo)")
else:
    st.error(f"No hay datos disponibles para {selected_ticker}")