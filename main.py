import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import warnings

# Suprimir warnings no críticos (quitar si necesitas depurar)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Lista de tickers de empresas españolas
tickers = [
    "TEF.MC", "BBVA.MC", "IBE.MC", "REP.MC", "ITX.MC",
    "SAN.MC", "CLNX.MC", "TLGO.MC", "AENA.MC", "FER.MC"
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
            return None, None, None, None, None
        
        # Limpiar datos
        history = history.sort_index()
        history = history[~history.index.duplicated(keep='last')]
        # Eliminar zona horaria del índice
        history.index = history.index.tz_localize(None)
        
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
                max_date = yearly_data[yearly_data["Close"] == yearly_data["Close"].max()].index[-1]
                min_date = yearly_data[yearly_data["Close"] == yearly_data["Close"].min()].index[-1]
                max_dates[year] = max_date
                min_dates[year] = min_date
            else:
                annual_data[f"Máximo {year}"] = None
                annual_data[f"Mínimo {year}"] = None
                max_dates[year] = None
                min_dates[year] = None
        
        # Encontrar fechas del 30 de junio (o día más cercano) para proyectar máximos/mínimos
        june30_dates = {}
        for year in years:
            target_date = datetime(year, 6, 30)
            date_range = history.index - target_date
            date_range = date_range.map(lambda x: abs(x.total_seconds()))
            if not date_range.empty:
                closest_date = history.index[date_range.argmin()]
                june30_dates[year] = closest_date
            else:
                june30_dates[year] = None
        
        # Precio actual
        current_price = history["Close"][-1] if not history.empty else None
        
        # Obtener métricas fundamentales
        info = stock.info
        dividend_rate = info.get("dividendRate", None)
        dividend_yield = (dividend_rate / current_price * 100) if dividend_rate and current_price else None
        if dividend_yield and dividend_yield > 20:
            dividend_yield = None
        ebitda = info.get("ebitda", None)
        if ebitda is None and hasattr(stock, "financials"):
            try:
                ebitda = stock.financials.loc["EBITDA"].iloc[-1] if "EBITDA" in stock.financials.index else None
            except:
                ebitda = None
        fcf = info.get("freeCashflow", None)
        if fcf is None and hasattr(stock, "cashflow"):
            try:
                fcf = stock.cashflow.loc["Free Cash Flow"].iloc[-1] if "Free Cash Flow" in stock.cashflow.index else None
            except:
                fcf = None
        fundamentals = {
            "PER": round(info.get("trailingPE", None), 2) if info.get("trailingPE") else None,
            "BPA": round(info.get("trailingEps", None), 2) if info.get("trailingEps") else None,
            "Rentabilidad Dividendo (%)": round(dividend_yield, 2) if dividend_yield else None,
            "EBITDA": ebitda,
            "FCF": fcf
        }
        
        data = {
            "Ticker": ticker,
            "Nombre": info.get("longName", "N/A"),
            "Precio Actual": round(current_price, 2) if current_price else None,
            "Sector": info.get("sector", "N/A"),
            **annual_data,
            **fundamentals
        }
        return data, history, max_dates, min_dates, june30_dates
    except Exception as e:
        print(f"Error al obtener datos de {ticker}: {e}")
        return None, None, None, None, None

# Recolectar datos
results = []
all_histories = {}
for ticker in tickers:
    data, history, max_dates, min_dates, june30_dates = get_historical_prices(ticker)
    if data:
        results.append(data)
        all_histories[ticker] = (history, max_dates, min_dates, june30_dates)

# Crear DataFrame
df_results = pd.DataFrame(results)

# Guardar resultados
df_results.to_csv("spanish_stocks_summary.csv", index=False)
st.write("Resumen guardado en 'spanish_stocks_summary.csv'")

for ticker, (history, _, _, _) in all_histories.items():
    if history is not None:
        history.to_csv(f"historical_prices_{ticker}.csv")
st.write("Precios históricos guardados en archivos individuales")

# Interfaz de Streamlit
st.title("Herramienta de Inversión - Valores Españoles")

# Mostrar tabla
st.subheader("Resumen de Cotizadas Españolas")
st.dataframe(df_results)

# Seleccionar empresa
selected_ticker = st.selectbox("Seleccionar Empresa", tickers)
data, history, max_dates, min_dates, june30_dates = get_historical_prices(selected_ticker)

if data and history is not None:
    # Preparar datos para canales
    years = [2022, 2023, 2024]
    max_points = [(max_dates[year], data[f"Máximo {year}"]) for year in years if data[f"Máximo {year}"] is not None]
    min_points = [(min_dates[year], data[f"Mínimo {year}"]) for year in years if data[f"Mínimo {year}"] is not None]
    
    # Función para crear gráfico base
    def create_base_chart():
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history.index,
            y=history["Close"],
            name="Precio",
            line=dict(color="blue")
        ))
        if max_points:
            max_dates, max_values = zip(*max_points)
            fig.add_trace(go.Scatter(
                x=max_dates,
                y=max_values,
                name="Máximos Anuales",
                mode="markers",
                marker=dict(color="red", size=10)
            ))
        if min_points:
            min_dates, min_values = zip(*min_points)
            fig.add_trace(go.Scatter(
                x=min_dates,
                y=min_values,
                name="Mínimos Anuales",
                mode="markers",
                marker=dict(color="green", size=10)
            ))
        return fig
    
    # Metodología 1: Conexión Directa
    st.subheader("Metodología 1: Conexión Directa de Máximos y Mínimos")
    st.write("Este gráfico conecta los precios más altos y más bajos de cada año (2022, 2023, 2024) con líneas. Los puntos rojos son los máximos anuales, y los verdes son los mínimos. Es útil para ver cómo han cambiado los extremos cada año.")
    fig1 = create_base_chart()
    if max_points:
        max_dates, max_values = zip(*max_points)
        fig1.add_trace(go.Scatter(
            x=max_dates,
            y=max_values,
            name="Canal Superior",
            line=dict(color="red", dash="dash")
        ))
    if min_points:
        min_dates, min_values = zip(*min_points)
        fig1.add_trace(go.Scatter(
            x=min_dates,
            y=min_values,
            name="Canal Inferior",
            line=dict(color="green", dash="dash")
        ))
    fig1.update_layout(title=f"Conexión Directa - {selected_ticker} ({data['Nombre']})", xaxis_title="Fecha", yaxis_title="Precio (€)")
    st.plotly_chart(fig1)
    
    # Metodología 2: Regresión Ajustada al Máximo y Mínimo Absoluto
    st.subheader("Metodología 2: Regresión Ajustada al Máximo y Mínimo Absoluto")
    st.write("Este gráfico muestra un canal con líneas paralelas que pasan exactamente por el precio más alto y el más bajo de los últimos tres años. La pendiente sigue la tendencia general de los precios mediante una regresión lineal. Los puntos naranja y verde indican estos extremos. Es simple y muestra el rango máximo del precio.")
    fig2 = create_base_chart()
    if max_points and min_points:
        max_value = max([v for _, v in max_points])
        min_value = min([v for _, v in min_points])
        max_date = [d for d, v in max_points if v == max_value][0]
        min_date = [d for d, v in min_points if v == min_value][0]
        dates_num = [pd.Timestamp(d).timestamp() for d in history.index]
        slope, _ = np.polyfit(dates_num, history["Close"], 1)
        max_intercept = max_value - slope * pd.Timestamp(max_date).timestamp()
        min_intercept = min_value - slope * pd.Timestamp(min_date).timestamp()
        canal_dates = [history.index[0], history.index[-1]]
        canal_dates_num = [pd.Timestamp(d).timestamp() for d in canal_dates]
        max_canal = [slope * x + max_intercept for x in canal_dates_num]
        min_canal = [slope * x + min_intercept for x in canal_dates_num]
        fig2.add_trace(go.Scatter(
            x=canal_dates,
            y=max_canal,
            name="Canal Superior",
            line=dict(color="red", dash="dash")
        ))
        fig2.add_trace(go.Scatter(
            x=canal_dates,
            y=min_canal,
            name="Canal Inferior",
            line=dict(color="green", dash="dash")
        ))
        # Añadir puntos de máximo y mínimo seleccionados
        fig2.add_trace(go.Scatter(
            x=[max_date],
            y=[max_value],
            name="Máximo Seleccionado",
            mode="markers",
            marker=dict(color="orange", size=12, symbol="star")
        ))
        fig2.add_trace(go.Scatter(
            x=[min_date],
            y=[min_value],
            name="Mínimo Seleccionado",
            mode="markers",
            marker=dict(color="lime", size=12, symbol="star")
        ))
    fig2.update_layout(title=f"Regresión Máximo/Mínimo - {selected_ticker} ({data['Nombre']})", xaxis_title="Fecha", yaxis_title="Precio (€)")
    st.plotly_chart(fig2)
    
    # Metodología 3: Regresión por Dos Máximos y Dos Mínimos (Recomendada)
    st.subheader("Metodología 3: Regresión por Dos Máximos y Dos Mínimos (Recomendada)")
    st.write("Este gráfico muestra un canal con líneas paralelas que se acercan a los dos precios más altos y los dos más bajos de los últimos tres años. La pendiente sigue la tendencia general de los precios mediante una regresión lineal, por lo que las líneas no pasan exactamente por los puntos. Los puntos naranjas y verdes destacan estos extremos. Es útil para ver la tendencia y el rango de precios más representativo.")
    fig3 = create_base_chart()
    if max_points and min_points and len(max_points) >= 2 and len(min_points) >= 2:
        # Seleccionar los dos máximos más altos y los dos mínimos más bajos
        sorted_max = sorted(max_points, key=lambda x: x[1], reverse=True)[:2]
        sorted_min = sorted(min_points, key=lambda x: x[1])[:2]
        max_dates_num = [pd.Timestamp(d).timestamp() for d, _ in sorted_max]
        max_values = [v for _, v in sorted_max]
        min_dates_num = [pd.Timestamp(d).timestamp() for d, _ in sorted_min]
        min_values = [v for _, v in sorted_min]
        # Regresión sobre precios para pendiente
        dates_num = [pd.Timestamp(d).timestamp() for d in history.index]
        slope, _ = np.polyfit(dates_num, history["Close"], 1)
        # Ajustar líneas paralelas
        max_intercept = np.mean(max_values) - slope * np.mean(max_dates_num)
        min_intercept = np.mean(min_values) - slope * np.mean(min_dates_num)
        canal_dates = [history.index[0], history.index[-1]]
        canal_dates_num = [pd.Timestamp(d).timestamp() for d in canal_dates]
        max_canal = [slope * x + max_intercept for x in canal_dates_num]
        min_canal = [slope * x + min_intercept for x in canal_dates_num]
        fig3.add_trace(go.Scatter(
            x=canal_dates,
            y=max_canal,
            name="Canal Superior",
            line=dict(color="red", dash="dash")
        ))
        fig3.add_trace(go.Scatter(
            x=canal_dates,
            y=min_canal,
            name="Canal Inferior",
            line=dict(color="green", dash="dash")
        ))
        # Añadir puntos de los dos máximos y mínimos seleccionados
        fig3.add_trace(go.Scatter(
            x=[d for d, _ in sorted_max],
            y=[v for _, v in sorted_max],
            name="Máximos Seleccionados",
            mode="markers",
            marker=dict(color="orange", size=12, symbol="star")
        ))
        fig3.add_trace(go.Scatter(
            x=[d for d, _ in sorted_min],
            y=[v for _, v in sorted_min],
            name="Mínimos Seleccionados",
            mode="markers",
            marker=dict(color="lime", size=12, symbol="star")
        ))
    fig3.update_layout(title=f"Regresión Dos Máximos/Mínimos - {selected_ticker} ({data['Nombre']})", xaxis_title="Fecha", yaxis_title="Precio (€)")
    st.plotly_chart(fig3)
    
    # Metodología 4: Canal Constante por Máximo y Mínimo Absoluto
    st.subheader("Metodología 4: Canal Constante por Máximo y Mínimo Absoluto")
    st.write("Este gráfico muestra un canal con líneas horizontales que pasan exactamente por el precio más alto y el más bajo de los últimos tres años. Los puntos naranja y verde indican estos extremos. Es muy simple y muestra el rango máximo del precio sin asumir una tendencia.")
    fig4 = create_base_chart()
    if max_points and min_points:
        max_value = max([v for _, v in max_points])
        min_value = min([v for _, v in min_points])
        max_date = [d for d, v in max_points if v == max_value][0]
        min_date = [d for d, v in min_points if v == min_value][0]
        canal_dates = [history.index[0], history.index[-1]]
        max_canal = [max_value, max_value]
        min_canal = [min_value, min_value]
        fig4.add_trace(go.Scatter(
            x=canal_dates,
            y=max_canal,
            name="Canal Superior",
            line=dict(color="red", dash="dash")
        ))
        fig4.add_trace(go.Scatter(
            x=canal_dates,
            y=min_canal,
            name="Canal Inferior",
            line=dict(color="green", dash="dash")
        ))
        # Añadir puntos de máximo y mínimo seleccionados
        fig4.add_trace(go.Scatter(
            x=[max_date],
            y=[max_value],
            name="Máximo Seleccionado",
            mode="markers",
            marker=dict(color="orange", size=12, symbol="star")
        ))
        fig4.add_trace(go.Scatter(
            x=[min_date],
            y=[min_value],
            name="Mínimo Seleccionado",
            mode="markers",
            marker=dict(color="lime", size=12, symbol="star")
        ))
    fig4.update_layout(title=f"Canal Constante Máximo/Mínimo - {selected_ticker} ({data['Nombre']})", xaxis_title="Fecha", yaxis_title="Precio (€)")
    st.plotly_chart(fig4)
    
    # Metodología 5: Regresión Proyectada al 30 de Junio
    st.subheader("Metodología 5: Regresión Proyectada al 30 de Junio")
    st.write("Este gráfico muestra un canal con líneas paralelas que usan el precio más alto y el más bajo de los últimos tres años, dibujados en el 30 de junio de cada año para mantener fechas consistentes. La pendiente sigue la tendencia general de los precios mediante una regresión lineal. Los puntos naranjas y verdes muestran estos extremos proyectados. Es útil para ver el rango máximo con fechas fijas.")
    fig5 = create_base_chart()
    if max_points and min_points:
        max_value = max([v for _, v in max_points])
        min_value = min([v for _, v in min_points])
        max_year = [d.year for d, v in max_points if v == max_value][0]
        min_year = [d.year for d, v in min_points if v == min_value][0]
        # Proyectar a las fechas del 30 de junio
        june30_max_points = [(june30_dates[year], max_value if year == max_year else data[f"Máximo {year}"]) for year in years if june30_dates[year] is not None and data[f"Máximo {year}"] is not None]
        june30_min_points = [(june30_dates[year], min_value if year == min_year else data[f"Mínimo {year}"]) for year in years if june30_dates[year] is not None and data[f"Mínimo {year}"] is not None]
        if june30_max_points and june30_min_points:
            max_date = [d for d, v in june30_max_points if v == max_value][0]
            min_date = [d for d, v in june30_min_points if v == min_value][0]
            dates_num = [pd.Timestamp(d).timestamp() for d in history.index]
            slope, _ = np.polyfit(dates_num, history["Close"], 1)
            max_intercept = max_value - slope * pd.Timestamp(max_date).timestamp()
            min_intercept = min_value - slope * pd.Timestamp(min_date).timestamp()
            canal_dates = [history.index[0], history.index[-1]]
            canal_dates_num = [pd.Timestamp(d).timestamp() for d in canal_dates]
            max_canal = [slope * x + max_intercept for x in canal_dates_num]
            min_canal = [slope * x + min_intercept for x in canal_dates_num]
            fig5.add_trace(go.Scatter(
                x=canal_dates,
                y=max_canal,
                name="Canal Superior",
                line=dict(color="red", dash="dash")
            ))
            fig5.add_trace(go.Scatter(
                x=canal_dates,
                y=min_canal,
                name="Canal Inferior",
                line=dict(color="green", dash="dash")
            ))
            # Añadir puntos proyectados al 30 de junio
            fig5.add_trace(go.Scatter(
                x=[d for d, _ in june30_max_points],
                y=[v for _, v in june30_max_points],
                name="Máximos Proyectados",
                mode="markers",
                marker=dict(color="orange", size=12, symbol="star")
            ))
            fig5.add_trace(go.Scatter(
                x=[d for d, _ in june30_min_points],
                y=[v for _, v in june30_min_points],
                name="Mínimos Proyectados",
                mode="markers",
                marker=dict(color="lime", size=12, symbol="star")
            ))
    fig5.update_layout(title=f"30 de Junio Proyectado - {selected_ticker} ({data['Nombre']})", xaxis_title="Fecha", yaxis_title="Precio (€)")
    st.plotly_chart(fig5)
    
    # Metodología 6: Canal Paralelo por Dos Máximos y Dos Mínimos
    st.subheader("Metodología 6: Canal Paralelo por Dos Máximos y Dos Mínimos")
    st.write("Este gráfico muestra un canal con líneas paralelas donde la línea superior pasa exactamente por los dos precios más altos y la línea inferior por los dos más bajos de los últimos tres años. Los puntos naranjas y verdes destacan estos extremos. Es útil para ver el rango de precios más significativo con precisión.")
    fig6 = create_base_chart()
    if max_points and min_points and len(max_points) >= 2 and len(min_points) >= 2:
        # Seleccionar los dos máximos más altos y los dos mínimos más bajos
        sorted_max = sorted(max_points, key=lambda x: x[1], reverse=True)[:2]
        sorted_min = sorted(min_points, key=lambda x: x[1])[:2]
        # Combinar puntos para calcular pendiente óptima
        all_points = sorted_max + sorted_min
        dates_num = [pd.Timestamp(d).timestamp() for d, _ in all_points]
        values = [v for _, v in all_points]
        # Regresión sobre los cuatro puntos
        slope, intercept = np.polyfit(dates_num, values, 1)
        # Ajustar líneas para que pasen exactamente por los máximos y mínimos
        max_intercept = np.max([v - slope * pd.Timestamp(d).timestamp() for d, v in sorted_max])
        min_intercept = np.min([v - slope * pd.Timestamp(d).timestamp() for d, v in sorted_min])
        canal_dates = [history.index[0], history.index[-1]]
        canal_dates_num = [pd.Timestamp(d).timestamp() for d in canal_dates]
        max_canal = [slope * x + max_intercept for x in canal_dates_num]
        min_canal = [slope * x + min_intercept for x in canal_dates_num]
        fig6.add_trace(go.Scatter(
            x=canal_dates,
            y=max_canal,
            name="Canal Superior",
            line=dict(color="red", dash="dash")
        ))
        fig6.add_trace(go.Scatter(
            x=canal_dates,
            y=min_canal,
            name="Canal Inferior",
            line=dict(color="green", dash="dash")
        ))
        # Añadir puntos de los dos máximos y mínimos seleccionados
        fig6.add_trace(go.Scatter(
            x=[d for d, _ in sorted_max],
            y=[v for _, v in sorted_max],
            name="Máximos Seleccionados",
            mode="markers",
            marker=dict(color="orange", size=12, symbol="star")
        ))
        fig6.add_trace(go.Scatter(
            x=[d for d, _ in sorted_min],
            y=[v for _, v in sorted_min],
            name="Mínimos Seleccionados",
            mode="markers",
            marker=dict(color="lime", size=12, symbol="star")
        ))
    fig6.update_layout(title=f"Dos Máximos/Dos Mínimos - {selected_ticker} ({data['Nombre']})", xaxis_title="Fecha", yaxis_title="Precio (€)")
    st.plotly_chart(fig6)
    
    # Posición relativa
    if max_points and min_points:
        latest_max = max_points[-1][1]
        latest_min = min_points[-1][1]
        current_price = data["Precio Actual"]
        relative_position = (current_price - latest_min) / (latest_max - latest_min) * 100 if latest_max != latest_min else 0
        st.write(f"Posición relativa del precio actual (2024): {relative_position:.2f}% (0% = Mínimo, 100% = Máximo)")
    
    # Métricas fundamentales
    st.subheader("Métricas Fundamentales")
    st.write(f"PER: {data['PER'] if data['PER'] is not None else 'N/A'}")
    st.write(f"BPA: {data['BPA'] if data['BPA'] is not None else 'N/A'}")
    st.write(f"Rentabilidad Dividendo: {data['Rentabilidad Dividendo (%)'] if data['Rentabilidad Dividendo (%)'] is not None else 'N/A'}%")
    st.write(f"EBITDA: {data['EBITDA'] if data['EBITDA'] is not None else 'N/A'}")
    st.write(f"FCF: {data['FCF'] if data['FCF'] is not None else 'N/A'}")
else:
    st.error(f"No hay datos disponibles para {selected_ticker}")