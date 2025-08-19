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
    
    # Selector de metodología
    methodology = st.selectbox("Seleccionar Metodología de Canal", 
                              ["Metodología 1: Regresión de 3 Puntos", 
                               "Metodología 2: Canal Cronológico Adaptativo",
                               "Metodología 3: Canal Adaptativo que Contiene Todos los Precios"])
    
    if methodology == "Metodología 1: Regresión de 3 Puntos":
        st.subheader("Metodología 1: Canal de Regresión de 3 Puntos")
        st.write("Esta metodología utiliza una regresión lineal sobre los 3 máximos y 3 mínimos anuales para crear un canal de precios. El canal muestra la tendencia y rango de precios más representativo.")
    elif methodology == "Metodología 2: Canal Cronológico Adaptativo":
        st.subheader("Metodología 2: Canal Cronológico Adaptativo")
        st.write("Esta metodología conecta cronológicamente los puntos de máximos y mínimos usando sus fechas reales. Comienza uniendo el primer año (2022) con el segundo (2023). Si el tercer año (2024) queda fuera de esas líneas, entonces conecta directamente el primer año con el tercero.")
    else:
        st.subheader("Metodología 3: Canal Adaptativo que Contiene Todos los Precios")
        st.write("Esta metodología comienza uniendo los dos máximos y mínimos más extremos proyectados al 30 de junio, pero expande automáticamente el canal incluyendo más puntos hasta que todos los precios históricos estén contenidos dentro del canal.")
    
    def calculate_3_point_regression_channel():
        if len(max_points) == 3 and len(min_points) == 3:
            # Usar todos los puntos de máximos y mínimos
            all_points = max_points + min_points
            dates_num = [pd.Timestamp(d).timestamp() for d, _ in all_points]
            values = [v for _, v in all_points]
            
            # Regresión lineal sobre los 6 puntos
            slope, intercept = np.polyfit(dates_num, values, 1)
            
            # Calcular límites del canal basados en residuos
            predicted_values = [slope * x + intercept for x in dates_num]
            residuals = [v - p for v, p in zip(values, predicted_values)]
            
            # Separar residuos de máximos y mínimos
            max_residuals = residuals[:3]
            min_residuals = residuals[3:]
            
            # Calcular offset para canal superior e inferior
            max_offset = np.mean(max_residuals)
            min_offset = np.mean(min_residuals)
            
            # Crear líneas del canal
            canal_dates = [history.index[0], history.index[-1]]
            canal_dates_num = [pd.Timestamp(d).timestamp() for d in canal_dates]
            
            max_canal = [slope * x + intercept + max_offset for x in canal_dates_num]
            min_canal = [slope * x + intercept + min_offset for x in canal_dates_num]
            
            return max_canal, min_canal, canal_dates, slope, intercept, max_offset, min_offset
        return None, None, None, None, None, None, None
    
    def calculate_chronological_adaptive_channel():
        if len(max_points) == 3 and len(min_points) == 3:
            # Usar fechas reales ordenadas cronológicamente
            years = [2022, 2023, 2024]
            chronological_max_points = []
            chronological_min_points = []
            
            # Ordenar puntos cronológicamente por año usando fechas reales
            for year in years:
                # Encontrar el máximo y mínimo de este año específico
                year_max_point = next(((d, v) for d, v in max_points if d.year == year), None)
                year_min_point = next(((d, v) for d, v in min_points if d.year == year), None)
                
                if year_max_point is not None:
                    chronological_max_points.append(year_max_point)
                if year_min_point is not None:
                    chronological_min_points.append(year_min_point)
            
            if len(chronological_max_points) >= 3 and len(chronological_min_points) >= 3:
                def test_line_contains_point(point1, point2, test_point):
                    """Verifica si el test_point está dentro de la línea formada por point1 y point2"""
                    x1, y1 = pd.Timestamp(point1[0]).timestamp(), point1[1]
                    x2, y2 = pd.Timestamp(point2[0]).timestamp(), point2[1]
                    x3, y3 = pd.Timestamp(test_point[0]).timestamp(), test_point[1]
                    
                    # Calcular el valor esperado en x3 según la línea point1-point2
                    if x2 != x1:
                        slope = (y2 - y1) / (x2 - x1)
                        expected_y = y1 + slope * (x3 - x1)
                        return y3 <= expected_y if point1 == chronological_max_points[0] else y3 >= expected_y
                    return True
                
                # Para máximos: probar línea año1-año2, verificar si año3 está debajo
                max_year1, max_year2, max_year3 = chronological_max_points[0], chronological_max_points[1], chronological_max_points[2]
                
                # Para mínimos: probar línea año1-año2, verificar si año3 está arriba  
                min_year1, min_year2, min_year3 = chronological_min_points[0], chronological_min_points[1], chronological_min_points[2]
                
                # Decisión para máximos
                if test_line_contains_point(max_year1, max_year2, max_year3):
                    # Usar año1-año2 para máximos
                    max_connection = [max_year1, max_year2]
                    max_connection_type = "1-2"
                else:
                    # Usar año1-año3 para máximos
                    max_connection = [max_year1, max_year3]
                    max_connection_type = "1-3"
                
                # Decisión para mínimos
                if test_line_contains_point(min_year1, min_year2, min_year3):
                    # Usar año1-año2 para mínimos
                    min_connection = [min_year1, min_year2]
                    min_connection_type = "1-2"
                else:
                    # Usar año1-año3 para mínimos
                    min_connection = [min_year1, min_year3]
                    min_connection_type = "1-3"
                
                # Calcular pendientes y crear canal
                max_date_diff = (pd.Timestamp(max_connection[1][0]) - pd.Timestamp(max_connection[0][0])).total_seconds()
                max_slope = (max_connection[1][1] - max_connection[0][1]) / max_date_diff if max_date_diff != 0 else 0
                max_intercept = max_connection[0][1] - max_slope * pd.Timestamp(max_connection[0][0]).timestamp()
                
                min_date_diff = (pd.Timestamp(min_connection[1][0]) - pd.Timestamp(min_connection[0][0])).total_seconds()
                min_slope = (min_connection[1][1] - min_connection[0][1]) / min_date_diff if min_date_diff != 0 else 0
                min_intercept = min_connection[0][1] - min_slope * pd.Timestamp(min_connection[0][0]).timestamp()
                
                # Crear líneas del canal extendidas
                canal_dates = [history.index[0], history.index[-1]]
                canal_dates_num = [pd.Timestamp(d).timestamp() for d in canal_dates]
                
                max_canal = [max_slope * x + max_intercept for x in canal_dates_num]
                min_canal = [min_slope * x + min_intercept for x in canal_dates_num]
                
                # Verificar que las líneas no se crucen
                if max_canal[-1] <= min_canal[-1]:
                    # Si se cruzan, hacer las líneas paralelas
                    initial_distance = max_canal[0] - min_canal[0]
                    if initial_distance <= 0:
                        initial_distance = abs(max_connection[0][1] - min_connection[0][1])
                    
                    avg_slope = (max_slope + min_slope) / 2
                    max_intercept = max_connection[0][1] - avg_slope * pd.Timestamp(max_connection[0][0]).timestamp()
                    min_intercept = max_intercept - initial_distance
                    
                    max_canal = [avg_slope * x + max_intercept for x in canal_dates_num]
                    min_canal = [avg_slope * x + min_intercept for x in canal_dates_num]
                    max_slope = min_slope = avg_slope
                
                return max_canal, min_canal, canal_dates, max_connection, min_connection, max_slope, min_slope, max_connection_type, min_connection_type
        
        return None, None, None, None, None, None, None, None, None
    
    def calculate_adaptive_channel_with_projection():
        if len(max_points) >= 2 and len(min_points) >= 2:
            # Proyectar todos los puntos al 30 de junio
            projected_max_points = []
            projected_min_points = []
            
            for date, value in max_points:
                year = date.year
                if year in june30_dates and june30_dates[year] is not None:
                    projected_max_points.append((june30_dates[year], value))
            
            for date, value in min_points:
                year = date.year
                if year in june30_dates and june30_dates[year] is not None:
                    projected_min_points.append((june30_dates[year], value))
            
            # Ordenar por fecha
            projected_max_points = sorted(projected_max_points, key=lambda x: x[0])
            projected_min_points = sorted(projected_min_points, key=lambda x: x[0])
            
            if len(projected_max_points) >= 2 and len(projected_min_points) >= 2:
                # Ordenar por valor para encontrar extremos
                sorted_max_by_value = sorted(projected_max_points, key=lambda x: x[1], reverse=True)
                sorted_min_by_value = sorted(projected_min_points, key=lambda x: x[1])
                
                def create_channel_from_points(max_points_used, min_points_used):
                    """Crea un canal desde los puntos dados"""
                    if len(max_points_used) < 2 or len(min_points_used) < 2:
                        return None, None, None, None
                    
                    # Ordenar por fecha para crear líneas
                    max_by_date = sorted(max_points_used, key=lambda x: x[0])
                    min_by_date = sorted(min_points_used, key=lambda x: x[0])
                    
                    # Calcular pendientes
                    max_date_diff = (pd.Timestamp(max_by_date[-1][0]) - pd.Timestamp(max_by_date[0][0])).total_seconds()
                    max_slope = (max_by_date[-1][1] - max_by_date[0][1]) / max_date_diff if max_date_diff != 0 else 0
                    max_intercept = max_by_date[0][1] - max_slope * pd.Timestamp(max_by_date[0][0]).timestamp()
                    
                    min_date_diff = (pd.Timestamp(min_by_date[-1][0]) - pd.Timestamp(min_by_date[0][0])).total_seconds()
                    min_slope = (min_by_date[-1][1] - min_by_date[0][1]) / min_date_diff if min_date_diff != 0 else 0
                    min_intercept = min_by_date[0][1] - min_slope * pd.Timestamp(min_by_date[0][0]).timestamp()
                    
                    # Crear líneas del canal
                    canal_dates = [history.index[0], history.index[-1]]
                    canal_dates_num = [pd.Timestamp(d).timestamp() for d in canal_dates]
                    max_canal = [max_slope * x + max_intercept for x in canal_dates_num]
                    min_canal = [min_slope * x + min_intercept for x in canal_dates_num]
                    
                    # Verificar que las líneas no se crucen
                    if max_canal[-1] <= min_canal[-1]:
                        # Si se cruzan, hacer líneas paralelas
                        initial_distance = max_canal[0] - min_canal[0]
                        if initial_distance <= 0:
                            initial_distance = abs(max_by_date[0][1] - min_by_date[0][1])
                        
                        avg_slope = (max_slope + min_slope) / 2
                        max_intercept = max_by_date[0][1] - avg_slope * pd.Timestamp(max_by_date[0][0]).timestamp()
                        min_intercept = max_intercept - initial_distance
                        
                        max_canal = [avg_slope * x + max_intercept for x in canal_dates_num]
                        min_canal = [avg_slope * x + min_intercept for x in canal_dates_num]
                        max_slope = min_slope = avg_slope
                    
                    return max_canal, min_canal, max_slope, min_slope
                
                # Intentar crear canal comenzando con 2 puntos extremos
                used_max_points = sorted_max_by_value[:2]
                used_min_points = sorted_min_by_value[:2]
                
                max_canal, min_canal, max_slope, min_slope = create_channel_from_points(used_max_points, used_min_points)
                
                if max_canal is not None and min_canal is not None:
                    # Verificar si todos los precios están dentro del canal
                    prices_outside = 0
                    for i, price in enumerate(history["Close"]):
                        date_num = pd.Timestamp(history.index[i]).timestamp()
                        max_at_date = max_slope * date_num + (used_max_points[0][1] - max_slope * pd.Timestamp(used_max_points[0][0]).timestamp())
                        min_at_date = min_slope * date_num + (used_min_points[0][1] - min_slope * pd.Timestamp(used_min_points[0][0]).timestamp())
                        
                        if price > max_at_date or price < min_at_date:
                            prices_outside += 1
                    
                    # Si hay muchos precios fuera, intentar expandir con más puntos
                    if prices_outside > len(history) * 0.1:  # Si más del 10% está fuera
                        for num_points in range(3, len(sorted_max_by_value) + 1):
                            if num_points <= len(sorted_max_by_value) and num_points <= len(sorted_min_by_value):
                                test_max_points = sorted_max_by_value[:num_points]
                                test_min_points = sorted_min_by_value[:num_points]
                                
                                test_max_canal, test_min_canal, test_max_slope, test_min_slope = create_channel_from_points(test_max_points, test_min_points)
                                
                                if test_max_canal is not None:
                                    # Verificar cobertura
                                    outside_count = 0
                                    for i, price in enumerate(history["Close"]):
                                        date_num = pd.Timestamp(history.index[i]).timestamp()
                                        max_at_date = test_max_slope * date_num + (test_max_points[0][1] - test_max_slope * pd.Timestamp(test_max_points[0][0]).timestamp())
                                        min_at_date = test_min_slope * date_num + (test_min_points[0][1] - test_min_slope * pd.Timestamp(test_min_points[0][0]).timestamp())
                                        
                                        if price > max_at_date or price < min_at_date:
                                            outside_count += 1
                                    
                                    if outside_count < prices_outside:
                                        used_max_points = test_max_points
                                        used_min_points = test_min_points
                                        max_canal, min_canal = test_max_canal, test_min_canal
                                        max_slope, min_slope = test_max_slope, test_min_slope
                                        prices_outside = outside_count
                    
                    canal_dates = [history.index[0], history.index[-1]]
                    used_max_by_date = sorted(used_max_points, key=lambda x: x[0])
                    used_min_by_date = sorted(used_min_points, key=lambda x: x[0])
                    
                    return max_canal, min_canal, canal_dates, used_max_by_date, used_min_by_date, max_slope, min_slope, projected_max_points, projected_min_points
        
        return None, None, None, None, None, None, None, None, None
    
    # Calcular el canal según la metodología seleccionada
    if methodology == "Metodología 1: Regresión de 3 Puntos":
        max_canal, min_canal, canal_dates, slope, intercept, max_offset, min_offset = calculate_3_point_regression_channel()
        channel_data = {"type": "regression", "slope": slope, "intercept": intercept, "max_offset": max_offset, "min_offset": min_offset}
    elif methodology == "Metodología 2: Canal Cronológico Adaptativo":
        max_canal, min_canal, canal_dates, max_connection, min_connection, max_slope, min_slope, max_type, min_type = calculate_chronological_adaptive_channel()
        channel_data = {"type": "chronological", "max_connection": max_connection, "min_connection": min_connection, "max_slope": max_slope, "min_slope": min_slope, "max_type": max_type, "min_type": min_type}
    else:  # Metodología 3: Canal Adaptativo que Contiene Todos los Precios
        max_canal, min_canal, canal_dates, used_max, used_min, max_slope, min_slope, all_max_points, all_min_points = calculate_adaptive_channel_with_projection()
        channel_data = {"type": "adaptive", "used_max": used_max, "used_min": used_min, "max_slope": max_slope, "min_slope": min_slope, "all_max_points": all_max_points, "all_min_points": all_min_points}
    
    fig_main = create_base_chart()
    
    if max_canal and min_canal:
        # Añadir líneas del canal
        fig_main.add_trace(go.Scatter(
            x=canal_dates,
            y=max_canal,
            name="Canal Superior",
            line=dict(color="red", dash="dash", width=2)
        ))
        fig_main.add_trace(go.Scatter(
            x=canal_dates,
            y=min_canal,
            name="Canal Inferior",
            line=dict(color="green", dash="dash", width=2)
        ))
        
        # Añadir elementos específicos según metodología
        if methodology == "Metodología 1: Regresión de 3 Puntos":
            # Añadir línea de regresión central
            central_line = [channel_data["slope"] * x + channel_data["intercept"] for x in [pd.Timestamp(d).timestamp() for d in canal_dates]]
            fig_main.add_trace(go.Scatter(
                x=canal_dates,
                y=central_line,
                name="Línea de Regresión",
                line=dict(color="blue", dash="dot", width=1)
            ))
        elif methodology == "Metodología 2: Canal Cronológico Adaptativo":
            # Añadir conexiones cronológicas
            if channel_data["max_connection"]:
                max_dates, max_values = zip(*channel_data["max_connection"])
                fig_main.add_trace(go.Scatter(
                    x=max_dates,
                    y=max_values,
                    name=f"Máximos Cronológicos ({channel_data['max_type']})",
                    mode="markers+lines",
                    marker=dict(color="orange", size=12, symbol="circle"),
                    line=dict(color="red", width=2)
                ))
            if channel_data["min_connection"]:
                min_dates, min_values = zip(*channel_data["min_connection"])
                fig_main.add_trace(go.Scatter(
                    x=min_dates,
                    y=min_values,
                    name=f"Mínimos Cronológicos ({channel_data['min_type']})",
                    mode="markers+lines",
                    marker=dict(color="lime", size=12, symbol="circle"),
                    line=dict(color="green", width=2)
                ))
        else:  # Metodología 3
            # Mostrar todos los puntos cronológicos (30 de junio)
            if channel_data["all_max_points"]:
                all_max_dates, all_max_values = zip(*channel_data["all_max_points"])
                fig_main.add_trace(go.Scatter(
                    x=all_max_dates,
                    y=all_max_values,
                    name="Todos los Máximos (30 Jun)",
                    mode="markers",
                    marker=dict(color="orange", size=8, symbol="circle", opacity=0.6),
                    showlegend=True
                ))
            if channel_data["all_min_points"]:
                all_min_dates, all_min_values = zip(*channel_data["all_min_points"])
                fig_main.add_trace(go.Scatter(
                    x=all_min_dates,
                    y=all_min_values,
                    name="Todos los Mínimos (30 Jun)",
                    mode="markers",
                    marker=dict(color="lime", size=8, symbol="circle", opacity=0.6),
                    showlegend=True
                ))
            
            # Mostrar puntos utilizados para el canal (conectados)
            if channel_data["used_max"]:
                max_dates, max_values = zip(*channel_data["used_max"])
                fig_main.add_trace(go.Scatter(
                    x=max_dates,
                    y=max_values,
                    name=f"Máximos Utilizados ({len(max_dates)} puntos)",
                    mode="markers+lines",
                    marker=dict(color="red", size=12, symbol="diamond"),
                    line=dict(color="red", width=2)
                ))
            if channel_data["used_min"]:
                min_dates, min_values = zip(*channel_data["used_min"])
                fig_main.add_trace(go.Scatter(
                    x=min_dates,
                    y=min_values,
                    name=f"Mínimos Utilizados ({len(min_dates)} puntos)",
                    mode="markers+lines",
                    marker=dict(color="green", size=12, symbol="diamond"),
                    line=dict(color="green", width=2)
                ))
        
        # Añadir área sombreada del canal
        fig_main.add_trace(go.Scatter(
            x=canal_dates + canal_dates[::-1],
            y=max_canal + min_canal[::-1],
            fill="toself",
            fillcolor="rgba(0,100,80,0.1)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Área del Canal",
            showlegend=False
        ))
        
        # Calcular precio actual del canal en fecha actual
        current_date_num = pd.Timestamp(history.index[-1]).timestamp()
        
        if methodology == "Metodología 1: Regresión de 3 Puntos":
            current_max_canal = channel_data["slope"] * current_date_num + channel_data["intercept"] + channel_data["max_offset"]
            current_min_canal = channel_data["slope"] * current_date_num + channel_data["intercept"] + channel_data["min_offset"]
        elif methodology == "Metodología 2: Canal Cronológico Adaptativo":
            # Para metodología 2, calcular usando las conexiones cronológicas
            max_intercept = channel_data["max_connection"][0][1] - channel_data["max_slope"] * pd.Timestamp(channel_data["max_connection"][0][0]).timestamp()
            min_intercept = channel_data["min_connection"][0][1] - channel_data["min_slope"] * pd.Timestamp(channel_data["min_connection"][0][0]).timestamp()
            current_max_canal = channel_data["max_slope"] * current_date_num + max_intercept
            current_min_canal = channel_data["min_slope"] * current_date_num + min_intercept
        else:  # Metodología 3
            # Para metodología 3, calcular usando las pendientes adaptativas
            max_intercept = channel_data["used_max"][0][1] - channel_data["max_slope"] * pd.Timestamp(channel_data["used_max"][0][0]).timestamp()
            min_intercept = channel_data["used_min"][0][1] - channel_data["min_slope"] * pd.Timestamp(channel_data["used_min"][0][0]).timestamp()
            current_max_canal = channel_data["max_slope"] * current_date_num + max_intercept
            current_min_canal = channel_data["min_slope"] * current_date_num + min_intercept
        
        current_price = data["Precio Actual"]
        
        # Validar que el precio actual esté dentro del canal (para metodologías 2 y 3)
        channel_expanded = False
        if methodology not in ["Metodología 1: Regresión de 3 Puntos"]:
            if current_price > current_max_canal:
                # Expandir el canal superior para incluir el precio actual
                current_max_canal = current_price + (current_price * 0.05)
                # Recalcular max_canal
                max_intercept = current_max_canal - channel_data["max_slope"] * current_date_num
                max_canal = [channel_data["max_slope"] * x + max_intercept for x in [pd.Timestamp(d).timestamp() for d in canal_dates]]
                channel_expanded = True
            elif current_price < current_min_canal:
                # Expandir el canal inferior para incluir el precio actual
                current_min_canal = current_price - (current_price * 0.05)
                # Recalcular min_canal
                min_intercept = current_min_canal - channel_data["min_slope"] * current_date_num
                min_canal = [channel_data["min_slope"] * x + min_intercept for x in [pd.Timestamp(d).timestamp() for d in canal_dates]]
                channel_expanded = True
        
        # Calcular proximidad a límites del canal
        canal_range = current_max_canal - current_min_canal
        distance_to_max = current_max_canal - current_price
        distance_to_min = current_price - current_min_canal
        
        position_in_canal = (current_price - current_min_canal) / canal_range * 100 if canal_range > 0 else 50
        proximity_to_max = (distance_to_max / canal_range) * 100 if canal_range > 0 else 0
        proximity_to_min = (distance_to_min / canal_range) * 100 if canal_range > 0 else 0
        
        # Mostrar información del canal
        if methodology == "Metodología 1: Regresión de 3 Puntos":
            methodology_name = "3 Puntos"
        elif methodology == "Metodología 2: Canal Cronológico Adaptativo":
            methodology_name = "Cronológico"
        else:
            methodology_name = "Adaptativo"
        st.write(f"### Análisis del Canal {methodology_name}")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Posición en Canal", f"{position_in_canal:.1f}%", 
                     help="0% = Límite inferior, 100% = Límite superior")
        
        with col2:
            st.metric("Distancia a Máximo", f"{proximity_to_max:.1f}%",
                     help="Porcentaje del rango del canal hasta el límite superior")
        
        with col3:
            st.metric("Distancia a Mínimo", f"{proximity_to_min:.1f}%",
                     help="Porcentaje del rango del canal hasta el límite inferior")
        
        # Interpretación del canal
        if channel_expanded:
            st.info("ℹ️ El canal ha sido expandido automáticamente para incluir el precio actual")
        
        if position_in_canal > 80:
            st.warning("🔴 El precio está cerca del límite superior del canal - Posible sobrecompra")
        elif position_in_canal < 20:
            st.success("🟢 El precio está cerca del límite inferior del canal - Posible sobreventa")
        else:
            st.info("🟡 El precio está en la zona media del canal")
        
        # Información técnica del canal
        st.write("### Parámetros del Canal")
        if methodology == "Metodología 1: Regresión de 3 Puntos":
            st.write(f"**Pendiente:** {channel_data['slope']:.6f} (€/día)")
        elif methodology == "Metodología 2: Canal Cronológico Adaptativo":
            st.write(f"**Pendiente Superior:** {channel_data['max_slope']:.6f} (€/día)")
            st.write(f"**Pendiente Inferior:** {channel_data['min_slope']:.6f} (€/día)")
            st.write(f"**Conexión Máximos:** Año {channel_data['max_type']}")
            st.write(f"**Conexión Mínimos:** Año {channel_data['min_type']}")
        else:  # Metodología 3
            st.write(f"**Pendiente Superior:** {channel_data['max_slope']:.6f} (€/día)")
            st.write(f"**Pendiente Inferior:** {channel_data['min_slope']:.6f} (€/día)")
            st.write(f"**Puntos Máximos Utilizados:** {len(channel_data['used_max'])}")
            st.write(f"**Puntos Mínimos Utilizados:** {len(channel_data['used_min'])}")
        st.write(f"**Límite Superior Actual:** {current_max_canal:.2f} €")
        st.write(f"**Límite Inferior Actual:** {current_min_canal:.2f} €")
        st.write(f"**Amplitud del Canal:** {canal_range:.2f} €")
    
    if methodology == "Metodología 1: Regresión de 3 Puntos":
        methodology_title = "Canal de Regresión de 3 Puntos"
    elif methodology == "Metodología 2: Canal Cronológico Adaptativo":
        methodology_title = "Canal Cronológico Adaptativo"
    else:
        methodology_title = "Canal Adaptativo que Contiene Todos los Precios"
    
    fig_main.update_layout(
        title=f"{methodology_title} - {selected_ticker} ({data['Nombre']})",
        xaxis_title="Fecha",
        yaxis_title="Precio (€)",
        height=600
    )
    st.plotly_chart(fig_main, use_container_width=True)
    
    # Métricas fundamentales
    st.subheader("Métricas Fundamentales")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("PER", f"{data['PER']:.2f}" if data['PER'] is not None else "N/A")
        st.metric("Rentabilidad Dividendo", f"{data['Rentabilidad Dividendo (%)']:.2f}%" if data['Rentabilidad Dividendo (%)'] is not None else "N/A")
    
    with col2:
        st.metric("BPA", f"{data['BPA']:.2f}" if data['BPA'] is not None else "N/A")
        ebitda_formatted = f"{data['EBITDA']/1e6:.0f}M€" if data['EBITDA'] is not None else "N/A"
        st.metric("EBITDA", ebitda_formatted)
    
    with col3:
        st.metric("Precio Actual", f"{data['Precio Actual']:.2f}€")
        fcf_formatted = f"{data['FCF']/1e6:.0f}M€" if data['FCF'] is not None else "N/A"
        st.metric("FCF", fcf_formatted)
else:
    st.error(f"No hay datos disponibles para {selected_ticker}")