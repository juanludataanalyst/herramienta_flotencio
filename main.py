import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import warnings

# Suprimir warnings no críticos de pandas/yfinance (opcional, quitar si necesitas depurar)
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
def get_historical_prices(ticker):
    try:
        stock = yf.Ticker(ticker)
        # Obtener datos de los últimos 5 años
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)
        history = stock.history(start=start_date, end=end_date)
        
        if history.empty:
            print(f"No hay datos para {ticker}")
            return None, None
        
        # Limpiar datos: ordenar índice y eliminar duplicados
        history = history.sort_index()
        history = history[~history.index.duplicated(keep='last')]
        
        # Obtener el año actual y los últimos 5 años
        current_year = end_date.year
        years = list(range(current_year - 4, current_year + 1))  # Ej. [2020, 2021, 2022, 2023, 2024]
        
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
        
        # Precio actual (último cierre)
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
print("Resumen guardado en 'spanish_stocks_summary.csv'")

# Guardar históricos de precios en un CSV separado
for ticker, history in all_histories.items():
    if history is not None:
        history.to_csv(f"historical_prices_{ticker}.csv")
print("Precios históricos guardados en archivos individuales")

# Mostrar resumen
print("\nResumen de cotizadas españolas:")
print(df_results)