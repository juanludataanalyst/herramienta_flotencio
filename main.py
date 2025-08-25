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
    
    "AIR.MC",
    "ITX.MC",
    "SAN.MC",
    "IBE.MC",
    "BBVA.MC",
    "XPBR.MC",
    "XPBRA.MC",
    "CABK.MC",
    "XVALO.MC",
    "CCEP.MC",
    "FER.MC",
    "DIA.MC",
    "AMS.MC",
    "ELE.MC",
    "TEF.MC",
    "NTGY.MC",
    "XBBDC.MC",
    "XNOR.MC",
    "MTS.MC",
    "IAG.MC",
    "CLNX.MC",
    "SAB.MC",
    "ACS.MC",
    "REP.MC",
    "XELTO.MC",
    "MAP.MC",
    "BKT.MC",
    "ANA.MC",
    "PUIG.MC",
    "RED.MC",
    "GRF.MC",
    "ANE.MC",
    "MRL.MC",
    "UNI.MC",
    "IDR.MC",
    "GCO.MC",
    "XGGB.MC",
    "XCMIG.MC",
    "XCOP.MC",
    "FCC.MC",
    "XNEO.MC",
    "FDR.MC",
    "AENA.MC",
    "LOG.MC",
    "COL.MC",
    "XALFA.MC",
    "ENG.MC",
    "CIE.MC",
    "VID.MC",
    "ROVI.MC",
    "SCYR.MC",
    "NHH.MC",
    "ACX.MC",
    "VIS.MC",
    "EBRO.MC",
    "ALM.MC",
    "GEST.MC",
    "APAM.MC",
    "ENO.MC",
    "GRE.MC",
    "MEL.MC",
    "CAF.MC",
    "MFEA.MC",
    "TRE.MC",
    "IMC.MC",
    "SLR.MC",
    "MVC.MC",
    "LDA.MC",
    "PSG.MC",
    "PHM.MC",
    "HOME.MC",
    "FAE.MC",
    "A3M.MC",
    "EDR.MC",
    "CASH.MC",
    "XBRK.MC",
    "AEDAS.MC",
    "XUSIO.MC",
    "YCPS.MC",
    "XUSI.MC",
    "R4.MC",
   # "EAT.MC",
    "RLIA.MC",
    "ART.MC",
    "CBAV.MC",
    "ENC.MC",
    "ADX.MC",
    "MCM.MC",
    "DOM.MC",
    "YATO.MC",
    "GSJ.MC",
    "TUB.MC",
    "ALC.MC",
    "EBROM.MC",
    "PRS.MC",
    "XVOLB.MC",
    "NAT.MC",
    "TLGO.MC",
    "YMHRE.MC",
    "ALNT.MC",
    "ENER.MC",
    "IZER.MC",
    "COXE.MC",
    "ISUR.MC",
    "ECR.MC",
    "RJF.MC",
    "OHLA.MC",
    "ATRY.MC",
    "ATSI.MC",
    "IBG.MC",
    "AZK.MC",
    "PRM.MC",
    "ARM.MC",
    "CEV.MC",
    "AMP.MC",
    "NXT.MC",
    "SCBYT.MC",
    "SQRL.MC",
    "ORY.MC",
    "INDXA.MC",
    "SNG.MC",
   # "AI.MC",
    "YDOA.MC",
    "GAM.MC",
    "YVIT.MC",
    "NEA.MC",
    "GGR.MC",
  #  "BKY.MC",
    "EIDF.MC",
    "NTH.MC",
    "AMEN.MC",
    "YTRM.MC",
    "YMRE.MC",
    "LIB.MC",
    "YHSP.MC",
    "LLYC.MC",
    "YGOP.MC",
    "ENRS.MC",
    "OLE.MC",
    "TRG.MC",
    "APG.MC",
    "MAKS.MC",
    "YIPS.MC",
    "VOC.MC",
    "YAI1.MC",
    "ETC.MC",
    "SCO.MC",
    "ALQ.MC",
    "YIBI.MC",
    "CLR.MC",
    "PAR.MC",
    "GIGA.MC",
    "LGT.MC",
    "PANG.MC",
    "YADV.MC",
    "FACE.MC",
    "EZE.MC",
    "MTB.MC",
    "MDF.MC",
    "ADZ.MC",
    "VYT.MC",
    "AGIL.MC",
    "480S.MC",
    "NBI.MC",
    "DESA.MC",
    "YQUO.MC",
    "YBAR.MC",
    "END.MC",
    "SAI.MC",
    "GRI.MC",
    "LAB.MC",
    "OPTS.MC",
    "HLZ.MC",
    "LLN.MC",
    "REN.MC",
    "BST.MC",
    "YMIB.MC",
    "COM.MC",
    "RIO.MC",
    "NYE.MC",
    "ELZ.MC",
    "RSS.MC",
    "IFLEX.MC",
    "PVA.MC",
    "MIO.MC",
    "VANA.MC",
    "SPH.MC",
    "HAN.MC",
    "YTRA.MC",
    "CITY.MC",
   # "CIRSA.MC",
    "COXG.MC",
    "GRF-P.MC",
   # "HBX.MC",
  #  "IFL-D.MC",
  #  "RDG.MC",
   # "SCAP7.MC",
  #  "SCCMM.MC",
    "SCIG4.MC",
    "XBBAR.MC",
    "XCOPO.MC",
   # "YEPSA.MC",
    "YGO2.MC",
    "YIRG.MC",
    # "YMIL.MC"
]

# Función para obtener precios históricos y métricas fundamentales
@st.cache_data
def get_historical_prices(ticker):
    try:
        stock = yf.Ticker(ticker)
        # Obtener datos desde 2022 hasta hoy
        start_date = datetime(2022, 1, 1)
        end_date = datetime.now()
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
        
        # Métricas básicas de valoración
        trailing_pe = info.get("trailingPE", None)
        forward_pe = info.get("forwardPE", None)
        peg_ratio = info.get("pegRatio", None)
        price_to_book = info.get("priceToBook", None)
        price_to_sales = info.get("priceToSalesTrailing12Months", None)
        
        # Métricas de rentabilidad
        roe = info.get("returnOnEquity", None)
        roa = info.get("returnOnAssets", None)
        profit_margin = info.get("profitMargins", None)
        operating_margin = info.get("operatingMargins", None)
        
        # Métricas de dividendos
        dividend_rate = info.get("dividendRate", None)
        dividend_yield = info.get("dividendYield", None)
        if dividend_yield:
            dividend_yield *= 100  # Convertir a porcentaje
        elif dividend_rate and current_price:
            dividend_yield = (dividend_rate / current_price) * 100
        
        # Validar dividend yield (eliminar valores anómalos)
        if dividend_yield and dividend_yield > 25:
            dividend_yield = None
            
        payout_ratio = info.get("payoutRatio", None)
        if payout_ratio:
            payout_ratio *= 100  # Convertir a porcentaje
        
        # Métricas de crecimiento
        revenue_growth = info.get("revenueGrowth", None)
        earnings_growth = info.get("earningsGrowth", None)
        if revenue_growth:
            revenue_growth *= 100  # Convertir a porcentaje
        if earnings_growth:
            earnings_growth *= 100  # Convertir a porcentaje
            
        # Métricas financieras
        market_cap = info.get("marketCap", None)
        enterprise_value = info.get("enterpriseValue", None)
        total_debt = info.get("totalDebt", None)
        total_cash = info.get("totalCash", None)
        
        # EBITDA
        ebitda = info.get("ebitda", None)
        if ebitda is None and hasattr(stock, "financials"):
            try:
                ebitda = stock.financials.loc["EBITDA"].iloc[-1] if "EBITDA" in stock.financials.index else None
            except:
                ebitda = None
                
        # Free Cash Flow
        fcf = info.get("freeCashflow", None)
        if fcf is None and hasattr(stock, "cashflow"):
            try:
                fcf = stock.cashflow.loc["Free Cash Flow"].iloc[-1] if "Free Cash Flow" in stock.cashflow.index else None
            except:
                fcf = None
        
        # Ratios calculados
        debt_to_equity = None
        if total_debt and market_cap:
            debt_to_equity = total_debt / market_cap
            
        ev_ebitda = None
        if enterprise_value and ebitda and ebitda > 0:
            ev_ebitda = enterprise_value / ebitda
            
        fcf_yield = None
        if fcf and market_cap and fcf > 0:
            fcf_yield = (fcf / market_cap) * 100
        
        fundamentals = {
            # Valoración
            "PER": round(trailing_pe, 2) if trailing_pe else None,
            "PER Forward": round(forward_pe, 2) if forward_pe else None,
            "PEG": round(peg_ratio, 2) if peg_ratio else None,
            "P/B": round(price_to_book, 2) if price_to_book else None,
            "P/S": round(price_to_sales, 2) if price_to_sales else None,
            "EV/EBITDA": round(ev_ebitda, 2) if ev_ebitda else None,
            
            # Rentabilidad
            "BPA": round(info.get("trailingEps", None), 2) if info.get("trailingEps") else None,
            "ROE (%)": round(roe * 100, 2) if roe else None,
            "ROA (%)": round(roa * 100, 2) if roa else None,
            "Margen Beneficio (%)": round(profit_margin * 100, 2) if profit_margin else None,
            "Margen Operativo (%)": round(operating_margin * 100, 2) if operating_margin else None,
            
            # Dividendos
            "Rentabilidad Dividendo (%)": round(dividend_yield, 2) if dividend_yield else None,
            "Payout Ratio (%)": round(payout_ratio, 2) if payout_ratio else None,
            
            # Crecimiento
            "Crecimiento Ingresos (%)": round(revenue_growth, 2) if revenue_growth else None,
            "Crecimiento Beneficios (%)": round(earnings_growth, 2) if earnings_growth else None,
            
            # Financieras
            "Cap. Mercado": market_cap,
            "EBITDA": ebitda,
            "FCF": fcf,
            "FCF Yield (%)": round(fcf_yield, 2) if fcf_yield else None,
            "Deuda/Equity": round(debt_to_equity, 2) if debt_to_equity else None,
            "Deuda Total": total_debt,
            "Efectivo Total": total_cash
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
print("Resumen guardado en 'spanish_stocks_summary.csv'")

for ticker, (history, _, _, _) in all_histories.items():
    if history is not None:
        history.to_csv(f"historical_prices_{ticker}.csv")
print("Precios históricos guardados en archivos individuales")

# Interfaz de Streamlit
st.title("Herramienta de Inversión - Valores Españoles")


# Filtros de empresas
st.subheader("🔍 Filtros de Empresas")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("**Filtro por PER**")
    per_filter_enabled = st.checkbox("Activar filtro PER")
    per_min = st.number_input("PER mínimo", value=0.0, step=0.1) if per_filter_enabled else 0.0
    per_max = st.number_input("PER máximo", value=30.0, step=0.1) if per_filter_enabled else 30.0

with col2:
    st.write("**Filtro por Deuda/Equity**")
    debt_filter_enabled = st.checkbox("Activar filtro Deuda/Equity")
    debt_max = st.number_input("Deuda/Equity máximo", value=1.0, step=0.1) if debt_filter_enabled else 1.0

with col3:
    st.write("**Aplicar Filtros**")
    
    col3_1, col3_2 = st.columns(2)
    with col3_1:
        filter_button = st.button("🔍 Filtrar Empresas")
    with col3_2:
        clear_button = st.button("🗑️ Limpiar Filtros")
    
    if clear_button:
        st.session_state.filtered_tickers = tickers
        st.success("✅ Filtros limpiados. Mostrando todas las empresas.")
    
    if filter_button:
        with st.spinner("Filtrando empresas..."):
            filtered_companies = []
            
            # Usar datos ya cargados en lugar de hacer nuevas llamadas a la API
            for company_data in results:
                ticker = company_data["Ticker"]
                
                # Obtener PER y Deuda/Equity de los datos ya cargados
                per = company_data.get("PER", None)
                debt_equity = company_data.get("Deuda/Equity", None)
                
                # Aplicar filtros
                include_company = True
                exclusion_reasons = []
                
                if per_filter_enabled:
                    if per is None:
                        include_company = False
                        exclusion_reasons.append("PER no disponible")
                    elif per < per_min or per > per_max:
                        include_company = False
                        exclusion_reasons.append(f"PER {per:.2f} fuera del rango [{per_min}-{per_max}]")
                
                if debt_filter_enabled:
                    if debt_equity is None:
                        include_company = False
                        exclusion_reasons.append("Deuda/Equity no disponible")
                    elif debt_equity > debt_max:
                        include_company = False
                        exclusion_reasons.append(f"Deuda/Equity {debt_equity:.2f} > {debt_max}")
                
                if include_company:
                    filtered_companies.append({
                        "Ticker": ticker,
                        "Nombre": company_data.get("Nombre", ticker),
                        "Precio": f'{company_data.get('Precio Actual', 0):.2f}€' if company_data.get("Precio Actual") else "No disponible",
                        "PER": round(per, 2) if per else "No disponible",
                        "Deuda/Equity": round(debt_equity, 2) if debt_equity else "No disponible"
                    })
            
            if filtered_companies:
                st.success(f"✅ {len(filtered_companies)} de {len(tickers)} empresas cumplen los filtros:")
                filtered_df = pd.DataFrame(filtered_companies)
                st.dataframe(filtered_df, use_container_width=True)
                
                # Crear lista de tickers filtrados para el selector
                filtered_tickers = [company["Ticker"] for company in filtered_companies]
                st.session_state.filtered_tickers = filtered_tickers
                
                # Mostrar estadísticas de filtrado
                st.info(f"📊 Se han excluido {len(tickers) - len(filtered_companies)} empresas que no cumplen los criterios.")
            else:
                st.warning("⚠️ Ninguna empresa cumple los filtros especificados")
                st.info("💡 Prueba a relajar los criterios de filtrado")
                st.session_state.filtered_tickers = tickers

# Si no se han aplicado filtros, usar todos los tickers
if 'filtered_tickers' not in st.session_state:
    st.session_state.filtered_tickers = tickers

# Seleccionar empresa (usar tickers filtrados si existen)
available_tickers = st.session_state.get('filtered_tickers', tickers)

# Crear diccionario de ticker -> nombre para el selector
ticker_to_name = {}
for ticker in available_tickers:
    # Buscar el nombre en los datos ya cargados
    ticker_data = next((item for item in results if item['Ticker'] == ticker), None)
    if ticker_data:
        company_name = ticker_data['Nombre']
        if company_name != "N/A":
            ticker_to_name[ticker] = f"{company_name} ({ticker})"
        else:
            ticker_to_name[ticker] = ticker
    else:
        ticker_to_name[ticker] = ticker

# Crear lista de opciones para mostrar en el selector
options_list = list(ticker_to_name.values())
name_to_ticker = {v: k for k, v in ticker_to_name.items()}

# Selector con nombres de empresas
selected_option = st.selectbox("🏢 Seleccionar Empresa", options_list)
selected_ticker = name_to_ticker[selected_option]
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
            max_dates_viz, max_values = zip(*max_points)
            fig.add_trace(go.Scatter(
                x=max_dates_viz,
                y=max_values,
                name="Máximos Anuales",
                mode="markers",
                marker=dict(color="red", size=10)
            ))
        if min_points:
            min_dates_viz, min_values = zip(*min_points)
            fig.add_trace(go.Scatter(
                x=min_dates_viz,
                y=min_values,
                name="Mínimos Anuales",
                mode="markers",
                marker=dict(color="green", size=10)
            ))
        return fig
    
    # Definir funciones primero
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
    
    # Usar metodología por defecto para mostrar el gráfico inicial
    methodology = "Metodología 1: Canal Cronológico Adaptativo"
    
    # Calcular y mostrar el gráfico según metodología
    if methodology == "Metodología 1: Canal Cronológico Adaptativo":
        max_canal, min_canal, canal_dates, max_connection, min_connection, max_slope, min_slope, max_type, min_type = calculate_chronological_adaptive_channel()
        channel_data = {"type": "chronological", "max_connection": max_connection, "min_connection": min_connection, "max_slope": max_slope, "min_slope": min_slope, "max_type": max_type, "min_type": min_type}
    elif methodology == "Metodología 2: Canal Adaptativo que Contiene Todos los Precios":
        max_canal, min_canal, canal_dates, used_max, used_min, max_slope, min_slope, all_max_points, all_min_points = calculate_adaptive_channel_with_projection()
        channel_data = {"type": "adaptive", "used_max": used_max, "used_min": used_min, "max_slope": max_slope, "min_slope": min_slope, "all_max_points": all_max_points, "all_min_points": all_min_points}
    else:  # Metodología 3: Regresión de 3 Puntos
        max_canal, min_canal, canal_dates, slope, intercept, max_offset, min_offset = calculate_3_point_regression_channel()
        channel_data = {"type": "regression", "slope": slope, "intercept": intercept, "max_offset": max_offset, "min_offset": min_offset}
    
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
        if methodology == "Metodología 1: Canal Cronológico Adaptativo":
            # Añadir conexiones cronológicas
            if channel_data["max_connection"]:
                max_dates_conn, max_values = zip(*channel_data["max_connection"])
                fig_main.add_trace(go.Scatter(
                    x=max_dates_conn,
                    y=max_values,
                    name=f"Máximos Cronológicos ({channel_data['max_type']})",
                    mode="markers+lines",
                    marker=dict(color="orange", size=12, symbol="circle"),
                    line=dict(color="red", width=2)
                ))
            if channel_data["min_connection"]:
                min_dates_conn, min_values = zip(*channel_data["min_connection"])
                fig_main.add_trace(go.Scatter(
                    x=min_dates_conn,
                    y=min_values,
                    name=f"Mínimos Cronológicos ({channel_data['min_type']})",
                    mode="markers+lines",
                    marker=dict(color="lime", size=12, symbol="circle"),
                    line=dict(color="green", width=2)
                ))
        elif methodology == "Metodología 2: Canal Adaptativo que Contiene Todos los Precios":
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
                max_dates_used, max_values = zip(*channel_data["used_max"])
                fig_main.add_trace(go.Scatter(
                    x=max_dates_used,
                    y=max_values,
                    name=f"Máximos Utilizados ({len(max_dates_used)} puntos)",
                    mode="markers+lines",
                    marker=dict(color="red", size=12, symbol="diamond"),
                    line=dict(color="red", width=2)
                ))
            if channel_data["used_min"]:
                min_dates_used, min_values = zip(*channel_data["used_min"])
                fig_main.add_trace(go.Scatter(
                    x=min_dates_used,
                    y=min_values,
                    name=f"Mínimos Utilizados ({len(min_dates_used)} puntos)",
                    mode="markers+lines",
                    marker=dict(color="green", size=12, symbol="diamond"),
                    line=dict(color="green", width=2)
                ))
        else:  # Metodología 3: Regresión de 3 Puntos
            # Añadir línea de regresión central
            central_line = [channel_data["slope"] * x + channel_data["intercept"] for x in [pd.Timestamp(d).timestamp() for d in canal_dates]]
            fig_main.add_trace(go.Scatter(
                x=canal_dates,
                y=central_line,
                name="Línea de Regresión",
                line=dict(color="blue", dash="dot", width=1)
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
    
    # Título del gráfico
    if methodology == "Metodología 1: Canal Cronológico Adaptativo":
        methodology_title = "Canal Cronológico Adaptativo"
    elif methodology == "Metodología 2: Canal Adaptativo que Contiene Todos los Precios":
        methodology_title = "Canal Adaptativo que Contiene Todos los Precios"
    else:
        methodology_title = "Canal de Regresión de 3 Puntos"
    
    fig_main.update_layout(
        title=f"{methodology_title} - {selected_ticker} ({data['Nombre']})",
        xaxis_title="Fecha",
        yaxis_title="Precio (€)",
        height=600
    )
    st.plotly_chart(fig_main, use_container_width=True)

    # --- INICIO: Nueva sección de Métrica de Posición en Canal ---
    st.subheader("📊 Posición Actual en el Canal de Precios")

    # Definir la función de cálculo aquí para que tenga acceso a las variables del scope
    def calculate_channel_position(current_price, canal_dates, max_canal, min_canal, history):
        if not all([current_price, canal_dates, max_canal, min_canal, not history.empty]):
            return None, None, None

        try:
            # Timestamps para las fechas del canal
            x1 = pd.Timestamp(canal_dates[0]).timestamp()
            x2 = pd.Timestamp(canal_dates[1]).timestamp()

            # Valores del canal en esas fechas
            y1_upper, y2_upper = max_canal[0], max_canal[1]
            y1_lower, y2_lower = min_canal[0], min_canal[1]

            # Evitar división por cero si las fechas son las mismas
            if x2 == x1:
                return None, None, None

            # Calcular pendientes e interceptos
            slope_upper = (y2_upper - y1_upper) / (x2 - x1)
            intercept_upper = y1_upper - slope_upper * x1
            slope_lower = (y2_lower - y1_lower) / (x2 - x1)
            intercept_lower = y1_lower - slope_lower * x1

            # Timestamp de la fecha actual (último dato)
            current_timestamp = pd.Timestamp(history.index[-1]).timestamp()

            # Calcular valor del canal en la fecha actual
            upper_value_today = slope_upper * current_timestamp + intercept_upper
            lower_value_today = slope_lower * current_timestamp + intercept_lower

            # Calcular la métrica
            channel_height = upper_value_today - lower_value_today
            if channel_height <= 0: # Usar <= para evitar divisiones por cero o negativas
                return None, upper_value_today, lower_value_today

            position_pct = ((current_price - lower_value_today) / channel_height) * 100
            return position_pct, upper_value_today, lower_value_today
        except Exception as e:
            print(f"Error calculando posición en canal: {e}")
            return None, None, None

    # Calcular la posición en el canal (reutilizable para alertas)
    if max_canal and min_canal:
        position_pct, upper_val, lower_val = calculate_channel_position(
            data['Precio Actual'], canal_dates, max_canal, min_canal, history
        )

        if position_pct is not None:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric(
                    label="Posición en Canal",
                    value=f"{position_pct:.1f}%"
                )
                st.write(f"**Soporte:** {lower_val:.2f}€")
                st.write(f"**Resistencia:** {upper_val:.2f}€")

            with col2:
                # Barra de progreso para visualización
                st.progress(min(max(position_pct, 0), 100) / 100)

                # Interpretación de la métrica
                if position_pct < 0:
                    st.error("🔴 **Ruptura Bajista:** El precio está por debajo del canal.")
                elif position_pct < 20:
                    st.success("🟢 **Zona de Soporte:** El precio está cerca de la base del canal.")
                elif position_pct < 80:
                    st.info("🔵 **Zona Media:** El precio está en la parte central del canal.")
                elif position_pct <= 100:
                    st.warning("🟠 **Zona de Resistencia:** El precio está cerca del techo del canal.")
                else:
                    st.error("🔴 **Ruptura Alcista:** El precio está por encima del canal.")
        else:
            st.info("No se pudo calcular la posición en el canal para esta acción.")
    # --- FIN: Nueva sección ---
    
    # Selector de metodología después del gráfico
    methodology = st.selectbox("Seleccionar Metodología de Canal", 
                              ["Metodología 1: Canal Cronológico Adaptativo", 
                               "Metodología 2: Canal Adaptativo que Contiene Todos los Precios",
                               "Metodología 3: Regresión de 3 Puntos"])
    
    # Mostrar descripción de la metodología seleccionada
    if methodology == "Metodología 1: Canal Cronológico Adaptativo":
        st.subheader("Metodología 1: Canal Cronológico Adaptativo")
        st.write("Esta metodología conecta cronológicamente los puntos de máximos y mínimos usando sus fechas reales. Comienza uniendo el primer año (2022) con el segundo (2023). Si el tercer año (2024) queda fuera de esas líneas, entonces conecta directamente el primer año con el tercero.")
    elif methodology == "Metodología 2: Canal Adaptativo que Contiene Todos los Precios":
        st.subheader("Metodología 2: Canal Adaptativo que Contiene Todos los Precios")
        st.write("Esta metodología comienza uniendo los dos máximos y mínimos más extremos proyectados al 30 de junio, pero expande automáticamente el canal incluyendo más puntos hasta que todos los precios históricos estén contenidos dentro del canal.")
    else:
        st.subheader("Metodología 3: Canal de Regresión de 3 Puntos")
        st.write("Esta metodología utiliza una regresión lineal sobre los 3 máximos y 3 mínimos anuales para crear un canal de precios. El canal muestra la tendencia y rango de precios más representativo.")
    
    
    # Métricas fundamentales
    st.subheader("Métricas Fundamentales")
    
    # Función auxiliar para formatear grandes números
    def format_large_number(value):
        if value is None:
            return "-"
        if abs(value) >= 1e9:
            return f"{value/1e9:.1f}B€"
        elif abs(value) >= 1e6:
            return f"{value/1e6:.0f}M€"
        elif abs(value) >= 1e3:
            return f"{value/1e3:.0f}K€"
        else:
            return f"{value:.0f}€"
    
    # Crear pestañas para organizar mejor las métricas
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Valoración", "💰 Rentabilidad", "📈 Crecimiento", "🏦 Financieras"])
    
    with tab1:
        st.write("### Ratios de Valoración")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Precio Actual", f"{data['Precio Actual']:.2f}€")
            st.metric("PER", f"{data['PER']:.2f}" if data['PER'] is not None else "-")
        
        with col2:
            st.metric("PER Forward", f"{data['PER Forward']:.2f}" if data['PER Forward'] is not None else "-")
            st.metric("PEG", f"{data['PEG']:.2f}" if data['PEG'] is not None else "-")
        
        with col3:
            st.metric("P/B", f"{data['P/B']:.2f}" if data['P/B'] is not None else "-")
            st.metric("P/S", f"{data['P/S']:.2f}" if data['P/S'] is not None else "-")
        
        with col4:
            st.metric("EV/EBITDA", f"{data['EV/EBITDA']:.2f}" if data['EV/EBITDA'] is not None else "-")
            market_cap_formatted = format_large_number(data['Cap. Mercado'])
            st.metric("Cap. Mercado", market_cap_formatted)

    with tab2:
        st.write("### Métricas de Rentabilidad")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("BPA", f"{data['BPA']:.2f}€" if data['BPA'] is not None else "-")
            st.metric("ROE", f"{data['ROE (%)']:.2f}%" if data['ROE (%)'] is not None else "-")
        
        with col2:
            st.metric("ROA", f"{data['ROA (%)']:.2f}%" if data['ROA (%)'] is not None else "-")
            st.metric("Margen Beneficio", f"{data['Margen Beneficio (%)']:.2f}%" if data['Margen Beneficio (%)'] is not None else "-")
        
        with col3:
            st.metric("Margen Operativo", f"{data['Margen Operativo (%)']:.2f}%" if data['Margen Operativo (%)'] is not None else "-")
            st.metric("Rent. Dividendo", f"{data['Rentabilidad Dividendo (%)']:.2f}%" if data['Rentabilidad Dividendo (%)'] is not None else "-")
        
        with col4:
            st.metric("Payout Ratio", f"{data['Payout Ratio (%)']:.2f}%" if data['Payout Ratio (%)'] is not None else "-")
            st.metric("FCF Yield", f"{data['FCF Yield (%)']:.2f}%" if data['FCF Yield (%)'] is not None else "-")

    with tab3:
        st.write("### Métricas de Crecimiento")
        col1, col2 = st.columns(2)
        
        with col1:
            revenue_growth = data['Crecimiento Ingresos (%)']
            revenue_delta = f"+{revenue_growth:.2f}%" if revenue_growth and revenue_growth > 0 else f"{revenue_growth:.2f}%" if revenue_growth else None
            st.metric("Crecimiento Ingresos", 
                     f"{revenue_growth:.2f}%" if revenue_growth is not None else "-",
                     delta=revenue_delta)
        
        with col2:
            earnings_growth = data['Crecimiento Beneficios (%)']
            earnings_delta = f"+{earnings_growth:.2f}%" if earnings_growth and earnings_growth > 0 else f"{earnings_growth:.2f}%" if earnings_growth else None
            st.metric("Crecimiento Beneficios",
                     f"{earnings_growth:.2f}%" if earnings_growth is not None else "-",
                     delta=earnings_delta)
        
        # Interpretación de crecimiento
        if revenue_growth is not None and earnings_growth is not None:
            if revenue_growth > 10 and earnings_growth > 10:
                st.success("🚀 Empresa en fuerte crecimiento")
            elif revenue_growth > 5 and earnings_growth > 5:
                st.info("📈 Crecimiento moderado y sostenible")
            elif revenue_growth < 0 or earnings_growth < 0:
                st.warning("⚠️ Crecimiento negativo - Revisar situación")
            else:
                st.info("📊 Crecimiento estable")

    with tab4:
        st.write("### Situación Financiera")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ebitda_formatted = format_large_number(data['EBITDA'])
            st.metric("EBITDA", ebitda_formatted)
            fcf_formatted = format_large_number(data['FCF'])
            st.metric("FCF", fcf_formatted)
        
        with col2:
            debt_formatted = format_large_number(data['Deuda Total'])
            st.metric("Deuda Total", debt_formatted)
            cash_formatted = format_large_number(data['Efectivo Total'])
            st.metric("Efectivo Total", cash_formatted)
        
        with col3:
            st.metric("Deuda/Equity", f"{data['Deuda/Equity']:.2f}" if data['Deuda/Equity'] is not None else "-")
            
            # Calcular deuda neta
            debt_total = data['Deuda Total'] if data['Deuda Total'] else 0
            cash_total = data['Efectivo Total'] if data['Efectivo Total'] else 0
            net_debt = debt_total - cash_total
            net_debt_formatted = format_large_number(net_debt) if debt_total or cash_total else "-"
            st.metric("Deuda Neta", net_debt_formatted)
        
        # Interpretación financiera
        debt_equity = data['Deuda/Equity']
        if debt_equity is not None:
            if debt_equity < 0.3:
                st.success("💪 Situación financiera muy sólida")
            elif debt_equity < 0.6:
                st.info("✅ Situación financiera saludable")
            elif debt_equity < 1.0:
                st.warning("⚠️ Endeudamiento moderado-alto")
            else:
                st.error("🔴 Endeudamiento elevado - Riesgo financiero")
else:
    st.error(f"No hay datos disponibles para {selected_ticker}")