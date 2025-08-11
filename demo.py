# demo.py
# Dash app para "Implementación de una solución de BI para analizar y predecir multas del SAT (2025)"
# Autor: Carlos E. Laupa (con asistencia)
# Requisitos: pip install dash==2.* pandas numpy plotly

import os
import sys
import json
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime
from dash import Dash, html, dcc, Input, Output, State, dash_table

# ----------------------------
# 1) Cargar datos
# ----------------------------
# Intenta leer desde ./SAT/MultasLimpias.csv (misma carpeta del script) o desde variable de entorno
DEFAULT_PATHS = [
    os.path.join(os.path.dirname(__file__), "MultasLimpias.csv"),
    os.path.join(os.path.dirname(__file__), "data", "MultasLimpias.csv"),
    "MultasLimpias.csv",
]

csv_path = next((p for p in DEFAULT_PATHS if os.path.exists(p)), None)
if csv_path is None:
    raise FileNotFoundError(
        "No se encontró 'MultasLimpias.csv'. Coloca el CSV en la misma carpeta de demo.py "
        "o en ./data/ y vuelve a ejecutar."
    )

df = pd.read_csv(csv_path, encoding="latin-1", sep=None, engine="python")


# Normaliza nombres esperados (ajusta si tus nombres difieren)
# Esperados: ['IdMulta','IdFalta','CodigoFalta','TipoFormato','NivelGravedad','IdTiempo','Anio','Mes','MultasImpuestas','ImporteImpuesto','ReincidenciaImpuesta']
df.columns = [c.strip() for c in df.columns]

# Tipos
int_cols = ["Anio","Mes","IdMulta","IdFalta","IdTiempo"]
for c in int_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

num_cols = ["MultasImpuestas","ImporteImpuesto","ReincidenciaImpuesta"]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Crea columna de periodo YYYY-MM (si hay Anio y Mes)
if "Anio" in df.columns and "Mes" in df.columns:
    # Rellena Mes de 1..12 para ordenar
    df["Mes"] = pd.to_numeric(df["Mes"], errors="coerce")
    df["Periodo"] = pd.to_datetime(dict(year=df["Anio"], month=df["Mes"].fillna(1), day=1), errors="coerce")
else:
    df["Periodo"] = pd.NaT

# Limpia valores de categorías
for cat in ["CodigoFalta","TipoFormato","NivelGravedad"]:
    if cat in df.columns:
        df[cat] = df[cat].astype(str).str.strip()

# ----------------------------
# 2) App
# ----------------------------
app = Dash(__name__)
server = app.server
app.title = "Multas SAT - BI 2025"

def kpi_card(title, id_value):
    return html.Div(
        className="card",
        children=[
            html.Div(title, className="card-title"),
            html.Div(id=id_value, className="card-value")
        ],
        style={
            "background":"#ffffff","borderRadius":"16px","padding":"16px",
            "boxShadow":"0 8px 24px rgba(0,0,0,0.08)","minWidth":"220px","textAlign":"center"
        }
    )

# Opciones de filtros
def options_from_series(s):
    return [{"label": str(v), "value": v} for v in sorted(s.dropna().unique())]

options_anio = options_from_series(df["Anio"]) if "Anio" in df.columns else []
options_mes = [{"label": f"{int(m):02d}", "value": int(m)} for m in sorted(df["Mes"].dropna().unique())] if "Mes" in df.columns else []
options_gravedad = options_from_series(df["NivelGravedad"]) if "NivelGravedad" in df.columns else []
options_formato = options_from_series(df["TipoFormato"]) if "TipoFormato" in df.columns else []
options_codigo = options_from_series(df["CodigoFalta"]) if "CodigoFalta" in df.columns else []

# Layout
app.layout = html.Div([
    html.H1("SAT | Multas 2022–2024 (y 2025) — Análisis Interactivo", style={"textAlign":"center"}),
    html.Div("Selecciona filtros para actualizar KPIs y gráficos. Exporta tablas si lo necesitas.", style={"textAlign":"center","marginBottom":"10px"}),

    html.Div([
        html.Div([
            html.Label("Año"),
            dcc.Dropdown(id="f_anio", options=options_anio, multi=True, placeholder="Todos")
        ], style={"flex":"1","minWidth":"160px"}),
        html.Div([
            html.Label("Mes"),
            dcc.Dropdown(id="f_mes", options=options_mes, multi=True, placeholder="Todos")
        ], style={"flex":"1","minWidth":"160px"}),
        html.Div([
            html.Label("Nivel de Gravedad"),
            dcc.Dropdown(id="f_gravedad", options=options_gravedad, multi=True, placeholder="Todos")
        ], style={"flex":"1","minWidth":"200px"}),
        html.Div([
            html.Label("Tipo de Formato"),
            dcc.Dropdown(id="f_formato", options=options_formato, multi=True, placeholder="Todos")
        ], style={"flex":"1","minWidth":"200px"}),
        html.Div([
            html.Label("Código de Falta"),
            dcc.Dropdown(id="f_codigo", options=options_codigo, multi=True, placeholder="Todos")
        ], style={"flex":"1","minWidth":"220px"}),
    ], style={"display":"flex","gap":"12px","flexWrap":"wrap","margin":"16px 0"}),

    html.Div([
        kpi_card("Multas impuestas", "kpi_multas"),
        kpi_card("Importe total (S/)", "kpi_importe"),
        kpi_card("% Reincidencia", "kpi_reincidencia"),
        kpi_card("Códigos únicos", "kpi_codigos"),
    ], style={"display":"flex","gap":"16px","flexWrap":"wrap","justifyContent":"center","marginBottom":"16px"}),

    html.Div([
        html.Div([dcc.Graph(id="g_mes", config={"displaylogo": False})], style={"flex":"1","minWidth":"320px"}),
        html.Div([dcc.Graph(id="g_gravedad", config={"displaylogo": False})], style={"flex":"1","minWidth":"320px"}),
    ], style={"display":"flex","gap":"16px","flexWrap":"wrap"}),

    html.Div([
        html.Div([dcc.Graph(id="g_top_codigos", config={"displaylogo": False})], style={"flex":"1","minWidth":"320px"}),
        html.Div([dcc.Graph(id="g_tendencia", config={"displaylogo": False})], style={"flex":"1","minWidth":"320px"}),
    ], style={"display":"flex","gap":"16px","flexWrap":"wrap","marginTop":"16px"}),

    html.Hr(),
    html.H3("Detalle de datos (filtrados)"),
    dash_table.DataTable(
        id="tabla",
        page_size=10,
        export_format="csv",
        style_table={"overflowX":"auto"},
        style_cell={"padding":"6px","fontFamily":"Inter, system-ui, sans-serif", "fontSize":"14px"},
        style_header={"backgroundColor":"#f5f7fa","fontWeight":"600"}
    ),
    html.Div(id="tabla_rows", style={"marginTop":"8px","color":"#555"}),

    html.Div(id="data_info", style={"display":"none"})
], style={"maxWidth":"1200px","margin":"0 auto","padding":"20px"})

# ----------------------------
# 3) Callbacks
# ----------------------------
def apply_filters(df_, anios, meses, gravedades, formatos, codigos):
    dff = df_.copy()
    if anios: dff = dff[dff["Anio"].isin(anios)]
    if meses: dff = dff[dff["Mes"].isin(meses)]
    if gravedades: dff = dff[dff["NivelGravedad"].isin(gravedades)]
    if formatos: dff = dff[dff["TipoFormato"].isin(formatos)]
    if codigos: dff = dff[dff["CodigoFalta"].isin(codigos)]
    return dff

@app.callback(
    Output("kpi_multas","children"),
    Output("kpi_importe","children"),
    Output("kpi_reincidencia","children"),
    Output("kpi_codigos","children"),
    Output("g_mes","figure"),
    Output("g_gravedad","figure"),
    Output("g_top_codigos","figure"),
    Output("g_tendencia","figure"),
    Output("tabla","columns"),
    Output("tabla","data"),
    Output("tabla_rows","children"),
    Input("f_anio","value"),
    Input("f_mes","value"),
    Input("f_gravedad","value"),
    Input("f_formato","value"),
    Input("f_codigo","value"),
)
def update_dashboard(anios, meses, gravedades, formatos, codigos):
    dff = apply_filters(df, anios, meses, gravedades, formatos, codigos)

    # KPIs
    tot_multas = dff["MultasImpuestas"].sum() if "MultasImpuestas" in dff.columns else np.nan
    tot_importe = dff["ImporteImpuesto"].sum() if "ImporteImpuesto" in dff.columns else np.nan
    tot_reinc = dff["ReincidenciaImpuesta"].sum() if "ReincidenciaImpuesta" in dff.columns else np.nan

    porc_reinc = (tot_reinc / tot_multas * 100) if (pd.notna(tot_reinc) and pd.notna(tot_multas) and tot_multas>0) else 0.0
    n_codigos = dff["CodigoFalta"].nunique() if "CodigoFalta" in dff.columns else 0

    k1 = f"{int(tot_multas):,}".replace(",", " ") if pd.notna(tot_multas) else "—"
    k2 = f"S/ {tot_importe:,.2f}".replace(",", " ") if pd.notna(tot_importe) else "—"
    k3 = f"{porc_reinc:,.2f}%".replace(",", " ")
    k4 = f"{n_codigos:,}".replace(",", " ")

    # Gráfico: Multas por Mes
    if {"Mes","MultasImpuestas"}.issubset(dff.columns):
        g1 = dff.groupby("Mes", dropna=True)["MultasImpuestas"].sum().reset_index()
        fig_mes = px.bar(g1, x="Mes", y="MultasImpuestas", title="Multas impuestas por mes")
    else:
        fig_mes = px.scatter(title="Multas impuestas por mes (faltan columnas)")

    # Gráfico: por Nivel de Gravedad
    if {"NivelGravedad","MultasImpuestas"}.issubset(dff.columns):
        g2 = dff.groupby("NivelGravedad", dropna=True)["MultasImpuestas"].sum().reset_index()
        fig_grav = px.bar(g2, x="NivelGravedad", y="MultasImpuestas", title="Multas por nivel de gravedad")
    else:
        fig_grav = px.scatter(title="Multas por nivel de gravedad (faltan columnas)")

    # Gráfico: Top 10 Códigos por Multas
    if {"CodigoFalta","MultasImpuestas"}.issubset(dff.columns):
        g3 = dff.groupby("CodigoFalta", dropna=True)["MultasImpuestas"].sum().reset_index()
        g3 = g3.sort_values("MultasImpuestas", ascending=False).head(10)
        fig_top = px.bar(g3, x="CodigoFalta", y="MultasImpuestas", title="Top 10 códigos de falta por multas impuestas")
    else:
        fig_top = px.scatter(title="Top códigos (faltan columnas)")

    # Gráfico: Tendencia en el tiempo (Periodo)
    if "Periodo" in dff.columns and "MultasImpuestas" in dff.columns:
        g4 = dff.dropna(subset=["Periodo"]).groupby("Periodo")["MultasImpuestas"].sum().reset_index()
        fig_tend = px.line(g4, x="Periodo", y="MultasImpuestas", title="Tendencia temporal de multas impuestas")
    else:
        fig_tend = px.scatter(title="Tendencia temporal (faltan columnas)")

    # Tabla
    cols = [{"name": c, "id": c} for c in dff.columns]
    data = dff.head(5000).to_dict("records")  # limite por performance
    info = f"Mostrando {len(data)} filas (de {len(dff):,})".replace(",", " ")

    return k1, k2, k3, k4, fig_mes, fig_grav, fig_top, fig_tend, cols, data, info

# ----------------------------
# 4) Main
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)


