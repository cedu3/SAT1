# File: multas-prediction/app.py

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# =======================
# 1) CARGA DE DATOS
# =======================
def load_sat_data():
    candidates = [
        "MultasLimpias.xlsx",
        "MultasLimpias.csv",
        os.path.join("data", "MultasLimpias.xlsx"),
        os.path.join("data", "MultasLimpias.csv"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError("No encuentro MultasLimpias.(xlsx/csv) en la carpeta actual ni en ./data/.")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".xlsx":
        df = pd.read_excel(path, engine="openpyxl")
        print(f"[OK] Cargado Excel: {path}")
        return df
    for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        for sep in [",", ";", "\t", "|"]:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, engine="python")
                print(f"[OK] Cargado CSV: {path} (encoding={enc}, sep='{sep}')")
                return df
            except Exception:
                continue
    raise RuntimeError(f"No pude leer el CSV {path}. Revisa encoding/separador.")

df = load_sat_data()
df.columns = [c.strip() for c in df.columns]

# Tipos
for col in ["Anio", "Mes", "MultasImpuestas", "ImporteImpuesto"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# =======================
# 2) ENCODERS
# =======================
df["CodigoFalta"]   = df["CodigoFalta"].astype(str)
df["TipoFormato"]   = df["TipoFormato"].astype(str)
df["NivelGravedad"] = df["NivelGravedad"].astype(str)

le_codigo   = LabelEncoder()
df["CodigoFaltaCod"]   = le_codigo.fit_transform(df["CodigoFalta"])
le_formato  = LabelEncoder()
df["TipoFormatoCod"]   = le_formato.fit_transform(df["TipoFormato"])
le_gravedad = LabelEncoder()
df["NivelGravedadCod"] = le_gravedad.fit_transform(df["NivelGravedad"])

target_col = "MultasImpuestas"

# =======================
# 3) AGREGACIÓN MENSUAL (SUMA de MultasImpuestas) — Entrena con 2022–2024
# =======================
base = df.dropna(subset=["Anio","Mes","NivelGravedadCod","TipoFormatoCod",target_col]).copy()
base = base[base["Anio"].isin([2022, 2023, 2024])]

group_cols_A = ["Anio","Mes","NivelGravedadCod","TipoFormatoCod","CodigoFaltaCod"]
group_cols_B = ["Anio","Mes","NivelGravedadCod","TipoFormatoCod"]

monthly_A = (base.dropna(subset=["CodigoFaltaCod"])
                .groupby(group_cols_A, as_index=False)
                .agg(Suma=(target_col,"sum")))
monthly_B = (base.groupby(group_cols_B, as_index=False)
                .agg(Suma=(target_col,"sum")))

# =======================
# 4) MODELOS (Random Forest sobre SUMAS mensuales)
# =======================
rf_A = RandomForestRegressor(n_estimators=500, random_state=42)
rf_A.fit(monthly_A[group_cols_A].values, monthly_A["Suma"].values)

rf_B = RandomForestRegressor(n_estimators=500, random_state=42)
rf_B.fit(monthly_B[group_cols_B].values, monthly_B["Suma"].values)

# =======================
# 5) APP DASH (UI)
# =======================
app = dash.Dash(__name__)
app.title = "Predicción de Multas (Suma de MultasImpuestas) — 2025"

# Opciones
opt_mes  = [{"label":f"{m:02d}","value":int(m)} for m in sorted(base["Mes"].dropna().astype(int).unique())]
opt_grav = [{"label":g,"value":int(le_gravedad.transform([g])[0])}
            for g in sorted(base["NivelGravedad"].astype(str).unique())]
opt_form = [{"label":f,"value":int(le_formato.transform([f])[0])}
            for f in sorted(base["TipoFormato"].astype(str).unique())]
opt_code = [{"label":c,"value":int(le_codigo.transform([c])[0])}
            for c in sorted(base["CodigoFalta"].astype(str).unique())]

app.layout = html.Div([
    html.H1("Predicción 2025 — Suma de MultasImpuestas", style={"textAlign":"center"}),

    html.Div([
        html.H3("Filtros"),

        # Meses
        html.Div([
            html.Label("Mes(es)"),
            html.Button("Seleccionar todo", id="btn_all_meses", n_clicks=0, style={"marginLeft":"8px"})
        ], style={"display":"flex","alignItems":"center","gap":"6px"}),
        dcc.Dropdown(id="meses", options=opt_mes, value=[1], multi=True,
                     placeholder="Selecciona uno o varios meses"),
        html.Br(),

        # Nivel de Gravedad
        html.Div([
            html.Label("Nivel de Gravedad (multi)"),
            html.Button("Seleccionar todo", id="btn_all_grav", n_clicks=0, style={"marginLeft":"8px"})
        ], style={"display":"flex","alignItems":"center","gap":"6px"}),
        dcc.Dropdown(id="gravedad", options=opt_grav, multi=True,
                     placeholder="Selecciona una o varias gravedades"),
        html.Br(),

        # Tipo de Formato
        html.Div([
            html.Label("Tipo de Formato (multi)"),
            html.Button("Seleccionar todo", id="btn_all_form", n_clicks=0, style={"marginLeft":"8px"})
        ], style={"display":"flex","alignItems":"center","gap":"6px"}),
        dcc.Dropdown(id="formato", options=opt_form, multi=True,
                     placeholder="Selecciona uno o varios formatos"),
        html.Br(),

        # Código de falta (opcional)
        html.Div([
            html.Label("Código de Falta (opcional, multi)"),
            html.Button("Seleccionar todo", id="btn_all_cod", n_clicks=0, style={"marginLeft":"8px"})
        ], style={"display":"flex","alignItems":"center","gap":"6px"}),
        dcc.Dropdown(id="codigo", options=opt_code, multi=True,
                     placeholder="(Opcional) Selecciona uno o varios códigos"),
        html.Br(),

        html.Button("Predecir 2025", id="btn_pred", n_clicks=0),
        html.Button("Descargar CSV", id="btn_dl", n_clicks=0, style={"marginLeft":"12px"}),
        dcc.Download(id="download_csv"),

        html.Div(id="pred_out", style={"marginTop":"16px","fontSize":"18px","fontWeight":"bold"}),
    ], style={"width":"460px","display":"inline-block","verticalAlign":"top","padding":"20px"}),

    html.Div([
        html.H3("Gráficos"),
        dcc.Graph(id="fig_pred_barras"),        # Predicción 2025 por mes
        dcc.Graph(id="fig_comp_2024_vs_2025")   # Evolución mensual 2024 vs Pred 2025 (filtrado)
    ], style={"width":"calc(100% - 520px)","display":"inline-block","padding":"20px","verticalAlign":"top"})
])

# =======================
# 6) CALLBACKS: Seleccionar todo
# =======================
@app.callback(Output("meses","value"), Input("btn_all_meses","n_clicks"), prevent_initial_call=True)
def sel_all_meses(n):
    return [opt["value"] for opt in opt_mes]

@app.callback(Output("gravedad","value"), Input("btn_all_grav","n_clicks"), prevent_initial_call=True)
def sel_all_grav(n):
    return [opt["value"] for opt in opt_grav]

@app.callback(Output("formato","value"), Input("btn_all_form","n_clicks"), prevent_initial_call=True)
def sel_all_form(n):
    return [opt["value"] for opt in opt_form]

@app.callback(Output("codigo","value"), Input("btn_all_cod","n_clicks"), prevent_initial_call=True)
def sel_all_cod(n):
    return [opt["value"] for opt in opt_code]

# =======================
# 7) UTILIDADES
# =======================
def _pred_2025_por_mes(meses, gravedades, formatos, codigos):
    """Predice SUMA de MultasImpuestas por mes para 2025 sumando sobre
    todas las combinaciones seleccionadas (gravedad × formato × código?)."""
    anio_pred = 2025
    usa_codigos = bool(codigos)

    pred_por_mes = []
    for m in meses:
        total_mes = 0.0
        for g in gravedades:
            for f in formatos:
                if usa_codigos:
                    for c in codigos:
                        row = [anio_pred, m, g, f, c]
                        total_mes += rf_A.predict([row])[0]
                else:
                    row = [anio_pred, m, g, f]
                    total_mes += rf_B.predict([row])[0]
        pred_por_mes.append(max(0.0, float(total_mes)))
    return pred_por_mes

def _historico_2024_filtrado_por_mes(meses, gravedades, formatos, codigos):
    """Suma mensual 2024 con filtros (gravedad, formato, código opcional)."""
    f = (base["Anio"]==2024) & (base["NivelGravedadCod"].isin(gravedades)) & (base["TipoFormatoCod"].isin(formatos))
    if codigos:
        f &= base["CodigoFaltaCod"].isin(codigos)
    hist = base.loc[f, ["Mes", target_col]]
    if hist.empty:
        return [0.0 for _ in meses]
    sums = hist.groupby("Mes", as_index=False).agg(Suma=(target_col,"sum"))
    mapa = dict(zip(sums["Mes"].astype(int), sums["Suma"].astype(float)))
    return [float(mapa.get(int(m), 0.0)) for m in meses]

# =======================
# 8) CALLBACK PRINCIPAL
# =======================
@app.callback(
    Output("pred_out","children"),
    Output("fig_pred_barras","figure"),
    Output("fig_comp_2024_vs_2025","figure"),
    Input("btn_pred","n_clicks"),
    State("meses","value"),
    State("gravedad","value"),
    State("formato","value"),
    State("codigo","value"),
)
def predecir_y_graficar(n_clicks, meses, gravedades, formatos, codigos):
    import plotly.graph_objects as go

    if not n_clicks:
        return ("", go.Figure(), go.Figure())

    # Defaults si vienen vacíos
    if not meses:
        meses = [opt["value"] for opt in opt_mes]
    if not gravedades:
        gravedades = [opt["value"] for opt in opt_grav]
    if not formatos:
        formatos = [opt["value"] for opt in opt_form]

    # Asegurar enteros y orden
    meses = sorted(int(m) for m in (meses if isinstance(meses, list) else [meses]))

    # ------ Predicción 2025 (SUMA sobre combinaciones) ------
    pred_vals = _pred_2025_por_mes(meses, gravedades, formatos, codigos)
    pred_int = [int(round(v)) for v in pred_vals]
    total_pred = int(sum(pred_int))

    # ------ Histórico 2024 (FILTRADO) ------
    hist_2024_vals = _historico_2024_filtrado_por_mes(meses, gravedades, formatos, codigos)
    hist_2024_int = [int(round(v)) for v in hist_2024_vals]
    total_hist_2024 = int(sum(hist_2024_int))

    # ------ Texto resumen ------
    pares = [f"2025-{m:02d}: {pv:,}".replace(",", " ") for m, pv in zip(meses, pred_int)]
    texto = "Predicciones 2025 (suma por mes) → " + " | ".join(pares)
    texto += f" | Total 2025: {total_pred:,}".replace(",", " ")
    texto += f" | Total 2024 (filtrado): {total_hist_2024:,}".replace(",", " ")

    # ------ Figura 1: Predicción 2025 ------
    df_pred = pd.DataFrame({"Mes": meses, "Prediccion2025": pred_int})
    fig1 = px.bar(df_pred, x="Mes", y="Prediccion2025",
                  title="Predicción 2025 — Suma de MultasImpuestas por Mes",
                  text="Prediccion2025")
    fig1.update_traces(textposition="outside")

    # ------ Figura 2: Evolución mensual 2024 vs Pred 2025 (FILTRADO) ------
    df_comp = pd.DataFrame({
        "Mes": meses,
        "Historico_2024": hist_2024_int,
        "Prediccion_2025": pred_int
    })
    fig2 = px.bar(df_comp, x="Mes", y=["Historico_2024","Prediccion_2025"],
                  title="Evolución mensual: 2024 (filtrado) vs Predicción 2025",
                  barmode="group",
                  labels={"value":"Suma", "variable":"Serie"})
    fig2.update_traces(texttemplate="%{y}", textposition="outside")

    return (texto, fig1, fig2)

# =======================
# 9) DESCARGA CSV
# =======================
@app.callback(
    Output("download_csv","data"),
    Input("btn_dl","n_clicks"),
    State("meses","value"),
    State("gravedad","value"),
    State("formato","value"),
    State("codigo","value"),
    prevent_initial_call=True
)
def descargar_csv(n_clicks, meses, gravedades, formatos, codigos):
    if not meses:
        meses = [opt["value"] for opt in opt_mes]
    if not gravedades:
        gravedades = [opt["value"] for opt in opt_grav]
    if not formatos:
        formatos = [opt["value"] for opt in opt_form]
    meses = sorted(int(m) for m in (meses if isinstance(meses, list) else [meses]))

    pred = _pred_2025_por_mes(meses, gravedades, formatos, codigos)
    hist24 = _historico_2024_filtrado_por_mes(meses, gravedades, formatos, codigos)

    df_out = pd.DataFrame({
        "Mes": meses,
        "Historico_2024": np.round(hist24, 0).astype(int),
        "Prediccion_2025": np.round(pred, 0).astype(int),
        "Total_2024_seleccion": int(np.round(np.sum(hist24), 0)),
        "Total_2025_pred_seleccion": int(np.round(np.sum(pred), 0)),
    })
    return dcc.send_data_frame(df_out.to_csv, filename="evolucion_y_totales_2024_vs_pred_2025.csv", index=False)

# =======================
# 10) MAIN
# =======================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=True, host="127.0.0.1", port=port)