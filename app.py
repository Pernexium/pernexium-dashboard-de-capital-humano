######################################### LIBRERIAS ############################################

import io
import re
import os
import pytz
import json
import boto3
import requests
import numpy as np
import unicodedata
import pandas as pd
import datetime as dt
from io import BytesIO
from markupsafe import Markup 
from dotenv import load_dotenv
from datetime import datetime, timedelta
from google.oauth2 import service_account
from googleapiclient.discovery import build
from flask import Flask, render_template, request 

########################################## AMBIENTE ############################################

load_dotenv()
s3 = boto3.client("s3")

######################################## CARGA VARIABLES #######################################

S3_SA_KEY = os.getenv("S3_SA_KEY")
SHEET_ID  = os.getenv("SHEET_ID")
SCOPES    = json.loads(os.getenv("SCOPES"))
S3_BUCKET = os.getenv("S3_BUCKET")
ONE_DRIVE = os.getenv("ONE_DRIVE")

################################# ACCESO A GOOGLE SHEETS #######################################

def get_service_account_credentials():
    obj       = s3.get_object(Bucket=S3_BUCKET, Key=S3_SA_KEY)
    sa_info   = json.loads(obj["Body"].read().decode())
    creds_sa  = service_account.Credentials.from_service_account_info(sa_info, scopes=SCOPES)
    return creds_sa

def build_sheets_service():
    creds = get_service_account_credentials()
    return build("sheets", "v4", credentials=creds, cache_discovery=False)

def fetch_sheet_data(service, sheet_name):
    rng  = f"{sheet_name}!A1:BT"
    vals = (service.spreadsheets()
                  .values()
                  .get(spreadsheetId=SHEET_ID, range=rng)
                  .execute()
                  .get("values", []))

    if not vals:
        return pd.DataFrame()

    headers, *data = vals                                                                 
    max_cols = max(len(headers), *(len(r) for r in data)) if data else len(headers)
    headers += [f"Col_{i}" for i in range(len(headers), max_cols)]                        
    padded  = [row + [None]*(max_cols - len(row)) for row in data]                        
    return pd.DataFrame(padded, columns=[h.strip() for h in headers])

######################################### FUNCIONES AUXILIARES #########################################

def moneda_es(x):
    try:
        return '$' + format(x, ',.2f').replace(',', 'X').replace('.', ',').replace('X', '.')
    except Exception:
        return '$0,00'

############################################# DASHBOARD ##############################################

app = Flask(__name__)
@app.route("/", methods=["GET"])

def index():
    
    ################################# BASES DE DATOS ##################################
    
    ################################# BASES DE DATOS ##################################

    # 1) Filtro de campaña desde la URL (?campania=...)
    campania_param = request.args.get("campania", default="", type=str).strip()
    campania_seleccionada = campania_param or None

    # 2) Cargar datos completos (sin filtrar)
    response = requests.get(ONE_DRIVE, allow_redirects=True)
    response.raise_for_status()
    df_reclutamiento_full = pd.read_excel(BytesIO(response.content), sheet_name="Reclutamiento")

    service = build_sheets_service()
    df_plantilla_activa_full     = fetch_sheet_data(service, "PLANTILLA AJUSTE")
    df_plantilla_autorizada_full = fetch_sheet_data(service, "PLANTILLA AUTORIZADA")
    df_plantilla_bajas_full      = fetch_sheet_data(service, "PLANTILLA BAJAS")

    # 3) Catálogo de campañas para el combo (sin filtrar)
    campanias_reclut = set(
        df_reclutamiento_full.get("Campaña", pd.Series(dtype=str))
        .dropna()
        .astype(str)
        .str.strip()
    )
    campanias_sheets = set(
        df_plantilla_activa_full.get("CAMPAÑA", pd.Series(dtype=str))
        .dropna()
        .astype(str)
        .str.strip()
    )
    campanias = sorted(campanias_reclut.union(campanias_sheets))

    # 4) Copias filtradas (las que usará el resto del dashboard)
    df_reclutamiento      = df_reclutamiento_full.copy()
    df_plantilla_activa   = df_plantilla_activa_full.copy()
    df_plantilla_autorizada = df_plantilla_autorizada_full.copy()
    df_plantilla_bajas    = df_plantilla_bajas_full.copy()

    if campania_seleccionada:
        # OneDrive: columna "Campaña"
        if "Campaña" in df_reclutamiento.columns:
            df_reclutamiento = df_reclutamiento[
                df_reclutamiento["Campaña"]
                    .astype(str)
                    .str.strip()
                    == campania_seleccionada
            ]

        # Sheets: columna "CAMPAÑA"
        if "CAMPAÑA" in df_plantilla_activa.columns:
            df_plantilla_activa = df_plantilla_activa[
                df_plantilla_activa["CAMPAÑA"]
                    .astype(str)
                    .str.strip()
                    == campania_seleccionada
            ]

        if "CAMPAÑA" in df_plantilla_autorizada.columns:
            df_plantilla_autorizada = df_plantilla_autorizada[
                df_plantilla_autorizada["CAMPAÑA"]
                    .astype(str)
                    .str.strip()
                    == campania_seleccionada
            ]

        if "CAMPAÑA" in df_plantilla_bajas.columns:
            df_plantilla_bajas = df_plantilla_bajas[
                df_plantilla_bajas["CAMPAÑA"]
                    .astype(str)
                    .str.strip()
                    == campania_seleccionada
            ]


    ##################################### KPIs ######################################
    
    mapa_meses = {
        1: 'ENERO', 2: 'FEBRERO', 3: 'MARZO', 4: 'ABRIL',
        5: 'MAYO', 6: 'JUNIO', 7: 'JULIO', 8: 'AGOSTO',
        9: 'SEPTIEMBRE', 10: 'OCTUBRE', 11: 'NOVIEMBRE', 12: 'DICIEMBRE'
    }
    mes_actual_num = dt.datetime.today().month
    mes_actual_nombre = mapa_meses[mes_actual_num]
    # --- Plantilla autorizada ---
    df_plantilla_autorizada_actual = df_plantilla_autorizada.copy()
    df_plantilla_autorizada_actual['MES'] = (
        df_plantilla_autorizada_actual['MES']
        .astype(str)
        .str.upper()
        .str.strip()
    )
    df_plantilla_autorizada_actual = df_plantilla_autorizada_actual[
        df_plantilla_autorizada_actual['MES'] == mes_actual_nombre
    ]
    df_plantilla_autorizada_actual['PERSONAL'] = pd.to_numeric(
        df_plantilla_autorizada_actual['PERSONAL'], errors='coerce'
    )
    sum_plantilla_general_autorizada = int(df_plantilla_autorizada_actual['PERSONAL'].sum())
    sum_plantilla_operativa_autorizada = int(
        df_plantilla_autorizada_actual
        .loc[df_plantilla_autorizada_actual['AREA'] == "OPERACION", 'PERSONAL']
        .sum()
    )
    sum_plantilla_administrativa_autorizada = int(
        df_plantilla_autorizada_actual
        .loc[df_plantilla_autorizada_actual['AREA'] == "ESTRUCTURA", 'PERSONAL']
        .sum()
    )
    # --- Género ---
    genero = df_plantilla_activa['GENERO']
    genero_limpio = (
        genero.dropna()
        .astype(str)
        .str.strip()
    )
    genero_limpio = genero_limpio[genero_limpio != ""]
    conteo_GENERO = genero_limpio.value_counts()
    porcentaje_GENERO = genero_limpio.value_counts(normalize=True).mul(100).round(2)
    resumen_GENERO = pd.DataFrame({
        'Cantidad': conteo_GENERO,
        'Porcentaje (%)': porcentaje_GENERO
    })
    genero_cantidad_dict = resumen_GENERO['Cantidad'].to_dict()
    genero_porcentaje_dict = resumen_GENERO['Porcentaje (%)'].to_dict()
    # --- Nómina ---
    df_nomina = df_plantilla_activa.copy()
    df_nomina = df_nomina[df_nomina['SUELDO MENSUAL'].notna()]

    df_nomina['SUELDO MENSUAL'] = (
        df_nomina['SUELDO MENSUAL'].astype(str)
        .str.replace(r'[^\d,.\-]', '', regex=True)   
        .str.replace('.', '', regex=False)           
        .str.replace(',', '.', regex=False)         
        .pipe(pd.to_numeric, errors='coerce')
    )
    nomina_por_area = (
        df_nomina.groupby('AREA', dropna=False)['SUELDO MENSUAL']
        .sum()
        .reset_index(name='TOTAL NOMINA')
    )
    total_general_nomina = float(df_nomina['SUELDO MENSUAL'].sum())
    total_general_nomina_fmt = moneda_es(total_general_nomina)
    nomina_formateada = nomina_por_area.copy()
    nomina_formateada['TOTAL NOMINA'] = nomina_formateada['TOTAL NOMINA'].map(moneda_es)
    nomina_por_area_dict = nomina_por_area.set_index('AREA')['TOTAL NOMINA'].to_dict()
    nomina_por_area_fmt_dict = nomina_formateada.set_index('AREA')['TOTAL NOMINA'].to_dict()
    nomina_estructura = float(nomina_por_area_dict.get('ESTRUCTURA', 0.0))
    nomina_operacion = float(nomina_por_area_dict.get('OPERACION', 0.0))
    nomina_estructura_fmt = nomina_por_area_fmt_dict.get('ESTRUCTURA', moneda_es(0))
    nomina_operacion_fmt = nomina_por_area_fmt_dict.get('OPERACION', moneda_es(0))
    # --- Contratos ---
    df_contratos = df_plantilla_activa.copy()
    df_contratos['CONTRATOS'] = (
        pd.to_numeric(
            df_contratos['CONTRATOS']
            .astype(str)
            .str.replace(r'[^0-9.-]', '', regex=True),
            errors='coerce'
        ).fillna(0)
    )
    df_contratos['AREA'] = (
        df_contratos['AREA']
        .astype(str)
        .str.replace('\xa0', ' ')
        .str.strip()
        .str.replace(r'\s+', ' ', regex=True)
    )
    df_contratos.loc[df_contratos['AREA'].isin(['', 'nan', 'NaN', 'None']), 'AREA'] = pd.NA
    contratos_por_area = (
        df_contratos.dropna(subset=['AREA'])
        .groupby('AREA', observed=True)['CONTRATOS']
        .sum()
        .reset_index()
        .sort_values('CONTRATOS', ascending=False)
    )
    contratos_por_area_dict = contratos_por_area.set_index('AREA')['CONTRATOS'].to_dict()
    contratos_total = int(contratos_por_area['CONTRATOS'].sum())
    contratos_estructura = int(contratos_por_area_dict.get('ESTRUCTURA', 0))
    contratos_operacion = int(contratos_por_area_dict.get('OPERACION', 0))

    ############################################# GRAFICAS ##############################################
    
    # --- FUNEL DE RECLUTAMIENTO ---
    def limpiar_nombre(nombre: str) -> str:
        if pd.isna(nombre):
            return ""
        nombre = str(nombre).strip()
        nombre = unicodedata.normalize("NFKD", nombre)
        nombre = "".join(c for c in nombre if not unicodedata.combining(c))
        nombre = nombre.replace("ñ", "n").replace("Ñ", "n")
        nombre = nombre.lower()
        nombre = re.sub(r"[^a-z\s]", "", nombre)
        nombre = re.sub(r"\s+", " ", nombre).strip()
        return nombre
    MESES_ES = ["ene", "feb", "mar", "abr", "may", "jun", 
                "jul", "ago", "sep", "oct", "nov", "dic"]
    def normalizar_mes(valor):
        if pd.isna(valor):
            return np.nan
        if isinstance(valor, datetime):
            dt_val = valor
        else:
            s = str(valor).strip().lower()
            try:
                return s
            except Exception:
                pass
            s_fix = s
            parts = s.split("/")
            if len(parts) == 3 and len(parts[2]) > 4:
                s_fix = parts[0] + "/" + parts[1] + "/" + parts[2][:4]
            try:
                dt_val = pd.to_datetime(s_fix, dayfirst=True, errors="coerce")
            except Exception:
                return np.nan
            if pd.isna(dt_val):
                return np.nan    
        mes = MESES_ES[dt_val.month - 1]
        return f"{mes}-{str(dt_val.year)[-2:]}"
    data_reclutamiento = df_reclutamiento
    data_reclutamiento = data_reclutamiento.dropna(axis=0, how='all')
    data_reclutamiento.columns = data_reclutamiento.iloc[3, :]
    data_reclutamiento = data_reclutamiento[4:]
    data_reclutamiento = data_reclutamiento.reset_index(drop=True)
    data_reclutamiento = data_reclutamiento[1:]
    data_reclutamiento.columns = [
        col.strip()
           .replace(" ", "_")
           .replace(".", "")
           .replace(":", "")
           .lower()
           .replace("ó", "o")
           .replace("í", "i")
           .replace("é", "e")
           .replace("ú", "u")
           .replace("ñ", "n")
           .replace("á", "a")
        for col in data_reclutamiento.columns
    ]
    data_reclutamiento.rename(columns={"¿es_viable?4": "es_viable_tecnica"}, inplace=True)
    data_reclutamiento.rename(columns={"¿es_viable?": "es_viable_psicometrica"}, inplace=True)
    data_reclutamiento.columns = [
        col.replace("?", "").replace("¿", "").replace("4", "").replace("ii_~_iii_", "")
        for col in data_reclutamiento.columns
    ]
    data_reclutamiento.nombre_candidato = data_reclutamiento.nombre_candidato.apply(limpiar_nombre)
    data_reclutamiento.mes = data_reclutamiento.mes.apply(normalizar_mes)
    data_reclutamiento.iv_realiza_psicometrias = data_reclutamiento.iv_realiza_psicometrias.apply(
        lambda x: x.strip().lower().capitalize() if isinstance(x, str) else x
    )
    data_reclutamiento.campana = data_reclutamiento.campana.apply(
        lambda x: x.strip().lower().capitalize() if isinstance(x, str) else x
    )
    data_reclutamiento.campana = data_reclutamiento.campana.apply(
        lambda x: x.replace("Fmp (posiciones)", "Fmp posiciones").replace("Fmp posiciones", "Fmp (posiciones)")
        if isinstance(x, str) else x
    )
    data_reclutamiento.se_asigna_a_campana = data_reclutamiento.se_asigna_a_campana.apply(
        lambda x: x.strip().lower().capitalize() if isinstance(x, str) else x
    )
    data_reclutamiento.se_asigna_a_campana = data_reclutamiento.se_asigna_a_campana.apply(
        lambda x: x.replace("Fmp (posiciones)", "Fmp posiciones").replace("Fmp posiciones", "Fmp (posiciones)")
        if isinstance(x, str) else x
    )
    data_reclutamiento['fecha_publicacion'] = pd.to_datetime(
        data_reclutamiento['fecha_publicacion'], errors="coerce"
    )
    data_reclutamiento.escolaridad = data_reclutamiento.escolaridad.apply(
        lambda x: x.strip().lower()
        .replace("bachillerato", "preparatoria")
        .replace("preparatoria terminado", "preparatoria terminada")
        .replace("preparatoria trunco", "preparatoria trunca")
        .replace("preparaatoria", "preparatoria")
        .replace("prepa ", "preparatoria ")
        if isinstance(x, str) else x
    )
    data_reclutamiento.calificacion = data_reclutamiento.calificacion.apply(
        lambda x: str(x).strip().replace("Baja", "7")
    )
    data_reclutamiento.fecha_de_ingreso = pd.to_datetime(data_reclutamiento.fecha_de_ingreso, errors="coerce")
    data_reclutamiento.fecha_de_ingreso = data_reclutamiento.fecha_de_ingreso.dt.strftime("%Y-%m-%d")
    data_reclutamiento.fecha_de_entrevista = data_reclutamiento.fecha_de_entrevista.apply(
        lambda x: np.nan if x == '-' else x
    )
    data_reclutamiento.fecha_de_ingreso = data_reclutamiento.fecha_de_ingreso.apply(
        lambda x: np.nan if x == '-' else x
    )
    data_reclutamiento.fecha_de_capacitacion = pd.to_datetime(
        data_reclutamiento.fecha_de_capacitacion, errors="coerce"
    )
    data_reclutamiento.fecha_de_capacitacion = data_reclutamiento.fecha_de_capacitacion.apply(
        lambda x: np.nan if x == '-' else x
    )
    data_reclutamiento['bin_aceptado'] = data_reclutamiento['aceptada'].apply(
        lambda x: 1 if str(x).strip().lower() == 'si' else 0
    )
    data_reclutamiento['bin_agenda_entrevista'] = data_reclutamiento['fecha_de_entrevista'].isna().apply(
        lambda x: 0 if x else 1
    )
    data_reclutamiento['bin_ingreso'] = data_reclutamiento['fecha_de_ingreso'].isna().apply(
        lambda x: 0 if x else 1
    )
    data_reclutamiento['bin_asiste_entrevista'] = data_reclutamiento['motivo'].apply(
        lambda x: 1 if str(x).strip() in
        ['Aceptado', 'Rechazado', 'No aceptado', 'Fallo en Role-Play', 'Rechaza la vacante'] else 0
    )
    data_reclutamiento['monto_pagado_distribuido'] = 0.0
    for folio in data_reclutamiento.folio.unique():
        if folio is None or pd.isna(folio):
            continue
        df_tmp = data_reclutamiento.query("folio == @folio")
        costo_lead = df_tmp.monto_pagado.iloc[0] / len(df_tmp)
        data_reclutamiento.loc[data_reclutamiento.folio == folio, 'monto_pagado_distribuido'] = costo_lead
    df_funnel_mes = (
        data_reclutamiento
        .dropna(subset=['fecha_publicacion'])
        .groupby(pd.Grouper(key='fecha_publicacion', freq='MS'))
        .agg({
            'fecha_publicacion': 'count',       
            'bin_agenda_entrevista': 'sum',
            'bin_asiste_entrevista': 'sum',
            'bin_aceptado': 'sum',
            'bin_ingreso': 'sum'
        })
        .rename(columns={'fecha_publicacion': 'leads_generados'})
        .reset_index()
        .sort_values('fecha_publicacion')
    )
    df_funnel_ultimos_3 = df_funnel_mes.tail(3)
    funnel_labels = [
        "Leads en Facebook",
        "Agenda Entrevista",
        "Asistieron a Entrevista",
        "Aceptados",
        "Ingresos a Operación",
    ]
    MESES_ES_CORTO = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                      "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    funnel_datasets = []
    for _, row in df_funnel_ultimos_3.iterrows():
        dt_mes = row['fecha_publicacion']
        mes_label = f"{MESES_ES_CORTO[dt_mes.month - 1]} {dt_mes.year}"
        funnel_datasets.append({
            "label": mes_label,
            "data": [
                int(row['leads_generados']),
                int(row['bin_agenda_entrevista']),
                int(row['bin_asiste_entrevista']),
                int(row['bin_aceptado']),
                int(row['bin_ingreso']),
            ],
        })
    # --- COBERTURA DE CARTERA - TABLA Y GRAFICO ---
    df_act = df_plantilla_activa.copy()
    df_act['FECHA_INGRESO'] = pd.to_datetime(df_act['FECHA DE INGRESO'],dayfirst=True,errors='coerce')
    df_act = df_act.dropna(subset=['FECHA_INGRESO'])
    df_act['FECHA_BAJA'] = pd.NaT
    df_act = df_act[['CAMPAÑA', 'FECHA_INGRESO', 'FECHA_BAJA']]
    df_baj = df_plantilla_bajas.copy()
    df_baj['FECHA_INGRESO'] = pd.to_datetime(df_baj['FECHA DE INGRESO'],dayfirst=True,errors='coerce')
    df_baj['FECHA_BAJA'] = pd.to_datetime(df_baj['BAJA'],dayfirst=True,errors='coerce')
    df_baj = df_baj.dropna(subset=['FECHA_INGRESO'])
    df_baj = df_baj[['CAMPAÑA', 'FECHA_INGRESO', 'FECHA_BAJA']]
    df_personal = pd.concat([df_act, df_baj], ignore_index=True)
    month_map = {
        'ENERO': 1, 'FEBRERO': 2, 'MARZO': 3, 'ABRIL': 4, 'MAYO': 5, 'JUNIO': 6,
        'JULIO': 7, 'AGOSTO': 8, 'SEPTIEMBRE': 9, 'OCTUBRE': 10, 'NOVIEMBRE': 11, 'DICIEMBRE': 12
    }
    df_aut = df_plantilla_autorizada.copy()
    df_aut['MES_UPPER'] = df_aut['MES'].astype(str).str.strip().str.upper()
    df_aut['MES_NUM'] = df_aut['MES_UPPER'].map(month_map)
    df_aut['AÑO_NUM'] = pd.to_numeric(df_aut['AÑO'], errors='coerce')
    df_aut['PERSONAL'] = pd.to_numeric(df_aut['PERSONAL'], errors='coerce')
    df_aut = df_aut.dropna(subset=['MES_NUM', 'AÑO_NUM', 'PERSONAL'])
    df_aut['MES_NUM'] = df_aut['MES_NUM'].astype(int)
    df_aut['AÑO_NUM'] = df_aut['AÑO_NUM'].astype(int)
    df_aut['AÑO'] = df_aut['AÑO_NUM'].astype(int)
    df_aut['PERSONAL'] = df_aut['PERSONAL'].astype(int)
    df_aut = (df_aut.groupby(['CAMPAÑA', 'MES', 'AÑO', 'MES_NUM', 'AÑO_NUM'], as_index=False)['PERSONAL'].sum())
    df_aut['FECHA_CORTE'] = pd.to_datetime({
        'year': df_aut['AÑO_NUM'],
        'month': df_aut['MES_NUM'],
        'day': 1
    }) + pd.offsets.MonthEnd(0)
    df_aut['_key'] = 1
    df_personal['_key'] = 1
    tmp = df_aut.merge(df_personal, on=['CAMPAÑA', '_key'], how='left')
    mask = (
        (tmp['FECHA_INGRESO'] <= tmp['FECHA_CORTE']) &
        (tmp['FECHA_BAJA'].isna() | (tmp['FECHA_BAJA'] > tmp['FECHA_CORTE']))
    )
    tmp_activas = tmp[mask]
    total_activos = (
        tmp_activas
        .groupby(['CAMPAÑA', 'MES', 'AÑO'], as_index=False)
        .size()
        .rename(columns={'size': 'TOTAL ACTIVOS'})
    )
    resultado = df_aut.merge(total_activos, on=['CAMPAÑA', 'MES', 'AÑO'], how='left')
    resultado['TOTAL ACTIVOS'] = resultado['TOTAL ACTIVOS'].fillna(0).astype(int)
    resultado = resultado.rename(columns={'PERSONAL': 'PLANTILLA AUTORIZADA'})
    resultado['PERSONAL FALTANTE'] = (
        resultado['PLANTILLA AUTORIZADA'] - resultado['TOTAL ACTIVOS']
    )#.clip(lower=0)
    df_cobertura_de_cartera = resultado[[
        'CAMPAÑA',
        'PLANTILLA AUTORIZADA',
        'TOTAL ACTIVOS',
        'PERSONAL FALTANTE',
        'MES',
        'AÑO'
    ]]
    df = df_cobertura_de_cartera.copy()
    df['MES_NUM'] = df['MES'].map(month_map)
    monthly = (
        df.groupby(['AÑO', 'MES', 'MES_NUM'], as_index=False)
        .agg({
            'PLANTILLA AUTORIZADA': 'sum',
            'TOTAL ACTIVOS': 'sum'
        })
    )
    monthly['% COBERTURA'] = (
        monthly['TOTAL ACTIVOS'] / monthly['PLANTILLA AUTORIZADA'] * 100
    )
    monthly = monthly.sort_values(['AÑO', 'MES_NUM'])
    monthly['MES'] = pd.Categorical(
        monthly['MES'],
        categories=list(month_map.keys()),
        ordered=True
    )
    monthly['% COBERTURA'] = monthly['% COBERTURA'].round(1)
    coverage_df = monthly.copy()
    coverage_df = coverage_df.sort_values(['AÑO', 'MES_NUM'], ascending=[True, True])
    coverage_df['MES_ANO'] = (
        coverage_df['MES']
        .astype(str)
        .str.title() + ' ' +
        coverage_df['AÑO'].astype(int).astype(str)
    )
    cobertura_labels = coverage_df['MES_ANO'].tolist()
    cobertura_values = coverage_df['% COBERTURA'].tolist()
    cobertura_autorizados = coverage_df['PLANTILLA AUTORIZADA'].tolist()
    cobertura_activos = coverage_df['TOTAL ACTIVOS'].tolist()
    df_detalle = df_cobertura_de_cartera.copy()
    df_detalle['MES_NUM'] = df_detalle['MES'].map(month_map)
    df_detalle = df_detalle.sort_values(
        ['AÑO', 'MES_NUM', 'CAMPAÑA'],
        ascending=[False, False, True]
    )
    cobertura_detalle_rows = df_detalle.to_dict(orient='records')
    # INGRESOS VS BAJAS - SEMANAL
    def _to_datetime(series, dayfirst=True):
        return pd.to_datetime(series, dayfirst=dayfirst, errors="coerce")
    def _weekly_counts_from_dates(dates, year):
        dates = pd.to_datetime(dates, errors="coerce")
        dates = dates[dates.notna()]
        mask_year = dates.dt.year == year
        weeks = dates[mask_year].dt.isocalendar().week
        counts = weeks.value_counts().sort_index()
        full_index = pd.Index(range(1, 54), name="Semana")
        return counts.reindex(full_index, fill_value=0)
    ingresos_activos = _to_datetime(df_plantilla_activa["FECHA DE INGRESO"], dayfirst=True)
    bajas_baja_dates = _to_datetime(df_plantilla_bajas["BAJA"], dayfirst=True)
    bajas_ingreso_dates = _to_datetime(df_plantilla_bajas["FECHA DE INGRESO"], dayfirst=True)
    mask_bajas_validas = bajas_ingreso_dates.notna()
    bajas_baja_dates_validas = bajas_baja_dates[mask_bajas_validas]
    ingresos_todos = pd.concat([ingresos_activos, bajas_ingreso_dates], ignore_index=True)
    YEAR = pd.Timestamp.today().year 
    def _years_in_series(d):
        d = pd.to_datetime(d, errors="coerce")
        return d.dt.year.dropna().astype(int).unique()
    years_candidates = sorted(set(_years_in_series(ingresos_todos)).union(set(_years_in_series(bajas_baja_dates_validas))), reverse=True)
    ingresos_year_counts = _weekly_counts_from_dates(ingresos_todos, YEAR)
    bajas_year_counts = _weekly_counts_from_dates(bajas_baja_dates_validas, YEAR)
    fallback_used = False
    if (ingresos_year_counts.sum() + bajas_year_counts.sum()) == 0 and len(years_candidates) > 0 and YEAR not in years_candidates:
        YEAR = years_candidates[0]
        ingresos_year_counts = _weekly_counts_from_dates(ingresos_todos, YEAR)
        bajas_year_counts = _weekly_counts_from_dates(bajas_baja_dates_validas, YEAR)
        fallback_used = True
    weeks = pd.Index(range(1, 54), name="Semana")
    df_semana = pd.DataFrame({
        "Semana": weeks,
        "Ingresos": ingresos_year_counts.reindex(weeks, fill_value=0).values,
        "Bajas": bajas_year_counts.reindex(weeks, fill_value=0).values
    })
    week_month_nums = []
    for w in df_semana["Semana"]:
        try:
            d = dt.datetime.fromisocalendar(int(YEAR), int(w), 1)
            week_month_nums.append(int(d.month))
        except Exception:
            week_month_nums.append(None)
    # --- INGRESOS VS BAJAS - MENSUAL (para el filtro Semanal/Mensual) ---
    def _monthly_counts_from_dates(dates, year):
        dates = pd.to_datetime(dates, errors="coerce")
        dates = dates[dates.notna()]
        dates = dates[dates.dt.year == year]
        counts = dates.dt.month.value_counts().sort_index()
        full_index = pd.Index(range(1, 13), name="Mes")
        return counts.reindex(full_index, fill_value=0)
    ingresos_month_counts = _monthly_counts_from_dates(ingresos_todos, YEAR)
    bajas_month_counts = _monthly_counts_from_dates(bajas_baja_dates_validas, YEAR)
    meses_numeros = list(range(1, 13))
    meses_labels_corto = MESES_ES_CORTO[:12]
    ingresos_mes = [int(ingresos_month_counts.get(m, 0)) for m in meses_numeros]
    bajas_mes = [int(bajas_month_counts.get(m, 0)) for m in meses_numeros]
    # INGRESOS VS BAJAS - ACUMULADO
    df_semana_cum = df_semana.copy()
    df_semana_cum["Ingresos acumulados"] = df_semana_cum["Ingresos"].cumsum()
    df_semana_cum["Bajas acumuladas"] = df_semana_cum["Bajas"].cumsum()
    # LEADs
    data_reclutamiento["fecha_publicacion"].replace("", pd.NA, inplace=True)
    data_reclutamiento["fecha_publicacion"] = pd.to_datetime(
        data_reclutamiento["fecha_publicacion"], errors="coerce"
    )
    data_reclutamiento["entrevista"] = pd.to_datetime(
        data_reclutamiento["entrevista"], errors="coerce"
    )
    mask = data_reclutamiento["fecha_publicacion"].isna()
    data_reclutamiento.loc[mask, "fecha_publicacion"] = data_reclutamiento.loc[mask, "entrevista"]
    data_reclutamiento["fecha_publicacion"] = pd.to_datetime(
        data_reclutamiento["fecha_publicacion"], errors="coerce"
    )
    data_reclutamiento["mes"] = data_reclutamiento["fecha_publicacion"].dt.to_period("M").dt.to_timestamp()
    df_mes = (
        data_reclutamiento
        .groupby(["reclutador", "mes"])
        .size()
        .reset_index(name="n_leads")
    )
    df_total = (
        df_mes
        .groupby("mes", as_index=False)["n_leads"]
        .sum()
    )
    df_total["reclutador"] = "TOTAL"
    df_leads_all = pd.concat([df_mes, df_total], ignore_index=True)
    df_leads_all = df_leads_all.dropna(subset=["mes"])
    df_leads_all["mes"] = pd.to_datetime(df_leads_all["mes"], errors="coerce")
    meses_leads = sorted(df_leads_all["mes"].dropna().unique())
    etiquetas_leads = [pd.to_datetime(m).strftime("%Y-%m-%d") for m in meses_leads]
    reclutadores = sorted(df_leads_all["reclutador"].dropna().unique())
    leads_por_reclutador = {}
    for rec in reclutadores:
        if rec == "TOTAL":
            continue
        serie = (
            df_leads_all[df_leads_all["reclutador"] == rec]
            .set_index("mes")["n_leads"]
        )
        leads_por_reclutador[rec] = [int(serie.get(m, 0)) for m in meses_leads]
    serie_total = (
        df_leads_all[df_leads_all["reclutador"] == "TOTAL"]
        .set_index("mes")["n_leads"]
    )
    leads_total = [int(serie_total.get(m, 0)) for m in meses_leads]
    # CANALES DE INGRESO - PLANTILLA Y PAUTA
    # PLANTILLA
    df = df_plantilla_activa.copy()
    df['FECHA DE INGRESO'] = pd.to_datetime(
        df['FECHA DE INGRESO'],
        dayfirst=True,
        errors='coerce'
    )
    df = df.dropna(subset=['FECHA DE INGRESO'])
    anio_actual = pd.Timestamp.today().year
    df = df[df['FECHA DE INGRESO'].dt.year == anio_actual]
    df['anio_mes'] = df['FECHA DE INGRESO'].dt.strftime('%Y-%m')
    df_grouped = (
        df
        .groupby(['anio_mes', 'FUENTE'])
        .size()
        .reset_index(name='CANTIDAD')
    )
    canales_plantilla_por_mes = {}
    canales_plantilla_meses_labels = []
    if not df_grouped.empty:
        df_grouped = df_grouped.dropna(subset=['anio_mes', 'FUENTE'])
        df_grouped['anio_mes'] = df_grouped['anio_mes'].astype(str)
        meses_unicos_plantilla = sorted(df_grouped['anio_mes'].unique())
        canales_plantilla_meses_labels = list(meses_unicos_plantilla)
        for m in meses_unicos_plantilla:
            df_m = df_grouped[df_grouped['anio_mes'] == m]
            canales_plantilla_por_mes[m] = dict(
                zip(df_m['FUENTE'], df_m['CANTIDAD'])
            )
    # PAUTA
    data_reclutamiento["fecha_publicacion"] = pd.to_datetime(
        data_reclutamiento["fecha_publicacion"],
        errors="coerce"
    )
    data_reclutamiento["anio_mes"] = data_reclutamiento["fecha_publicacion"].dt.strftime("%Y-%m")
    mask_referido = data_reclutamiento["folio"] == "Referido"
    df_ref = data_reclutamiento[mask_referido].copy()
    agg_ref = (
        df_ref
        .groupby(["canal", "anio_mes"])
        .size()
        .reset_index(name="folios_unicos")
    )
    df_no_ref = data_reclutamiento[~mask_referido].copy()
    df_no_ref = df_no_ref.drop_duplicates(subset=["canal", "folio", "anio_mes"])
    agg_no_ref = (
        df_no_ref
        .groupby(["canal", "anio_mes"])["folio"]
        .nunique()
        .reset_index(name="folios_unicos")
    )
    resumen = pd.concat([agg_ref, agg_no_ref], ignore_index=True)
    resumen = (
        resumen
        .groupby(["canal", "anio_mes"])["folios_unicos"]
        .sum()
        .reset_index()
    )
    canales_pauta_por_mes = {}
    canales_pauta_meses_labels = []
    if not resumen.empty:
        resumen = resumen.dropna(subset=["anio_mes", "canal"])
        resumen["anio_mes"] = resumen["anio_mes"].astype(str)
        meses_unicos_pauta = sorted(resumen["anio_mes"].unique())
        canales_pauta_meses_labels = list(meses_unicos_pauta)
        for m in meses_unicos_pauta:
            df_m = resumen[resumen["anio_mes"] == m]
            canales_pauta_por_mes[m] = dict(
                zip(df_m["canal"], df_m["folios_unicos"])
            )
    # INVERSION EN PAUTA - SEMANAL Y MENSUAL
    data_reclutamiento["monto_pagado"] = pd.to_numeric(
        data_reclutamiento["monto_pagado"],
        errors="coerce"
    )
    df_pagos = data_reclutamiento.dropna(subset=["monto_pagado", "fecha_publicacion"]).copy()
    df_pagos["fecha_publicacion"] = pd.to_datetime(df_pagos["fecha_publicacion"], errors="coerce")
    df_pagos = df_pagos.sort_values("fecha_publicacion")
    df_pagos_unicos = df_pagos.drop_duplicates(subset=["folio"], keep="first")
    df_pagos_unicos["fecha_semana"] = (
        df_pagos_unicos["fecha_publicacion"]
        - pd.to_timedelta(df_pagos_unicos["fecha_publicacion"].dt.weekday, unit="D")
    )
    pagos_semanales = (
        df_pagos_unicos
        .groupby("fecha_semana")["monto_pagado"]
        .sum()
        .reset_index(name="monto_total_pagado")
    )
    df_pagos_unicos["anio_mes_pago"] = df_pagos_unicos["fecha_publicacion"].dt.to_period("M").astype(str)
    pagos_mensuales = (
        df_pagos_unicos
        .groupby("anio_mes_pago")["monto_pagado"]
        .sum()
        .reset_index(name="monto_total_pagado")
    )
    pagos_semanales_labels = pagos_semanales["fecha_semana"].dt.strftime("%Y-%m-%d").tolist()
    pagos_semanales_values = pagos_semanales["monto_total_pagado"].round(2).tolist()
    pagos_mensuales_labels = pagos_mensuales["anio_mes_pago"].astype(str).tolist()
    pagos_mensuales_values = pagos_mensuales["monto_total_pagado"].round(2).tolist()
    # MOTIVOS DE BAJAS
    df = df_plantilla_bajas.copy()
    df["BAJA"] = pd.to_datetime(df["BAJA"], dayfirst=True, errors="coerce")
    mask_year = df["BAJA"].dt.year == YEAR
    df_year = df.loc[mask_year]
    serie = df_year["MOTIVO"].fillna("SIN MOTIVO")
    resumen_pct = (
        serie.value_counts(dropna=False, normalize=True)
        .rename_axis("MOTIVO")
        .reset_index(name="porcentaje")
    )
    resumen_cnt = (
        serie.value_counts(dropna=False)
        .rename_axis("MOTIVO")
        .reset_index(name="conteo")
    )
    resumen = (
        resumen_pct.merge(resumen_cnt, on="MOTIVO")
        .sort_values("porcentaje", ascending=False)
    )
    umbral = 0.028  
    resumen["porcentaje"] = resumen["porcentaje"].astype(float)
    principales = resumen[resumen["porcentaje"] >= umbral].copy()
    otros_conteo = resumen.loc[resumen["porcentaje"] < umbral, "conteo"].sum()
    motivos_baja_labels = principales["MOTIVO"].astype(str).tolist()
    motivos_baja_values = principales["conteo"].astype(int).tolist()
    if otros_conteo > 0:
        motivos_baja_labels.append("Otros")
        motivos_baja_values.append(int(otros_conteo))
    # TABLA DE MOTIVOS DE BAJA POR MES
    df = df_plantilla_bajas.copy()
    df["BAJA"] = pd.to_datetime(df["BAJA"], dayfirst=True, errors="coerce")
    df_year = df.loc[df["BAJA"].dt.year == YEAR].copy()

    meses_es = ["Enero","Febrero","Marzo","Abril","Mayo","Junio",
                "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"]
    mes_map = {i+1: mes for i, mes in enumerate(meses_es)}

    df_year["Mes"] = pd.Categorical(
        df_year["BAJA"].dt.month.map(mes_map),
        categories=meses_es,
        ordered=True
    )
    pivot = pd.crosstab(
        df_year["MOTIVO"].fillna("SIN MOTIVO"),
        df_year["Mes"],
        margins=True,
        margins_name="Total"
    )

    meses_es = ["Enero","Febrero","Marzo","Abril","Mayo","Junio",
                "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"]
    pivot = pivot.reindex(columns=meses_es + ["Total"]).fillna(0).astype(int)

    if "Total" in pivot.index:
        total_row = pivot.loc[["Total"]]
        cuerpo = pivot.drop(index="Total")
        cuerpo = cuerpo.sort_values(by="Total", ascending=False)
        pivot_sorted = pd.concat([cuerpo, total_row])
    else:
        pivot_sorted = pivot.sort_values(by="Total", ascending=False)

    tabla = pivot_sorted.reset_index().rename(columns={"index": "MOTIVO"})
    tabla_motivos_baja_rows = tabla.to_dict(orient="records")
    tabla_motivos_baja_columns = [col for col in tabla.columns if col != "MOTIVO"]
    # ANTIGUEDAD PROMEDIO DE BAJAS
    df = df_plantilla_bajas.copy()
    for col in ["FECHA DE INGRESO", "BAJA"]:
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
    mask_valid = (
        df["BAJA"].notna() &
        df["FECHA DE INGRESO"].notna() &
        (df["BAJA"].dt.year == YEAR)
    )
    df_calc = (
        df.loc[mask_valid]
        .assign(
            antiguedad_meses=lambda d: (d["BAJA"] - d["FECHA DE INGRESO"]).dt.days / 30.4375,
            MesNum=lambda d: d["BAJA"].dt.month
        )
    )
    idx = pd.Index(range(1, 13), name="MesNum")
    prom = (
        df_calc.groupby("MesNum", as_index=True)["antiguedad_meses"]
            .mean()
            .reindex(idx)
            .reset_index()
    )
    meses_esp = {
        1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr",
        5: "May", 6: "Jun", 7: "Jul", 8: "Ago",
        9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"
    }
    prom["Mes"] = prom["MesNum"].map(meses_esp)
    orden_meses = list(meses_esp.values())
    prom["Mes"] = pd.Categorical(prom["Mes"], categories=orden_meses, ordered=True)
    prom = prom.sort_values("MesNum")
    antiguedad_bajas_labels = prom["Mes"].tolist()
    antiguedad_bajas_values = (
        prom["antiguedad_meses"]
        .round(1)
        .fillna(0)
        .tolist()
    )

    ############################################# VARIABLES A ENVIAR ##############################################

    return render_template(
        "dashboard-de-capital-humano.html",
        # --- KPIS ---
        # Plantilla autorizada
        plantilla_autorizada_total=sum_plantilla_general_autorizada,
        plantilla_autorizada_operativa=sum_plantilla_operativa_autorizada,
        plantilla_autorizada_administrativa=sum_plantilla_administrativa_autorizada,
        # Género
        genero_cantidad=genero_cantidad_dict,
        genero_porcentaje=genero_porcentaje_dict,
        # Nómina
        nomina_total_general=total_general_nomina,
        nomina_total_general_fmt=total_general_nomina_fmt,
        nomina_estructura=nomina_estructura,
        nomina_operacion=nomina_operacion,
        nomina_estructura_fmt=nomina_estructura_fmt,
        nomina_operacion_fmt=nomina_operacion_fmt,
        nomina_por_area=nomina_por_area_dict,
        nomina_por_area_fmt=nomina_por_area_fmt_dict,
        # Contratos
        contratos_total=contratos_total,
        contratos_estructura=contratos_estructura,
        contratos_operacion=contratos_operacion,
        contratos_por_area=contratos_por_area_dict,
        # FUNELES ÚLTIMOS 3 MESES
        funnel_labels=funnel_labels,
        funnel_datasets=funnel_datasets,
        # COBERTURA DE CARTERA
        cobertura_labels=cobertura_labels,
        cobertura_values=cobertura_values,
        cobertura_autorizados=cobertura_autorizados,
        cobertura_activos=cobertura_activos,
        cobertura_detalle_rows=cobertura_detalle_rows,
        # INGRESOS VS BAJAS - SEMANAL / MENSUAL
        ingresos_bajas_year=int(YEAR),
        ingresos_bajas_week_labels=df_semana["Semana"].astype(int).tolist(),
        ingresos_bajas_week_ingresos=df_semana["Ingresos"].astype(int).tolist(),
        ingresos_bajas_week_bajas=df_semana["Bajas"].astype(int).tolist(),
        ingresos_bajas_week_month_numbers=week_month_nums,
        ingresos_bajas_month_labels=meses_labels_corto,
        ingresos_bajas_month_ingresos=ingresos_mes,
        ingresos_bajas_month_bajas=bajas_mes,
        # INGRESOS VS BAJAS - ACUMULADO
        ingresos_bajas_cum_week_labels=df_semana_cum["Semana"].astype(int).tolist(),
        ingresos_bajas_cum_ingresos=df_semana_cum["Ingresos acumulados"].astype(int).tolist(),
        ingresos_bajas_cum_bajas=df_semana_cum["Bajas acumuladas"].astype(int).tolist(),
        # LEADs POR RECLUTADOR
        leads_labels=etiquetas_leads,
        leads_por_reclutador=leads_por_reclutador,
        leads_total=leads_total,
        # CANALES DE INGRESO
        canales_plantilla_por_mes=canales_plantilla_por_mes,
        canales_plantilla_meses_labels=canales_plantilla_meses_labels,
        canales_pauta_por_mes=canales_pauta_por_mes,
        canales_pauta_meses_labels=canales_pauta_meses_labels,
        # LÍNEA DEL TIEMPO DE PAGOS
        pagos_semanales_labels=pagos_semanales_labels,
        pagos_semanales_values=pagos_semanales_values,
        pagos_mensuales_labels=pagos_mensuales_labels,
        pagos_mensuales_values=pagos_mensuales_values,
        # MOTIVOS DE BAJAS - PIE CHART
        motivos_baja_labels=motivos_baja_labels,
        motivos_baja_values=motivos_baja_values,
        # DETALLE MOTIVOS DE BAJA POR MES
        tabla_motivos_baja_rows=tabla_motivos_baja_rows,
        tabla_motivos_baja_columns=tabla_motivos_baja_columns,
        # ANTIGÜEDAD PROMEDIO DE BAJAS
        antiguedad_bajas_year=int(YEAR),
        antiguedad_bajas_labels=antiguedad_bajas_labels,
        antiguedad_bajas_values=antiguedad_bajas_values,
        # >>> NUEVO: filtro de campaña
        campanias=campanias,
        campania_seleccionada=campania_seleccionada,
    )


######################################### EJECUTADOR #############################################

if __name__ == "__main__":
    app.run(debug = True, host = "0.0.0.0", port = 5000)