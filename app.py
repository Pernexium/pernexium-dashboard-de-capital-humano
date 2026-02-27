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
import calendar
from io import BytesIO
from markupsafe import Markup 
from dotenv import load_dotenv
from datetime import datetime, timedelta
from google.oauth2 import service_account
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from werkzeug.exceptions import HTTPException
from flask import Flask, render_template, request 

######################################## CARGA VARIABLES #######################################

load_dotenv()

S3_SA_KEY      = os.getenv("S3_SA_KEY")
SHEET_ID       = os.getenv("SHEET_ID")     
SHEET_ID_2     = os.getenv("SHEET_ID_2")   
INGRESOS_OPERACION_SHEET_ID = os.getenv("INGRESOS_OPERACION_SHEET_ID")
SHEET_ID_LEADS = INGRESOS_OPERACION_SHEET_ID 


SCOPES         = json.loads(os.getenv("SCOPES")) if os.getenv("SCOPES") else []
S3_BUCKET      = os.getenv("S3_BUCKET")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

################################# ACCESO A GOOGLE SHEETS (SERVICE ACCOUNT) #####################

def get_service_account_credentials():
    obj       = s3.get_object(Bucket=S3_BUCKET, Key=S3_SA_KEY)
    sa_info   = json.loads(obj["Body"].read().decode())
    creds_sa  = service_account.Credentials.from_service_account_info(sa_info, scopes=SCOPES)
    return creds_sa

def build_sheets_service():
    creds = get_service_account_credentials()
    return build("sheets", "v4", credentials=creds, cache_discovery=False)

def fetch_sheet_data(service, sheet_name):
    """Lee un rango amplio de una pestaña y detecta automáticamente la fila de encabezados.

    Algunas pestañas tienen encabezados reales después de varias filas (p.ej. filas 12-13).
    Esta función escanea las primeras ~30 filas buscando una fila que contenga
    varios encabezados conocidos y usa esa como header.
    """
    safe_sheet = sheet_name.replace("'", "''")
    rng = f"'{safe_sheet}'!A1:ZZ"

    vals = (
        service.spreadsheets()
        .values()
        .get(spreadsheetId=SHEET_ID, range=rng)
        .execute()
        .get("values", [])
    )

    if not vals:
        return pd.DataFrame()

    def _norm(s: str) -> str:
        s = "" if s is None else str(s)
        s = s.strip().lower()
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
        s = re.sub(r"\s+", " ", s)
        return s

    # Encabezados esperados para detectar la fila correcta
    expected = {
        _norm("NOMBRE COMPLETO"),
        _norm("RECLUTADOR"),
        _norm("FECHA DE INGRESO A CAPA"),
        _norm("DIA 1"),
        _norm("DÍA 1"),
        _norm("ESTATUS"),
    }

    # Buscar mejor fila de encabezados dentro de las primeras 30 filas
    scan_rows = vals[:30]
    best_idx, best_score = 0, -1
    for i, row in enumerate(scan_rows):
        row_norm = {_norm(c) for c in row if c is not None and str(c).strip() != ""}
        score = len(row_norm & expected)
        # preferimos mayor score, y en empate, mayor cantidad de celdas no vacías
        if score > best_score or (score == best_score and len(row_norm) > len({_norm(c) for c in scan_rows[best_idx] if c is not None and str(c).strip() != ""})):
            best_idx, best_score = i, score

    # Si encontramos una fila posterior con señales claras, úsala como header
    header_row_idx = best_idx if best_score >= 2 else 0

    headers = vals[header_row_idx]

    # ── Soporte para encabezados en 2 filas (p.ej. fila 12 con secciones y fila 13 con subheaders DIA 1..4) ──
    # Si la fila siguiente contiene sub-encabezados relevantes (DIA 1, DIA 2, etc.) y en la fila actual
    # hay celdas vacías / secciones (ASISTENCIA), combinamos ambos y saltamos 2 filas.
    data_start_idx = header_row_idx + 1
    if header_row_idx + 1 < len(vals):
        next_row = vals[header_row_idx + 1]

        next_norm = {_norm(c) for c in next_row if c is not None and str(c).strip() != ""}
        sub_expected = {
            _norm("DIA 1"), _norm("DÍA 1"), _norm("DIA 2"), _norm("DIA 3"), _norm("DIA 4"),
        }
        cur_norm = {_norm(c) for c in headers if c is not None and str(c).strip() != ""}

        # Heurística: si la fila siguiente trae 'DIA 1' (u otros) y la actual trae 'ASISTENCIA' u otros headers,
        # consideramos que es un header de 2 filas.
        if (len(next_norm & sub_expected) >= 1) and (len(cur_norm & expected) >= 1):
            max_cols_tmp = max(len(headers), len(next_row))
            merged_headers = []
            for j in range(max_cols_tmp):
                h = headers[j] if j < len(headers) else ""
                n = next_row[j] if j < len(next_row) else ""
                h_str = ("" if h is None else str(h)).strip()
                n_str = ("" if n is None else str(n)).strip()

                # Si el header superior está vacío o es una sección tipo 'ASISTENCIA', usar el subheader si existe
                if (h_str == "" or _norm(h_str) in {_norm("ASISTENCIA")} or h_str.lower().startswith("col")) and n_str != "":
                    merged_headers.append(n_str)
                else:
                    merged_headers.append(h_str)
            headers = merged_headers
            data_start_idx = header_row_idx + 2

    data = vals[data_start_idx:]

    # Completar columnas faltantes y padear filas
    max_cols = max(len(headers), *(len(r) for r in data)) if data else len(headers)
    headers = list(headers) + [f"Col_{i}" for i in range(len(headers), max_cols)]
    padded = [row + [None] * (max_cols - len(row)) for row in data]

    # Limpiar headers, evitar vacíos y duplicados simples
    clean_headers = []
    seen = {}
    for h in headers:
        h0 = ("" if h is None else str(h)).strip()
        if h0 == "":
            h0 = "col"
        key = h0.lower()
        if key in seen:
            seen[key] += 1
            h0 = f"{h0}_{seen[key]}"
        else:
            seen[key] = 0
        clean_headers.append(h0)

    return pd.DataFrame(padded, columns=clean_headers)

######################################### FUNCIONES AUXILIARES #########################################

def moneda_es(x):
    try:
        return '$' + format(x, ',.2f').replace(',', 'X').replace('.', ',').replace('X', '.')
    except Exception:
        return '$0,00'

MESES_ES = ["ene", "feb", "mar", "abr", "may", "jun", 
            "jul", "ago", "sep", "oct", "nov", "dic"]

MESES_ES_CORTO = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                  "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

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

def parse_fecha_mixta(s: pd.Series) -> pd.Series:
    s = (
        s.astype(str)
         .str.strip()
         .replace({"-": None, "": None, "nan": None, "NaN": None, "None": None})
    )

    d1 = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")

    mask = d1.isna()
    if mask.any():
        d2 = pd.to_datetime(s[mask], format="%d/%m/%Y %H:%M:%S", errors="coerce")
        d1.loc[mask] = d2

    mask = d1.isna()
    if mask.any():
        d3 = pd.to_datetime(s[mask], dayfirst=True, errors="coerce")
        d1.loc[mask] = d3

    return d1

def get_data_sheets():
    if not SHEET_ID_2:
        raise RuntimeError("SHEET_ID_2 no está definido en las variables de entorno.")

    SHEET_ID_LEADS   = SHEET_ID_2
    SHEET_NAME_LEADS = "Hoja 1"
    S3_BUCKET_TOKENS = "gcp-tokens"

    S3_CLIENT_SECRET = "credentials_hibran_sheets.json"
    S3_TOKEN_READ    = "token_hibran_sheets.json"
    S3_TOKEN_WRITE   = "token_hibran_sheets_escritura.json"

    SCOPES_READ  = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    SCOPES_WRITE = ["https://www.googleapis.com/auth/spreadsheets"]

    s3_tokens = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
    )

    def load_json_from_s3(key):
        obj = s3_tokens.get_object(Bucket=S3_BUCKET_TOKENS, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))

    def save_to_s3(key, content):
        s3_tokens.put_object(Bucket=S3_BUCKET_TOKENS, Key=key, Body=content.encode("utf-8"))

    def get_creds(secret_file, token_file, scopes):
        client_config = load_json_from_s3(secret_file)
        try:
            token_json = load_json_from_s3(token_file)
            creds = Credentials.from_authorized_user_info(token_json, scopes)
        except Exception:
            creds = None

        if creds and creds.valid:
            return creds

        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                save_to_s3(token_file, creds.to_json())
                return creds
            except Exception:
                pass

        flow = InstalledAppFlow.from_client_config(client_config, scopes)
        creds = flow.run_local_server(port=8080, prompt="consent", access_type="offline")
        save_to_s3(token_file, creds.to_json())
        return creds

    def build_read_service():
        return build(
            "sheets",
            "v4",
            credentials=get_creds(S3_CLIENT_SECRET, S3_TOKEN_READ, SCOPES_READ)
        )

    service_r = build_read_service()



def fetch_sheet_df(service):
    
    safe_sheet = SHEET_NAME_LEADS.replace("'", "''")

    result = service.spreadsheets().values().get(
        spreadsheetId=SHEET_ID_LEADS,
        range=f"'{safe_sheet}'!A1:ZZ5000"
    ).execute()

    values = result.get("values", [])
    if not values:
        return pd.DataFrame()

    # --- helpers ---
    def _norm(s):
        return str(s or "").strip()

    def _is_empty_row(r):
        return all(_norm(x) == "" for x in r)

    def _score_header_row(row):
        # Heurística: mientras más texto y menos celdas vacías, más probable que sea header
        texts = [_norm(x) for x in row]
        non_empty = sum(1 for x in texts if x)
        alpha = sum(1 for x in texts if any(ch.isalpha() for ch in x))
        return non_empty + alpha

    # --- detectar fila de header (soporta header en fila 12) ---
    max_scan = min(len(values), 30)  # escanea primeras 30 filas
    header_idx = None
    best_score = -1

    for i in range(max_scan):
        row = values[i]
        if _is_empty_row(row):
            continue
        sc = _score_header_row(row)
        if sc > best_score:
            best_score = sc
            header_idx = i

    if header_idx is None:
        return pd.DataFrame()

    header1 = [_norm(x) for x in values[header_idx]]
    header2 = []
    # si la siguiente fila parece sub-header, úsala (header en 2 niveles)
    if header_idx + 1 < len(values) and not _is_empty_row(values[header_idx + 1]):
        # condición simple: que tenga varios textos y no sea puramente data numérica
        next_row = values[header_idx + 1]
        next_texts = [_norm(x) for x in next_row]
        if sum(1 for x in next_texts if x) >= max(2, len(header1) // 4):
            header2 = next_texts

    # construir columnas
    cols = []
    for j in range(max(len(header1), len(header2))):
        h1 = header1[j] if j < len(header1) else ""
        h2 = header2[j] if j < len(header2) else ""
        name = " ".join([x for x in [h1, h2] if x]).strip()
        cols.append(name if name else f"COL_{j}")

    # data empieza después del header (1 o 2 filas)
    data_start = header_idx + (2 if header2 else 1)
    rows = values[data_start:]

    # pad rows para que tengan el mismo largo que cols
    fixed_rows = []
    for r in rows:
        r = list(r)
        if len(r) < len(cols):
            r += [""] * (len(cols) - len(r))
        fixed_rows.append(r[:len(cols)])

    df = pd.DataFrame(fixed_rows, columns=cols)

    # limpiar nombres duplicados
    seen = {}
    new_cols = []
    for c in df.columns:
        base = c.strip() if c.strip() else "COL"
        if base not in seen:
            seen[base] = 0
            new_cols.append(base)
        else:
            seen[base] += 1
            new_cols.append(f"{base}_{seen[base]}")
    df.columns = new_cols

    return df

############################################# DASHBOARD ##############################################

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    
    ################################# BASES DE DATOS ##################################
    campania_param = request.args.get("campania", default="", type=str).strip()
    campania_seleccionada = campania_param or None

    # Fecha/hora actual en México (reutilizada en todo el dashboard)
    mx_now = pd.Timestamp.now(tz=pytz.timezone("America/Mexico_City")).tz_localize(None)

    # Filtro para el gráfico histórico (acumulado anual): por defecto el año actual
    _mx_today = mx_now.date()
    _current_year = int(_mx_today.year)

    _historico_year_raw = request.args.get(
        "historico_year",
        default=str(_current_year),
        type=str
    ).strip()

    try:
        historico_year_selected = int(_historico_year_raw) if _historico_year_raw else _current_year
    except Exception:
        historico_year_selected = _current_year

    if historico_year_selected > _current_year:
        historico_year_selected = _current_year



    service = build_sheets_service()
    df_reclutamiento_full = get_data_sheets()

    df_plantilla_activa_full     = fetch_sheet_data(service, "PLANTILLA AJUSTE")
    df_plantilla_autorizada_full = fetch_sheet_data(service, "PLANTILLA AUTORIZADA")
    df_plantilla_bajas_full      = fetch_sheet_data(service, "PLANTILLA BAJAS")

    campanias_reclut = set(
        df_reclutamiento_full.get("Campaña", pd.Series(dtype=str))
        .dropna()
        .astype(str)
        .str.strip()
    ) if "Campaña" in df_reclutamiento_full.columns else set()

    campanias_sheets = set(
        df_plantilla_activa_full.get("CAMPAÑA", pd.Series(dtype=str))
        .dropna()
        .astype(str)
        .str.strip()
    )
    campanias = sorted(campanias_reclut.union(campanias_sheets))

    df_reclutamiento        = df_reclutamiento_full.copy()
    df_plantilla_activa     = df_plantilla_activa_full.copy()
    df_plantilla_autorizada = df_plantilla_autorizada_full.copy()
    df_plantilla_bajas      = df_plantilla_bajas_full.copy()

    if campania_seleccionada:
        if "Campaña" in df_reclutamiento.columns:
            df_reclutamiento = df_reclutamiento[
                df_reclutamiento["Campaña"]
                    .astype(str)
                    .str.strip()
                    == campania_seleccionada
            ]

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
    mes_actual_num    = dt.datetime.today().month
    mes_actual_nombre = mapa_meses[mes_actual_num]

    # Plantilla autorizada
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

    # Género
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
    genero_cantidad_dict   = resumen_GENERO['Cantidad'].to_dict()
    genero_porcentaje_dict = resumen_GENERO['Porcentaje (%)'].to_dict()

    # Nómina
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
    total_general_nomina     = float(df_nomina['SUELDO MENSUAL'].sum())
    total_general_nomina_fmt = moneda_es(total_general_nomina)
    nomina_formateada        = nomina_por_area.copy()
    nomina_formateada['TOTAL NOMINA'] = nomina_formateada['TOTAL NOMINA'].map(moneda_es)
    nomina_por_area_dict     = nomina_por_area.set_index('AREA')['TOTAL NOMINA'].to_dict()
    nomina_por_area_fmt_dict = nomina_formateada.set_index('AREA')['TOTAL NOMINA'].to_dict()
    nomina_estructura        = float(nomina_por_area_dict.get('ESTRUCTURA', 0.0))
    nomina_operacion         = float(nomina_por_area_dict.get('OPERACION', 0.0))
    nomina_estructura_fmt    = nomina_por_area_fmt_dict.get('ESTRUCTURA', moneda_es(0))
    nomina_operacion_fmt     = nomina_por_area_fmt_dict.get('OPERACION', moneda_es(0))

    # Contratos
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
    contratos_total         = int(contratos_por_area['CONTRATOS'].sum())
    contratos_estructura    = int(contratos_por_area_dict.get('ESTRUCTURA', 0))
    contratos_operacion     = int(contratos_por_area_dict.get('OPERACION', 0))

    ############################################# FUNNEL RECLUTAMIENTO #################################

    data_reclutamiento = df_reclutamiento_full.copy()
    data_reclutamiento = data_reclutamiento.dropna(axis=0, how='all')

    if not data_reclutamiento.empty:
        # ── Normalización robusta de encabezados ──
        def _norm_col(col):
            col = "" if col is None else str(col)
            col = col.strip().replace(" ", "_").replace(".", "").replace(":", "").lower()
            col = (col.replace("ó", "o").replace("í", "i").replace("é", "e")
                      .replace("ú", "u").replace("ñ", "n").replace("á", "a"))
            return col

        cols_norm = [_norm_col(c) for c in list(data_reclutamiento.columns)]
        has_expected_headers = (
            ("nombre_completo" in cols_norm)
            or ("reclutador" in cols_norm)
            or ("fecha_de_ingreso_a_capa" in cols_norm)
            or ("dia_1" in cols_norm)
        )

        # Algunas pestañas traen el header real varias filas abajo (legacy).
        # Solo aplicamos el "shift" de encabezados si NO detectamos headers válidos.
        if (not has_expected_headers) and data_reclutamiento.shape[0] > 3:
            header_idx = None
            for i in range(min(6, len(data_reclutamiento))):
                row_vals = [str(x).strip().lower() for x in data_reclutamiento.iloc[i, :].tolist()]
                if any("nombre completo" in v for v in row_vals) or any("reclutador" == v for v in row_vals):
                    header_idx = i
                    break
            if header_idx is None:
                header_idx = 1  # fallback histórico

            data_reclutamiento.columns = data_reclutamiento.iloc[header_idx, :]
            data_reclutamiento = data_reclutamiento[header_idx + 1 :]
            data_reclutamiento = data_reclutamiento.reset_index(drop=True)
        else:
            data_reclutamiento = data_reclutamiento.reset_index(drop=True)

        # Normaliza nombres de columnas a snake_case simple
        data_reclutamiento.columns = [_norm_col(col) for col in data_reclutamiento.columns]
        data_reclutamiento.rename(columns={"¿es_viable?4": "es_viable_tecnica"}, inplace=True)
        data_reclutamiento.rename(columns={"¿es_viable?": "es_viable_psicometrica"}, inplace=True)
        data_reclutamiento.columns = [
            col.replace("?", "").replace("¿", "").replace("4", "").replace("ii_~_iii_", "")
            for col in data_reclutamiento.columns
        ]

        # Nombre
        if "nombre_candidato" in data_reclutamiento.columns:
            data_reclutamiento["nombre_candidato"] = data_reclutamiento["nombre_candidato"].apply(limpiar_nombre)

        # Mes
        if "mes" in data_reclutamiento.columns:
            data_reclutamiento["mes"] = data_reclutamiento["mes"].apply(normalizar_mes)

        # IV psicometrías
        if "iv_realiza_psicometrias" in data_reclutamiento.columns:
            data_reclutamiento["iv_realiza_psicometrias"] = data_reclutamiento["iv_realiza_psicometrias"].apply(
                lambda x: x.strip().lower().capitalize() if isinstance(x, str) else x
            )

        # Campaña lead
        if "campana" in data_reclutamiento.columns:
            data_reclutamiento["campana"] = data_reclutamiento["campana"].apply(
                lambda x: x.strip().lower().capitalize() if isinstance(x, str) else x
            )
            data_reclutamiento["campana"] = data_reclutamiento["campana"].apply(
                lambda x: x.replace("Fmp (posiciones)", "Fmp posiciones").replace("Fmp posiciones", "Fmp (posiciones)")
                if isinstance(x, str) else x
            )

        # Campaña asignada
        if "se_asigna_a_campana" in data_reclutamiento.columns:
            data_reclutamiento["se_asigna_a_campana"] = data_reclutamiento["se_asigna_a_campana"].apply(
                lambda x: x.strip().lower().capitalize() if isinstance(x, str) else x
            )
            data_reclutamiento["se_asigna_a_campana"] = data_reclutamiento["se_asigna_a_campana"].apply(
                lambda x: x.replace("Fmp (posiciones)", "Fmp posiciones").replace("Fmp posiciones", "Fmp (posiciones)")
                if isinstance(x, str) else x
            )

        # Filtro por campaña en el funnel
        if campania_seleccionada and "campana" in data_reclutamiento.columns:
            data_reclutamiento = data_reclutamiento[
                data_reclutamiento["campana"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    == campania_seleccionada.strip().lower()
            ]

        # Fechas
        if "fecha_publicacion" in data_reclutamiento.columns:
            data_reclutamiento["fecha_publicacion"] = pd.to_datetime(
                data_reclutamiento["fecha_publicacion"], errors="coerce"
            )

        # Escolaridad
        if "escolaridad" in data_reclutamiento.columns:
            data_reclutamiento["escolaridad"] = data_reclutamiento["escolaridad"].apply(
                lambda x: x.strip().lower()
                .replace("bachillerato", "preparatoria")
                .replace("preparatoria terminado", "preparatoria terminada")
                .replace("preparatoria trunco", "preparatoria trunca")
                .replace("preparaatoria", "preparatoria")
                .replace("prepa ", "preparatoria ")
                if isinstance(x, str) else x
            )

        # Calificación
        if "calificacion" in data_reclutamiento.columns:
            data_reclutamiento["calificacion"] = data_reclutamiento["calificacion"].apply(
                lambda x: str(x).strip().replace("Baja", "7")
            )

        # Fecha ingreso / entrevista / capacitación
        if "fecha_de_ingreso" in data_reclutamiento.columns:
            data_reclutamiento["fecha_de_ingreso"] = parse_fecha_mixta(
                data_reclutamiento["fecha_de_ingreso"]
            )

        if "fecha_de_entrevista" in data_reclutamiento.columns:
            data_reclutamiento["fecha_de_entrevista"] = data_reclutamiento["fecha_de_entrevista"].apply(
                lambda x: np.nan if x == '-' else x
            )

        if "fecha_de_capacitacion" in data_reclutamiento.columns:
            data_reclutamiento["fecha_de_capacitacion"] = parse_fecha_mixta(
                data_reclutamiento["fecha_de_capacitacion"]
            )


        # === Ingresos en Capacitación (nuevo cálculo) ===
        # Regla: tomar la fecha de la columna "FECHA DE INGRESO A CAPA" (normalizada a snake_case)
        # y contar solo los registros donde la columna "L14" o "DIA 1" sea "SI".
        def _pick_col(_df: pd.DataFrame, _candidates):
            for _c in _candidates:
                if _c in _df.columns:
                    return _c
            return None

        _col_fecha_capa = _pick_col(
            data_reclutamiento,
            [
                "fecha_de_ingreso_a_capa",
                "fecha_ingreso_a_capa",
                "fecha_de_ingreso_a_capacitacion",
                "fecha_ingreso_a_capacitacion",
            ],
        )

        if _col_fecha_capa is None:
            for _c in data_reclutamiento.columns:
                _cs = str(_c)
                if ("fecha" in _cs) and ("ingreso_a_capa" in _cs or "ingreso_a_cap" in _cs or "ingreso_a_capacit" in _cs):
                    _col_fecha_capa = _cs
                    break
        _col_dia1 = _pick_col(data_reclutamiento, ["dia_1", "dia1", "l14"])

        # Fallback: algunas hojas traen DIA 1 duplicado o con sufijo (p.ej. dia_1_1)
        if _col_dia1 is None:
            for _c in data_reclutamiento.columns:
                if re.fullmatch(r"dia_?1(_\d+)?", str(_c)):
                    _col_dia1 = str(_c)
                    break

        if _col_fecha_capa:
            data_reclutamiento["_fecha_ingreso_capa_dt"] = parse_fecha_mixta(
                data_reclutamiento[_col_fecha_capa]
            )
        else:
            data_reclutamiento["_fecha_ingreso_capa_dt"] = pd.NaT

        if _col_dia1:
            _dia1_s = (
                data_reclutamiento[_col_dia1]
                .astype(str)
                .str.strip()
                .str.upper()
            )
        else:
            _dia1_s = pd.Series([], dtype="object")

        data_reclutamiento["_bin_ingreso_capa"] = (
            data_reclutamiento["_fecha_ingreso_capa_dt"].notna()
            & _dia1_s.eq("SI")
        ).astype(int)

        # Variables sintéticas
        if "aceptada" in data_reclutamiento.columns:
            data_reclutamiento['bin_aceptado'] = data_reclutamiento['aceptada'].apply(
                lambda x: 1 if str(x).strip().lower() == 'si' else 0
            )
        else:
            data_reclutamiento['bin_aceptado'] = 0

        if "fecha_de_entrevista" in data_reclutamiento.columns:
            data_reclutamiento['bin_agenda_entrevista'] = data_reclutamiento['fecha_de_entrevista'].isna().apply(
                lambda x: 0 if x else 1
            )
        else:
            data_reclutamiento['bin_agenda_entrevista'] = 0

        if "fecha_de_ingreso" in data_reclutamiento.columns:
            data_reclutamiento['bin_ingreso'] = data_reclutamiento['fecha_de_ingreso'].isna().apply(
                lambda x: 0 if x else 1
            )
        else:
            data_reclutamiento['bin_ingreso'] = 0

        if "motivo" in data_reclutamiento.columns:
            data_reclutamiento['bin_asiste_entrevista'] = data_reclutamiento['motivo'].apply(
                lambda x: 1 if str(x).strip() in
                ['Aceptado', 'Rechazado', 'No aceptado', 'Fallo en Role-Play', 'Rechaza la vacante'] else 0
            )
        else:
            data_reclutamiento['bin_asiste_entrevista'] = 0

        # Monto pagado
        if "monto_pagado" in data_reclutamiento.columns:
            data_reclutamiento["monto_pagado"] = data_reclutamiento["monto_pagado"].apply(
                lambda x: str(x).replace(",", "").replace("$", "") if pd.notna(x) else x
            )
            data_reclutamiento["monto_pagado"] = data_reclutamiento["monto_pagado"].apply(
                lambda x: float(x) if pd.notna(x) and str(x).replace(".", "", 1).isdigit() else 0.0
            )
        else:
            data_reclutamiento["monto_pagado"] = 0.0

        data_reclutamiento['monto_pagado_distribuido'] = 0.0
        if "folio" in data_reclutamiento.columns:
            for folio in data_reclutamiento["folio"].unique():
                if folio is None or pd.isna(folio):
                    continue
                df_tmp = data_reclutamiento.query("folio == @folio")
                if len(df_tmp) == 0:
                    continue
                costo_lead = df_tmp["monto_pagado"].iloc[0] / len(df_tmp)
                data_reclutamiento.loc[data_reclutamiento["folio"] == folio, 'monto_pagado_distribuido'] = costo_lead
        # Ordenar
        if "fecha_publicacion" in data_reclutamiento.columns:
            data_reclutamiento = data_reclutamiento.sort_values(by="fecha_publicacion")

        # Fecha de entrevista (para contar LEADs en Facebook por mes)
        # Se usa la columna "entrevista" si existe; si no, se intenta con "fecha_de_entrevista".
        if "entrevista" in data_reclutamiento.columns:
            data_reclutamiento["entrevista_dt"] = pd.to_datetime(
                data_reclutamiento["entrevista"], errors="coerce"
            )
        elif "fecha_de_entrevista" in data_reclutamiento.columns:
            data_reclutamiento["entrevista_dt"] = parse_fecha_mixta(
                data_reclutamiento["fecha_de_entrevista"]
            )
        else:
            data_reclutamiento["entrevista_dt"] = pd.NaT

        # Fecha de ingreso (para funnel semanal)
        if "fecha_de_ingreso" in data_reclutamiento.columns:
            data_reclutamiento["ingreso_dt"] = parse_fecha_mixta(
                data_reclutamiento["fecha_de_ingreso"]
            )
        else:
            data_reclutamiento["ingreso_dt"] = pd.NaT

        # Funnel mensual (base por fecha_publicacion para "Asistieron a Entrevista")
        if "fecha_publicacion" in data_reclutamiento.columns:
            df_funnel_mes = (
                data_reclutamiento
                .dropna(subset=["fecha_publicacion"])
                .groupby(pd.Grouper(key="fecha_publicacion", freq="MS"))
                .agg({
                    "fecha_publicacion": "count",
                    "bin_asiste_entrevista": "sum",
                    "bin_ingreso": "sum",
                    "monto_pagado_distribuido": "sum",
                })
                .rename(columns={"fecha_publicacion": "leads_generados"})
                .reset_index()
                .sort_values("fecha_publicacion")
            )
        else:
            df_funnel_mes = pd.DataFrame()
    else:
        data_reclutamiento = pd.DataFrame()
        df_funnel_mes      = pd.DataFrame()

    # --- Funnel: últimos 3 meses SIEMPRE (mes-2, mes-1, mes actual) ---
    current_month_start = pd.Timestamp(int(mx_now.year), int(mx_now.month), 1)
    months_for_funnel = [
        current_month_start - pd.DateOffset(months=2),
        current_month_start - pd.DateOffset(months=1),
        current_month_start,
    ]

    df_funnel_idx = (
        df_funnel_mes.set_index("fecha_publicacion")
        if not df_funnel_mes.empty and "fecha_publicacion" in df_funnel_mes.columns
        else pd.DataFrame()
    )

    entrevista_month = (
        data_reclutamiento["entrevista_dt"].dt.to_period("M").dt.to_timestamp()
        if (not data_reclutamiento.empty and "entrevista_dt" in data_reclutamiento.columns)
        else pd.Series([], dtype="datetime64[ns]")
    )

    funnel_labels = [
        "Leads en Facebook",
        "Asistieron a Entrevista",
        "Ingresos en Capacitación",   
        "Ingresos a Operación",    
    ]



    def _get_ingresos_operacion_mes(_service, _dt_mes):
        try:
            sheet_name = f"{mapa_meses[int(_dt_mes.month)]} {str(int(_dt_mes.year))[-2:]}"
            safe_sheet = sheet_name.replace("'", "''")
            rng = f"'{safe_sheet}'!C6"
            resp = (_service.spreadsheets()
                            .values()
                            .get(spreadsheetId=INGRESOS_OPERACION_SHEET_ID, range=rng)
                            .execute())
            vals = resp.get("values", [])
            if not vals or not vals[0]:
                return 0
            raw = str(vals[0][0]).strip()
            raw = raw.replace(",", "")
            raw = re.sub(r"[^\d\.\-]", "", raw)
            if not raw or raw == "-":
                return 0
            return int(float(raw))
        except Exception:
            return 0



    def _get_ingresos_operacion_week_counts_current_month(_service, _start, _end):
        """
        Lee la hoja del mes actual (mismo nombre que usa el dashboard: 'MES YY') del
        spreadsheet INGRESOS_OPERACION_SHEET_ID y devuelve conteos por semana (lunes)
        para registros donde:
          - STATUS (columna P, desde P12 hacia abajo) == 'INGRESO'
          - FECHA DE INGRESO/SALIDA (columna Q, desde Q12 hacia abajo) define la semana
        """
        if not INGRESOS_OPERACION_SHEET_ID:
            return pd.Series(dtype=int)

        try:
            sheet_name = f"{mapa_meses[int(mx_now.month)]} {str(int(mx_now.year))[-2:]}"
            safe_sheet = sheet_name.replace("'", "''")
            # P = STATUS, Q = FECHA DE INGRESO/SALIDA
            rng = f"'{safe_sheet}'!P12:Q"

            resp = (_service.spreadsheets()
                            .values()
                            .get(spreadsheetId=INGRESOS_OPERACION_SHEET_ID, range=rng)
                            .execute())
            rows = resp.get("values", []) or []
            if not rows:
                return pd.Series(dtype=int)

            status_vals = []
            fecha_vals = []
            for r in rows:
                status_vals.append(r[0] if len(r) > 0 else None)
                fecha_vals.append(r[1] if len(r) > 1 else None)

            status_s = (
                pd.Series(status_vals, dtype="object")
                  .astype(str)
                  .str.strip()
                  .str.upper()
            )
            fecha_dt = parse_fecha_mixta(pd.Series(fecha_vals, dtype="object"))

            mask = status_s.eq("INGRESO") & fecha_dt.notna()
            d = fecha_dt[mask]
            if d.empty:
                return pd.Series(dtype=int)

            start = pd.to_datetime(_start, errors="coerce")
            end   = pd.to_datetime(_end, errors="coerce")
            if pd.notna(start):
                d = d[d >= start]
            if pd.notna(end):
                d = d[d <= end]

            if d.empty:
                return pd.Series(dtype=int)

            week_start = d.dt.normalize() - pd.to_timedelta(d.dt.weekday, unit="D")
            return week_start.value_counts().sort_index().astype(int)
        except Exception:
            return pd.Series(dtype=int)

    def _get_ingresos_operacion_daily_counts_current_month(_service, _year, _month):
        """
        Lee la hoja del mes dado del spreadsheet INGRESOS_OPERACION_SHEET_ID
        y devuelve conteos DIARIOS para registros donde STATUS (col P) == 'INGRESO'
        y FECHA (col Q) cae en ese año/mes.
        """
        if not INGRESOS_OPERACION_SHEET_ID:
            return pd.Series(dtype=int)
        try:
            sheet_name = f"{mapa_meses[int(_month)]} {str(int(_year))[-2:]}"
            safe_sheet = sheet_name.replace("'", "''")
            rng = f"'{safe_sheet}'!P12:Q"
            resp = (_service.spreadsheets()
                            .values()
                            .get(spreadsheetId=INGRESOS_OPERACION_SHEET_ID, range=rng)
                            .execute())
            rows = resp.get("values", []) or []
            if not rows:
                return pd.Series(dtype=int)
            status_vals = [r[0] if len(r) > 0 else None for r in rows]
            fecha_vals  = [r[1] if len(r) > 1 else None for r in rows]
            status_s = (
                pd.Series(status_vals, dtype="object")
                  .astype(str).str.strip().str.upper()
            )
            fecha_dt = parse_fecha_mixta(pd.Series(fecha_vals, dtype="object"))
            mask = status_s.eq("INGRESO") & fecha_dt.notna()
            d = fecha_dt[mask]
            if d.empty:
                return pd.Series(dtype=int)
            d = d[(d.dt.year == int(_year)) & (d.dt.month == int(_month))]
            if d.empty:
                return pd.Series(dtype=int)
            return d.dt.day.value_counts().sort_index().astype(int)
        except Exception:
            return pd.Series(dtype=int)

    funnel_datasets = []
    for dt_mes in months_for_funnel:
        mes_label = f"{MESES_ES_CORTO[dt_mes.month - 1]} {dt_mes.year}"

        leads_fb_mes = int((entrevista_month == dt_mes).sum()) if not entrevista_month.empty else 0

        asiste_mes = 0
        if (
            not df_funnel_idx.empty
            and dt_mes in df_funnel_idx.index
            and "bin_asiste_entrevista" in df_funnel_idx.columns
        ):
            try:
                asiste_mes = int(df_funnel_idx.at[dt_mes, "bin_asiste_entrevista"])
            except Exception:
                asiste_mes = 0

        # Por ahora se deja en 0; después se conectará el cálculo real.
        ingresos_capacitacion_mes = 0
        try:
            if (not data_reclutamiento.empty
                and "_fecha_ingreso_capa_dt" in data_reclutamiento.columns
                and "_bin_ingreso_capa" in data_reclutamiento.columns):
                _capa_dates = data_reclutamiento.loc[
                    data_reclutamiento["_bin_ingreso_capa"] == 1,
                    "_fecha_ingreso_capa_dt"
                ]
                _capa_month = _capa_dates.dt.to_period("M").dt.to_timestamp()
                ingresos_capacitacion_mes = int((_capa_month == dt_mes).sum())
        except Exception:
            ingresos_capacitacion_mes = 0
        ingresos_operacion_mes = _get_ingresos_operacion_mes(service, dt_mes)        # <-- C6

        funnel_datasets.append({
            "label": mes_label,
            "data": [
                leads_fb_mes,
                asiste_mes,
                ingresos_capacitacion_mes,  # <-- nuevo nivel entre ambos
                ingresos_operacion_mes,
            ],
        })


    
    # Funnel semanal (últimas 8 semanas, inicio lunes)
    funnel_datasets_weekly = []
    try:
        current_week_start = (pd.Timestamp(mx_now.date()) - pd.Timedelta(days=int(mx_now.weekday()))).normalize()
    except Exception:
        current_week_start = pd.Timestamp.now().normalize() - pd.Timedelta(days=int(pd.Timestamp.now().weekday()))

    weeks_for_funnel = [current_week_start - pd.DateOffset(weeks=i) for i in range(7, -1, -1)]  # antiguo -> reciente

    # Ingresos a Operación (desde INGRESOS_OPERACION_SHEET_ID -> hoja del mes actual)
    wk_range_start = weeks_for_funnel[0]
    wk_range_end   = (weeks_for_funnel[-1] + pd.Timedelta(days=6)).normalize()
    _ing_wk_series = _get_ingresos_operacion_week_counts_current_month(service, wk_range_start, wk_range_end)
    ingresos_operacion_week_dict = (
        {pd.to_datetime(k).normalize(): int(v) for k, v in _ing_wk_series.to_dict().items()}
        if isinstance(_ing_wk_series, pd.Series) and not _ing_wk_series.empty
        else {}
    )


    # Ingresos en Capacitación por semana (lunes) usando FECHA DE INGRESO A CAPA + (L14/DIA 1 == SI)
    ingresos_capacitacion_week_dict = {}
    try:
        if (not data_reclutamiento.empty
            and "_fecha_ingreso_capa_dt" in data_reclutamiento.columns
            and "_bin_ingreso_capa" in data_reclutamiento.columns):
            _capa_w = data_reclutamiento.loc[
                data_reclutamiento["_bin_ingreso_capa"] == 1,
                "_fecha_ingreso_capa_dt"
            ]
            _capa_w = pd.to_datetime(_capa_w, errors="coerce")
            _capa_w = _capa_w[_capa_w.notna()]
            if len(_capa_w):
                # Filtra al rango de semanas del funnel
                _capa_w = _capa_w[(_capa_w >= wk_range_start) & (_capa_w <= wk_range_end)]
                _wk_start = _capa_w.dt.normalize() - pd.to_timedelta(_capa_w.dt.weekday, unit="D")
                _wk_counts = _wk_start.value_counts().sort_index().astype(int)
                ingresos_capacitacion_week_dict = {pd.to_datetime(k).normalize(): int(v) for k, v in _wk_counts.to_dict().items()}
    except Exception:
        ingresos_capacitacion_week_dict = {}

    if not data_reclutamiento.empty:
        entrevista_week = (
            data_reclutamiento["entrevista_dt"].dt.normalize()
            - pd.to_timedelta(data_reclutamiento["entrevista_dt"].dt.weekday, unit="D")
            if "entrevista_dt" in data_reclutamiento.columns
            else pd.Series([], dtype="datetime64[ns]")
        )
        ingreso_week = (
            data_reclutamiento["ingreso_dt"].dt.normalize()
            - pd.to_timedelta(data_reclutamiento["ingreso_dt"].dt.weekday, unit="D")
            if "ingreso_dt" in data_reclutamiento.columns
            else pd.Series([], dtype="datetime64[ns]")
        )

        for wk in weeks_for_funnel:
            iso = wk.isocalendar()
            wk_end = (wk + pd.Timedelta(days=6)).normalize()
            wk_label = (
                f"Semana {int(iso.week):02d} {int(iso.year)} - "
                f"{wk.strftime('%d/%m/%Y')} a {wk_end.strftime('%d/%m/%Y')}"
            )

            leads_fb_wk = int((entrevista_week == wk).sum()) if len(entrevista_week) else 0
            asiste_wk = 0
            if "bin_asiste_entrevista" in data_reclutamiento.columns and len(entrevista_week):
                try:
                    asiste_wk = int(data_reclutamiento.loc[entrevista_week == wk, "bin_asiste_entrevista"].sum())
                except Exception:
                    asiste_wk = 0

            ingresos_capa_wk = int(ingresos_capacitacion_week_dict.get(wk.normalize(), 0))
            ingresos_operacion_wk = int(ingresos_operacion_week_dict.get(wk.normalize(), 0))

            funnel_datasets_weekly.append({
                "label": wk_label,
                "data": [leads_fb_wk, asiste_wk, ingresos_capa_wk, ingresos_operacion_wk],
            })
    else:
        # Si no hay datos, aún construye semanas vacías para el selector del modal
        for wk in weeks_for_funnel:
            iso = wk.isocalendar()
            wk_end = (wk + pd.Timedelta(days=6)).normalize()
            wk_label = (
                f"Semana {int(iso.week):02d} {int(iso.year)} - "
                f"{wk.strftime('%d/%m/%Y')} a {wk_end.strftime('%d/%m/%Y')}"
            )
            funnel_datasets_weekly.append({
                "label": wk_label,
                "data": [0, 0, int(ingresos_capacitacion_week_dict.get(wk.normalize(), 0)), int(ingresos_operacion_week_dict.get(wk.normalize(), 0))],
            })


    # === FUNNEL DIARIO (mes actual) ===
    _cur_y = int(mx_now.year)
    _cur_m = int(mx_now.month)
    _days_in_month = calendar.monthrange(_cur_y, _cur_m)[1]
    _day_idx = list(range(1, _days_in_month + 1))
    _day_labels_f = [f"{d:02d}" for d in _day_idx]

    # Leads por día (entrevista_dt, mes actual)
    if not data_reclutamiento.empty and "entrevista_dt" in data_reclutamiento.columns:
        _ent = data_reclutamiento["entrevista_dt"]
        _mask_c = (_ent.dt.year == _cur_y) & (_ent.dt.month == _cur_m)
        _dl_s = _ent[_mask_c].dt.day.value_counts()
        _daily_leads_f = [int(_dl_s.get(d, 0)) for d in _day_idx]

        # Asistieron entrevista por día (fecha_publicacion + bin_asiste_entrevista)
        if "fecha_publicacion" in data_reclutamiento.columns and "bin_asiste_entrevista" in data_reclutamiento.columns:
            _pub = data_reclutamiento["fecha_publicacion"]
            _mp = (_pub.dt.year == _cur_y) & (_pub.dt.month == _cur_m)
            _df_pd = data_reclutamiento[_mp].copy()
            if not _df_pd.empty:
                _df_pd["_d"] = _pub[_mp].dt.day
                _as_s = _df_pd.groupby("_d")["bin_asiste_entrevista"].sum()
                _daily_ent_f = [int(_as_s.get(d, 0)) for d in _day_idx]
            else:
                _daily_ent_f = [0] * len(_day_idx)
        else:
            _daily_ent_f = [0] * len(_day_idx)

        # Ingresos capacitación por día (FECHA DE INGRESO A CAPA + (L14/DIA 1 == SI))
        if "_fecha_ingreso_capa_dt" in data_reclutamiento.columns and "_bin_ingreso_capa" in data_reclutamiento.columns:
            _cap = data_reclutamiento.loc[data_reclutamiento["_bin_ingreso_capa"] == 1, "_fecha_ingreso_capa_dt"]
            _cap = pd.to_datetime(_cap, errors="coerce")
            _mc = (_cap.dt.year == _cur_y) & (_cap.dt.month == _cur_m)
            _dc_s = _cap[_mc].dt.day.value_counts()
            _daily_cap_f = [int(_dc_s.get(d, 0)) for d in _day_idx]
        else:
            _daily_cap_f = [0] * len(_day_idx)
    else:
        _daily_leads_f = [0] * len(_day_idx)
        _daily_ent_f   = [0] * len(_day_idx)
        _daily_cap_f   = [0] * len(_day_idx)

    # Ingresos operación por día (desde hoja de operaciones)
    _ops_daily = _get_ingresos_operacion_daily_counts_current_month(service, _cur_y, _cur_m)
    _daily_op_f = [int(_ops_daily.get(d, 0)) for d in _day_idx]

    _MESES_COMPLETOS_F = [
        'Enero','Febrero','Marzo','Abril','Mayo','Junio',
        'Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre'
    ]
    funnel_daily_data = {
        "labels":       _day_labels_f,
        "leads":        _daily_leads_f,
        "entrevista":   _daily_ent_f,
        "capacitacion": _daily_cap_f,
        "operacion":    _daily_op_f,
        "month_label":  f"{_MESES_COMPLETOS_F[_cur_m - 1]} {_cur_y}"
    }

################################ COBERTURA DE CARTERA
 #####################################

    def _filtrar_didi_solo_agentes(_df: pd.DataFrame) -> pd.DataFrame:
        """
        Regla especial: Para CAMPAÑA == 'DIDI', solo conservar filas donde PUESTO contenga 'AGENTE'.
        Para el resto de campañas, no filtra nada.
        """
        if _df is None or _df.empty:
            return _df

        if "CAMPAÑA" not in _df.columns or "PUESTO" not in _df.columns:
            # Si no existe PUESTO (o CAMPAÑA), no aplicamos filtro para no romper el flujo.
            return _df

        camp = _df["CAMPAÑA"].astype(str).str.strip().str.upper()
        puesto = _df["PUESTO"].astype(str).str.upper()

        is_didi = camp.eq("DIDI")
        is_agente = puesto.str.contains("AGENTE", na=False)

        # Mantener todo lo que NO sea DIDI, y de DIDI solo AGENTE
        return _df.loc[~is_didi | is_agente].copy()


    df_act = df_plantilla_activa.copy()
    df_act = _filtrar_didi_solo_agentes(df_act)  # <-- AQUI aplica el filtro para DIDI
    df_act['FECHA_INGRESO'] = pd.to_datetime(df_act['FECHA DE INGRESO'], dayfirst=True, errors='coerce')
    df_act = df_act.dropna(subset=['FECHA_INGRESO'])
    df_act['FECHA_BAJA'] = pd.NaT
    df_act = df_act[['CAMPAÑA', 'FECHA_INGRESO', 'FECHA_BAJA']]

    df_baj = df_plantilla_bajas.copy()
    df_baj = _filtrar_didi_solo_agentes(df_baj)  # <-- AQUI también para bajas de DIDI
    df_baj['FECHA_INGRESO'] = pd.to_datetime(df_baj['FECHA DE INGRESO'], dayfirst=True, errors='coerce')
    df_baj['FECHA_BAJA']    = pd.to_datetime(df_baj['BAJA'], dayfirst=True, errors='coerce')
    df_baj = df_baj.dropna(subset=['FECHA_INGRESO'])
    df_baj = df_baj[['CAMPAÑA', 'FECHA_INGRESO', 'FECHA_BAJA']]

    df_personal = pd.concat([df_act, df_baj], ignore_index=True)


    month_map = {
        'ENERO': 1, 'FEBRERO': 2, 'MARZO': 3, 'ABRIL': 4, 'MAYO': 5, 'JUNIO': 6,
        'JULIO': 7, 'AGOSTO': 8, 'SEPTIEMBRE': 9, 'OCTUBRE': 10, 'NOVIEMBRE': 11, 'DICIEMBRE': 12
    }
    df_aut = df_plantilla_autorizada.copy()
    df_aut['MES_UPPER'] = df_aut['MES'].astype(str).str.strip().str.upper()
    df_aut['MES_NUM']   = df_aut['MES_UPPER'].map(month_map)
    df_aut['AÑO_NUM']   = pd.to_numeric(df_aut['AÑO'], errors='coerce')
    df_aut['PERSONAL']  = pd.to_numeric(df_aut['PERSONAL'], errors='coerce')
    df_aut = df_aut.dropna(subset=['MES_NUM', 'AÑO_NUM', 'PERSONAL'])
    df_aut['MES_NUM']   = df_aut['MES_NUM'].astype(int)
    df_aut['AÑO_NUM']   = df_aut['AÑO_NUM'].astype(int)
    df_aut['AÑO']       = df_aut['AÑO_NUM'].astype(int)
    df_aut['PERSONAL']  = df_aut['PERSONAL'].astype(int)
    df_aut = (df_aut.groupby(['CAMPAÑA', 'MES', 'AÑO', 'MES_NUM', 'AÑO_NUM'], as_index=False)['PERSONAL'].sum())
    df_aut['FECHA_CORTE'] = pd.to_datetime({
        'year': df_aut['AÑO_NUM'],
        'month': df_aut['MES_NUM'],
        'day': 1
    }) + pd.offsets.MonthEnd(0)
    df_aut['_key']       = 1
    df_personal['_key']  = 1
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
    )
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
    cobertura_labels      = coverage_df['MES_ANO'].tolist()
    cobertura_values      = coverage_df['% COBERTURA'].tolist()
    cobertura_autorizados = coverage_df['PLANTILLA AUTORIZADA'].tolist()
    cobertura_activos     = coverage_df['TOTAL ACTIVOS'].tolist()

    # Mini-chart: últimos 3 meses para mostrar junto al KPI de Personal Autorizado
    cobertura_labels_mini      = cobertura_labels[-3:]
    cobertura_values_mini      = cobertura_values[-3:]
    cobertura_autorizados_mini = cobertura_autorizados[-3:]
    cobertura_activos_mini     = cobertura_activos[-3:]
    df_detalle = df_cobertura_de_cartera.copy()
    df_detalle['MES_NUM'] = df_detalle['MES'].map(month_map)
    df_detalle = df_detalle.sort_values(
        ['AÑO', 'MES_NUM', 'CAMPAÑA'],
        ascending=[False, False, True]
    )
    cobertura_detalle_rows = df_detalle.to_dict(orient='records')

    ################################ INGRESOS VS BAJAS ############################################

    def _to_datetime(series, dayfirst=True):
        return pd.to_datetime(series, dayfirst=dayfirst, errors="coerce")

    # Ventana: últimos 12 meses **incluyendo el mes actual** (mes actual a la fecha)
    # Ej.: si hoy es 2026-01-01, el rango queda 2025-02-01 .. 2026-01-01 (y los meses: feb-2025 .. ene-2026)
    _month_start = mx_now.normalize().replace(day=1)
    # Nota: usamos fin del día de hoy para no perder registros con hora.
    RANGE_END   = (mx_now.normalize() + pd.Timedelta(days=1)) - pd.Timedelta(microseconds=1)
    RANGE_START = _month_start - pd.DateOffset(months=11)
    months_12   = pd.date_range(start=RANGE_START, periods=12, freq="MS")

    ingresos_activos        = _to_datetime(df_plantilla_activa["FECHA DE INGRESO"], dayfirst=True)
    bajas_baja_dates        = _to_datetime(df_plantilla_bajas["BAJA"], dayfirst=True)
    bajas_ingreso_dates     = _to_datetime(df_plantilla_bajas["FECHA DE INGRESO"], dayfirst=True)
    mask_bajas_validas      = bajas_ingreso_dates.notna()
    bajas_baja_dates_validas = bajas_baja_dates[mask_bajas_validas]

    # Ingresos = ingresos de activos + ingresos de bajas (histórico)
    ingresos_todos = pd.concat([ingresos_activos, bajas_ingreso_dates], ignore_index=True)

    def _filter_range(dates, start, end):
        d = pd.to_datetime(dates, errors="coerce")
        d = d[d.notna()]
        return d[(d >= start) & (d <= end)]

    def _weekly_counts_from_dates(dates, start, end):
        d = _filter_range(dates, start, end)
        if d.empty:
            # índice completo de semanas en rango
            start_monday = start - pd.Timedelta(days=int(start.weekday()))
            end_monday   = end   - pd.Timedelta(days=int(end.weekday()))
            idx = pd.date_range(start=start_monday, end=end_monday, freq="W-MON")
            return pd.Series(0, index=idx)

        week_start = d - pd.to_timedelta(d.dt.weekday, unit="D")  # lunes
        counts = week_start.value_counts().sort_index()

        start_monday = start - pd.Timedelta(days=int(start.weekday()))
        end_monday   = end   - pd.Timedelta(days=int(end.weekday()))
        idx = pd.date_range(start=start_monday, end=end_monday, freq="W-MON")
        return counts.reindex(idx, fill_value=0).astype(int)

    def _weekly_counts_for_year(dates, year_start, year_end):
        """Conteo semanal dentro de un año, incluyendo una primera 'semana parcial' desde year_start hasta el primer lunes."""
        year_start = pd.to_datetime(year_start)
        year_end   = pd.to_datetime(year_end)
        d = _filter_range(dates, year_start, year_end)

        first_monday = year_start + pd.Timedelta(days=(7 - int(year_start.weekday())) % 7)

        def _build_index():
            if first_monday == year_start:
                return pd.date_range(start=year_start, end=year_end, freq="W-MON")
            return pd.DatetimeIndex([year_start]).append(
                pd.date_range(start=first_monday, end=year_end, freq="W-MON")
            )

        if d.empty:
            idx = _build_index()
            return pd.Series(0, index=idx).astype(int)

        # Asignación de bucket: antes del primer lunes -> year_start; después -> lunes de esa semana
        d = pd.to_datetime(d, errors="coerce")
        d = d[d.notna()]
        if first_monday > year_start:
            mask_first = d < first_monday
            buckets = pd.Series(index=d.index, dtype="datetime64[ns]")
            buckets.loc[mask_first] = year_start
            buckets.loc[~mask_first] = d[~mask_first] - pd.to_timedelta(d[~mask_first].dt.weekday, unit="D")
        else:
            buckets = d - pd.to_timedelta(d.dt.weekday, unit="D")

        counts = buckets.value_counts().sort_index()
        idx = _build_index()
        return counts.reindex(idx, fill_value=0).astype(int)

    def _monthly_counts_from_dates(dates, month_starts, start, end):
        d = _filter_range(dates, start, end)
        if d.empty:
            return pd.Series(0, index=month_starts).astype(int)
        month_key = d.dt.to_period("M").dt.to_timestamp()
        counts = month_key.value_counts().sort_index()
        return counts.reindex(month_starts, fill_value=0).astype(int)

    # Semanal (últimos 12 meses)
    ingresos_week_counts = _weekly_counts_from_dates(ingresos_todos, RANGE_START, RANGE_END)
    bajas_week_counts    = _weekly_counts_from_dates(bajas_baja_dates_validas, RANGE_START, RANGE_END)

    week_index      = ingresos_week_counts.index
    week_labels     = [pd.to_datetime(w).strftime("%d/%m/%y") for w in week_index]
    # Etiquetas ISO para reutilizar en otros gráficos (ej. inversión en pauta)
    week_labels_iso = [pd.to_datetime(w).strftime("%Y-%m-%d") for w in week_index]
    ingresos_pauta_week_labels = week_labels_iso
    ingresos_pauta_week_values = [int(x) for x in ingresos_week_counts.values]

    week_month_nums = [int(pd.to_datetime(w).month) for w in week_index]

    df_semana = pd.DataFrame({
        "SemanaLabel": week_labels,
        "Ingresos": ingresos_week_counts.values,
        "Bajas": bajas_week_counts.reindex(week_index, fill_value=0).values
    })

    # Mensual (últimos 12 meses)
    ingresos_month_counts = _monthly_counts_from_dates(ingresos_todos, months_12, RANGE_START, RANGE_END)
    bajas_month_counts    = _monthly_counts_from_dates(bajas_baja_dates_validas, months_12, RANGE_START, RANGE_END)

    # Etiquetas YYYY-MM para reutilizar en otros gráficos (ej. inversión en pauta)
    ingresos_pauta_month_labels = [f"{m.year}-{m.month:02d}" for m in months_12]
    ingresos_pauta_month_values = [int(x) for x in ingresos_month_counts.values]


    meses_labels_12 = [f"{MESES_ES_CORTO[m.month - 1]} {m.year}" for m in months_12]
    ingresos_mes    = ingresos_month_counts.tolist()
    bajas_mes       = bajas_month_counts.tolist()

    # Diario (drilldown por cada uno de los 12 meses del rango)
    def _daily_counts_for_month(dates, year, month):
        d = pd.to_datetime(dates, errors="coerce")
        d = d[d.notna()]
        d = d[(d.dt.year == int(year)) & (d.dt.month == int(month))]
        days_in_month = calendar.monthrange(int(year), int(month))[1]
        idx = pd.Index(range(1, days_in_month + 1), name="Dia")
        counts = d.dt.day.value_counts().sort_index().reindex(idx, fill_value=0).astype(int)
        labels = [f"{day:02d}" for day in idx.tolist()]
        return labels, counts.tolist()

    ingresos_bajas_daily = {}
    for i, mstart in enumerate(months_12, start=1):
        y, m = int(mstart.year), int(mstart.month)
        dlabels, dingresos = _daily_counts_for_month(ingresos_todos, y, m)
        _, dbajas          = _daily_counts_for_month(bajas_baja_dates_validas, y, m)
        ingresos_bajas_daily[str(i)] = {"labels": dlabels, "ingresos": dingresos, "bajas": dbajas}

    # === Histórico (Acumulado semanal por año seleccionado) ===
    historico_year_start = pd.Timestamp(int(historico_year_selected), 1, 1)
    if int(historico_year_selected) == int(mx_now.year):
        historico_year_end = RANGE_END
        historico_periodo_text = f"{int(historico_year_selected)} (al {mx_now.strftime('%d/%m/%Y')})"
    else:
        historico_year_end = pd.Timestamp(int(historico_year_selected), 12, 31, 23, 59, 59, 999999)
        historico_periodo_text = f"{int(historico_year_selected)} (enero - diciembre)"

    ingresos_week_year = _weekly_counts_for_year(ingresos_todos, historico_year_start, historico_year_end)
    bajas_week_year    = _weekly_counts_for_year(bajas_baja_dates_validas, historico_year_start, historico_year_end)

    week_cum_index      = ingresos_week_year.index
    week_cum_labels     = [pd.to_datetime(w).strftime("%d/%m/%y") for w in week_cum_index]
    week_cum_month_nums = [int(pd.to_datetime(w).month) for w in week_cum_index]

    df_semana_cum = pd.DataFrame({
        "SemanaLabel": week_cum_labels,
        "Ingresos": ingresos_week_year.values,
        "Bajas": bajas_week_year.reindex(week_cum_index, fill_value=0).values,
    })
    df_semana_cum["Ingresos acumulados"] = df_semana_cum["Ingresos"].cumsum()
    df_semana_cum["Bajas acumuladas"]    = df_semana_cum["Bajas"].cumsum()

    # Lista de años disponibles para el selector (basado en los datos, más el año actual)
    _all_years_series = pd.concat([
        pd.to_datetime(ingresos_todos, errors="coerce"),
        pd.to_datetime(bajas_baja_dates_validas, errors="coerce"),
    ], ignore_index=True).dt.year.dropna().astype(int)
    historico_years = sorted(set(_all_years_series.tolist() + [int(_current_year)]), reverse=True)

    # Etiqueta de periodo para UI
    ingresos_bajas_periodo = f"{meses_labels_12[0]} - {meses_labels_12[-1]}"
    ################################ LEADS POR RECLUTADOR ########################################

    if not data_reclutamiento.empty and "fecha_publicacion" in data_reclutamiento.columns:
        data_reclutamiento["fecha_publicacion"].replace("", pd.NA, inplace=True)
        data_reclutamiento["fecha_publicacion"] = pd.to_datetime(
            data_reclutamiento["fecha_publicacion"], errors="coerce"
        )
        if "entrevista" in data_reclutamiento.columns:
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
        meses_leads     = sorted(df_leads_all["mes"].dropna().unique())
        etiquetas_leads = [pd.to_datetime(m).strftime("%Y-%m-%d") for m in meses_leads]
        reclutadores    = sorted(df_leads_all["reclutador"].dropna().unique())
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
    else:
        etiquetas_leads      = []
        leads_por_reclutador = {}
        leads_total          = []

    ################################ CANALES INGRESO / PAUTA #####################################

    # PLANTILLA
    df = df_plantilla_activa.copy()
    df['FECHA DE INGRESO'] = pd.to_datetime(
        df['FECHA DE INGRESO'],
        dayfirst=True,
        errors='coerce'
    )
    df = df.dropna(subset=['FECHA DE INGRESO'])
    # Últimos 12 meses completos
    df = df[(df['FECHA DE INGRESO'] >= RANGE_START) & (df['FECHA DE INGRESO'] <= RANGE_END)]
    df['anio_mes'] = df['FECHA DE INGRESO'].dt.strftime('%Y-%m')
    print(df_plantilla_activa.columns.tolist())

    df_grouped = (
        df
        .groupby(['anio_mes', 'FUENTE'])
        .size()
        .reset_index(name='CANTIDAD')
    )
    canales_plantilla_por_mes      = {}
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
    if not data_reclutamiento.empty and "fecha_publicacion" in data_reclutamiento.columns:
        dr_canales = data_reclutamiento.copy()
        dr_canales["fecha_publicacion"] = pd.to_datetime(
            dr_canales["fecha_publicacion"],
            errors="coerce"
        )
        # Últimos 12 meses completos
        dr_canales = dr_canales[
            (dr_canales["fecha_publicacion"] >= RANGE_START) &
            (dr_canales["fecha_publicacion"] <= RANGE_END)
        ]
        dr_canales["anio_mes"] = dr_canales["fecha_publicacion"].dt.strftime("%Y-%m")

        if "folio" in dr_canales.columns:
            mask_referido = dr_canales["folio"] == "Referido"
        else:
            mask_referido = pd.Series(False, index=dr_canales.index)

        df_ref = dr_canales[mask_referido].copy()
        agg_ref = (
            df_ref
            .groupby(["canal", "anio_mes"])
            .size()
            .reset_index(name="folios_unicos")
        )

        df_no_ref = dr_canales[~mask_referido].copy()
        if {"canal", "folio", "anio_mes"}.issubset(df_no_ref.columns):
            df_no_ref = df_no_ref.drop_duplicates(subset=["canal", "folio", "anio_mes"])
            agg_no_ref = (
                df_no_ref
                .groupby(["canal", "anio_mes"])["folio"]
                .nunique()
                .reset_index(name="folios_unicos")
            )
            resumen = pd.concat([agg_ref, agg_no_ref], ignore_index=True)
        else:
            resumen = agg_ref.copy()

        resumen = (
            resumen
            .groupby(["canal", "anio_mes"])["folios_unicos"]
            .sum()
            .reset_index()
        )

        canales_pauta_por_mes      = {}
        canales_pauta_meses_labels = []
        if not resumen.empty:
            resumen["anio_mes"] = resumen["anio_mes"].astype(str)
            resumen = resumen.dropna(subset=["anio_mes", "canal"])
            meses_unicos_pauta = sorted(resumen["anio_mes"].unique())
            canales_pauta_meses_labels = list(meses_unicos_pauta)
            for m in meses_unicos_pauta:
                df_m = resumen[resumen["anio_mes"] == m]
                canales_pauta_por_mes[m] = dict(
                    zip(df_m["canal"], df_m["folios_unicos"])
                )
    else:
        canales_pauta_por_mes      = {}
        canales_pauta_meses_labels = []

    ################################ INVERSION EN PAUTA ##########################################

    if not data_reclutamiento.empty and "monto_pagado" in data_reclutamiento.columns and "fecha_publicacion" in data_reclutamiento.columns:
        data_reclutamiento["monto_pagado"] = pd.to_numeric(
            data_reclutamiento["monto_pagado"],
            errors="coerce"
        )
        df_pagos = data_reclutamiento.dropna(subset=["monto_pagado", "fecha_publicacion"]).copy()
        df_pagos["fecha_publicacion"] = pd.to_datetime(df_pagos["fecha_publicacion"], errors="coerce")
        df_pagos = df_pagos.sort_values("fecha_publicacion")
        if "folio" in df_pagos.columns:
            df_pagos_unicos = df_pagos.drop_duplicates(subset=["folio"], keep="first")
        else:
            df_pagos_unicos = df_pagos.copy()
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
    else:
        pagos_semanales_labels = []
        pagos_semanales_values = []
        pagos_mensuales_labels = []
        pagos_mensuales_values = []

    ################################ MOTIVOS DE BAJA #############################################

    # Motivos (últimos 12 meses completos)
    df = df_plantilla_bajas.copy()
    df["BAJA"] = pd.to_datetime(df["BAJA"], dayfirst=True, errors="coerce")
    df = df[(df["BAJA"] >= RANGE_START) & (df["BAJA"] <= RANGE_END)]

    serie = df["MOTIVO"].fillna("SIN MOTIVO")
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

    # TABLA motivos baja por mes (últimos 12 meses)
    df_tab = df_plantilla_bajas.copy()
    df_tab["BAJA"] = pd.to_datetime(df_tab["BAJA"], dayfirst=True, errors="coerce")
    df_tab = df_tab[(df_tab["BAJA"] >= RANGE_START) & (df_tab["BAJA"] <= RANGE_END)].copy()

    meses_12_labels = [f"{MESES_ES_CORTO[m.month - 1]} {m.year}" for m in months_12]

    df_tab["Mes"] = df_tab["BAJA"].dt.to_period("M").dt.to_timestamp()
    df_tab["MesLabel"] = df_tab["Mes"].apply(
        lambda d: f"{MESES_ES_CORTO[int(d.month) - 1]} {int(d.year)}" if pd.notna(d) else ""
    )

    df_tab["MesLabel"] = pd.Categorical(
        df_tab["MesLabel"],
        categories=meses_12_labels,
        ordered=True
    )

    pivot = pd.crosstab(
        df_tab["MOTIVO"].fillna("SIN MOTIVO"),
        df_tab["MesLabel"],
        margins=True,
        margins_name="Total"
    )
    pivot = pivot.reindex(columns=meses_12_labels + ["Total"]).fillna(0).astype(int)

    if "Total" in pivot.index:
        total_row = pivot.loc[["Total"]]
        cuerpo    = pivot.drop(index="Total")
        cuerpo    = cuerpo.sort_values(by="Total", ascending=False)
        pivot_sorted = pd.concat([cuerpo, total_row])
    else:
        pivot_sorted = pivot.sort_values(by="Total", ascending=False)

    tabla = pivot_sorted.reset_index().rename(columns={"index": "MOTIVO"})
    tabla_motivos_baja_rows    = tabla.to_dict(orient="records")
    tabla_motivos_baja_columns = [col for col in tabla.columns if col != "MOTIVO"]

    # Antigüedad promedio de bajas (últimos 12 meses)
    df_ant = df_plantilla_bajas.copy()
    for col in ["FECHA DE INGRESO", "BAJA"]:
        df_ant[col] = pd.to_datetime(df_ant[col], dayfirst=True, errors="coerce")

    mask_valid = (
        df_ant["BAJA"].notna() &
        df_ant["FECHA DE INGRESO"].notna() &
        (df_ant["BAJA"] >= RANGE_START) &
        (df_ant["BAJA"] <= RANGE_END)
    )
    df_calc = (
        df_ant.loc[mask_valid]
        .assign(
            antiguedad_meses=lambda d: (d["BAJA"] - d["FECHA DE INGRESO"]).dt.days / 30.4375,
            Mes=lambda d: d["BAJA"].dt.to_period("M").dt.to_timestamp()
        )
    )

    prom = (
        df_calc.groupby("Mes", as_index=False)["antiguedad_meses"]
        .mean()
    )

    prom = prom.set_index("Mes").reindex(months_12).reset_index().rename(columns={"index": "Mes"})
    prom["MesLabel"] = prom["Mes"].apply(lambda d: f"{MESES_ES_CORTO[int(d.month)-1]} {int(d.year)}" if pd.notna(d) else "")
    antiguedad_bajas_labels = prom["MesLabel"].tolist()
    antiguedad_bajas_values = (
        prom["antiguedad_meses"]
        .round(1)
        .fillna(0)
        .tolist()
    )
    ############################################# VARIABLES A ENVIAR ##############################################

    return render_template(
        "dashboard-de-capital-humano.html",
        # KPIs
        plantilla_autorizada_total=sum_plantilla_general_autorizada,
        plantilla_autorizada_operativa=sum_plantilla_operativa_autorizada,
        plantilla_autorizada_administrativa=sum_plantilla_administrativa_autorizada,
        genero_cantidad=genero_cantidad_dict,
        genero_porcentaje=genero_porcentaje_dict,
        nomina_total_general=total_general_nomina,
        nomina_total_general_fmt=total_general_nomina_fmt,
        nomina_estructura=nomina_estructura,
        nomina_operacion=nomina_operacion,
        nomina_estructura_fmt=nomina_estructura_fmt,
        nomina_operacion_fmt=nomina_operacion_fmt,
        nomina_por_area=nomina_por_area_dict,
        nomina_por_area_fmt=nomina_por_area_fmt_dict,
        contratos_total=contratos_total,
        contratos_estructura=contratos_estructura,
        contratos_operacion=contratos_operacion,
        contratos_por_area=contratos_por_area_dict,
        # Funnel (mensual + semanal + diario)
        funnel_labels=funnel_labels,
        funnel_datasets=funnel_datasets,
        funnel_datasets_weekly=funnel_datasets_weekly,
        funnel_daily_data=funnel_daily_data,
        # Cobertura cartera
        cobertura_labels=cobertura_labels,
        cobertura_values=cobertura_values,
        cobertura_autorizados=cobertura_autorizados,
        cobertura_activos=cobertura_activos,
        cobertura_detalle_rows=cobertura_detalle_rows,
        # Mini-chart cobertura (últimos 3 meses)
        cobertura_labels_mini=cobertura_labels_mini,
        cobertura_values_mini=cobertura_values_mini,
        cobertura_autorizados_mini=cobertura_autorizados_mini,
        cobertura_activos_mini=cobertura_activos_mini,
        # Ingresos vs bajas semanal / mensual
        # Ingresos vs bajas semanal / mensual (últimos 12 meses)
        ingresos_bajas_periodo=ingresos_bajas_periodo,
        ingresos_bajas_week_labels=df_semana["SemanaLabel"].astype(str).tolist(),
        ingresos_bajas_week_ingresos=df_semana["Ingresos"].astype(int).tolist(),
        ingresos_bajas_week_bajas=df_semana["Bajas"].astype(int).tolist(),
        ingresos_bajas_week_month_numbers=week_month_nums,
        ingresos_bajas_month_labels=meses_labels_12,
        ingresos_bajas_month_ingresos=[int(x) for x in ingresos_mes],
        ingresos_bajas_month_bajas=[int(x) for x in bajas_mes],
        ingresos_bajas_daily=ingresos_bajas_daily,
        # Ingresos vs bajas acumulado
        # Ingresos vs bajas acumulado (últimos 12 meses)
        ingresos_bajas_cum_week_labels=df_semana_cum["SemanaLabel"].astype(str).tolist(),
        ingresos_bajas_cum_ingresos=df_semana_cum["Ingresos acumulados"].astype(int).tolist(),
        ingresos_bajas_cum_bajas=df_semana_cum["Bajas acumuladas"].astype(int).tolist(),
        ingresos_bajas_cum_week_month_numbers=week_cum_month_nums,
        historico_years=historico_years,
        historico_year_selected=historico_year_selected,
        historico_periodo_text=historico_periodo_text,
        # Leads por reclutador
        leads_labels=etiquetas_leads,
        leads_por_reclutador=leads_por_reclutador,
        leads_total=leads_total,
        # Canales de ingreso
        canales_plantilla_por_mes=canales_plantilla_por_mes,
        canales_plantilla_meses_labels=canales_plantilla_meses_labels,
        canales_pauta_por_mes=canales_pauta_por_mes,
        canales_pauta_meses_labels=canales_pauta_meses_labels,
        # Inversión pauta
        pagos_semanales_labels=pagos_semanales_labels,
        pagos_semanales_values=pagos_semanales_values,
        pagos_mensuales_labels=pagos_mensuales_labels,
        pagos_mensuales_values=pagos_mensuales_values,
        ingresos_pauta_week_labels=ingresos_pauta_week_labels,
        ingresos_pauta_week_values=ingresos_pauta_week_values,
        ingresos_pauta_month_labels=ingresos_pauta_month_labels,
        ingresos_pauta_month_values=ingresos_pauta_month_values,
        # Motivos de bajas
        motivos_baja_labels=motivos_baja_labels,
        motivos_baja_values=motivos_baja_values,
        tabla_motivos_baja_rows=tabla_motivos_baja_rows,
        tabla_motivos_baja_columns=tabla_motivos_baja_columns,
        # Antigüedad promedio de bajas
        antiguedad_bajas_labels=antiguedad_bajas_labels,
        antiguedad_bajas_values=antiguedad_bajas_values,
        # Filtro campaña
        campanias=campanias,
        campania_seleccionada=campania_seleccionada,
    )

@app.errorhandler(500)
def handle_500(e):
    return render_template("error_page.html"), 500

@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, HTTPException):
        return e
    return render_template("error_page.html"), 500

######################################### EJECUTADOR #############################################

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
