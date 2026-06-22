
# ============================================================
# PREDICTOR MUNDIAL 2026 - MODELO EDUCATIVO DE ANALÍTICA FÚTBOL
# Autor: generado para Spyder / Streamlit
# Objetivo:
#   - Descargar datos históricos de partidos internacionales.
#   - Actualizar con datos del Mundial 2026 desde API-Football si tienes API key.
#   - Entrenar varios modelos de Machine Learning.
#   - Comparar modelos con matriz de confusión y métricas.
#   - Generar probabilidades de resultado y marcadores probables.
#
# IMPORTANTE:
#   Este código es para análisis estadístico/educativo. No usa cuotas ni casas de apuestas.
# ============================================================

import os
import json
import math
import warnings
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

from scipy.stats import poisson

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    confusion_matrix,
    classification_report,
    brier_score_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor, Ridge

warnings.filterwarnings("ignore")

try:
    import streamlit as st
    import streamlit.components.v1 as components
    STREAMLIT_OK = True
except Exception:
    STREAMLIT_OK = False


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

DATA_DIR = Path("datos")
DATA_DIR.mkdir(exist_ok=True)

CACHE_DIR = DATA_DIR / "cache_api"
CACHE_DIR.mkdir(exist_ok=True)


# Sedes aproximadas Mundial 2026 para clima y altitud.
# Valores aproximados para análisis educativo. El usuario puede escoger la sede en la app.
WORLD_CUP_2026_VENUES = {
    "Atlanta, United States": {"country": "United States", "city": "Atlanta", "lat": 33.755, "lon": -84.401, "altitude_m": 320},
    "Boston/Foxborough, United States": {"country": "United States", "city": "Foxborough", "lat": 42.091, "lon": -71.264, "altitude_m": 90},
    "Dallas/Arlington, United States": {"country": "United States", "city": "Arlington", "lat": 32.747, "lon": -97.092, "altitude_m": 190},
    "Houston, United States": {"country": "United States", "city": "Houston", "lat": 29.684, "lon": -95.411, "altitude_m": 15},
    "Kansas City, United States": {"country": "United States", "city": "Kansas City", "lat": 39.049, "lon": -94.484, "altitude_m": 265},
    "Los Angeles/Inglewood, United States": {"country": "United States", "city": "Inglewood", "lat": 33.953, "lon": -118.339, "altitude_m": 40},
    "Miami, United States": {"country": "United States", "city": "Miami Gardens", "lat": 25.958, "lon": -80.239, "altitude_m": 2},
    "New York/New Jersey, United States": {"country": "United States", "city": "East Rutherford", "lat": 40.813, "lon": -74.074, "altitude_m": 10},
    "Philadelphia, United States": {"country": "United States", "city": "Philadelphia", "lat": 39.901, "lon": -75.168, "altitude_m": 12},
    "San Francisco Bay Area, United States": {"country": "United States", "city": "Santa Clara", "lat": 37.403, "lon": -121.970, "altitude_m": 5},
    "Seattle, United States": {"country": "United States", "city": "Seattle", "lat": 47.595, "lon": -122.331, "altitude_m": 60},
    "Guadalajara, Mexico": {"country": "Mexico", "city": "Guadalajara", "lat": 20.666, "lon": -103.341, "altitude_m": 1566},
    "Mexico City, Mexico": {"country": "Mexico", "city": "Mexico City", "lat": 19.303, "lon": -99.150, "altitude_m": 2240},
    "Monterrey, Mexico": {"country": "Mexico", "city": "Monterrey", "lat": 25.668, "lon": -100.244, "altitude_m": 540},
    "Toronto, Canada": {"country": "Canada", "city": "Toronto", "lat": 43.633, "lon": -79.419, "altitude_m": 76},
    "Vancouver, Canada": {"country": "Canada", "city": "Vancouver", "lat": 49.276, "lon": -123.111, "altitude_m": 10},
}


PLAYER_RATINGS_FILE = DATA_DIR / "jugadores_rating.csv"

HISTORICAL_RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

MANUAL_WC_FILE = DATA_DIR / "mundial_2026_manual.csv"
INTERNAL_WC_CALENDAR_FILE = DATA_DIR / "calendario_mundial_2026.csv"

API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
WORLD_CUP_LEAGUE_ID = 1
WORLD_CUP_SEASON = 2026

RANDOM_STATE = 42

# La precisión acumulada del Mundial se empieza a medir desde este partido hacia adelante.
# El usuario pidió iniciar desde Japan vs Tunisia/Tunisia vs Japan.
PRECISION_START_FIXTURE_ID = 66456974
PRECISION_START_TEAMS = {"Japan", "Tunisia"}
PRECISION_START_LABEL = "Japan vs Tunisia"
PRECISION_START_DATE = "2026-06-21"
DEFAULT_MANUAL_FIXTURE_ID = "236"

# Equipos del Mundial 2026 usados por el modelo.
WC_2026_TEAMS = [
    "Mexico", "South Africa", "South Korea", "Czechia",
    "Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland",
    "Brazil", "Morocco", "Haiti", "Scotland",
    "United States", "Paraguay", "Australia", "Turkey",
    "Germany", "Curacao", "Ivory Coast", "Ecuador",
    "Netherlands", "Japan", "Sweden", "Tunisia",
    "Belgium", "Egypt", "Iran", "New Zealand",
    "Spain", "Cape Verde", "Saudi Arabia", "Uruguay",
    "France", "Senegal", "Iraq", "Norway",
    "Argentina", "Algeria", "Austria", "Jordan",
    "Portugal", "DR Congo", "Uzbekistan", "Colombia",
    "England", "Croatia", "Ghana", "Panama",
]

TEAM_NAME_ES = {
    "Mexico": "México", "South Africa": "Sudáfrica", "South Korea": "Corea del Sur",
    "Czechia": "Chequia", "Canada": "Canadá",
    "Bosnia and Herzegovina": "Bosnia y Herzegovina", "Qatar": "Catar",
    "Switzerland": "Suiza", "Brazil": "Brasil", "Morocco": "Marruecos",
    "Haiti": "Haití", "Scotland": "Escocia", "United States": "Estados Unidos",
    "Paraguay": "Paraguay", "Australia": "Australia", "Turkey": "Turquía",
    "Germany": "Alemania", "Curacao": "Curazao", "Ivory Coast": "Costa de Marfil",
    "Ecuador": "Ecuador", "Netherlands": "Países Bajos", "Japan": "Japón",
    "Sweden": "Suecia", "Tunisia": "Túnez", "Belgium": "Bélgica",
    "Egypt": "Egipto", "Iran": "Irán", "New Zealand": "Nueva Zelanda",
    "Spain": "España", "Cape Verde": "Cabo Verde", "Saudi Arabia": "Arabia Saudita",
    "Uruguay": "Uruguay", "France": "Francia", "Senegal": "Senegal",
    "Iraq": "Irak", "Norway": "Noruega", "Argentina": "Argentina",
    "Algeria": "Argelia", "Austria": "Austria", "Jordan": "Jordania",
    "Portugal": "Portugal", "DR Congo": "RD Congo", "Uzbekistan": "Uzbekistán",
    "Colombia": "Colombia", "England": "Inglaterra", "Croatia": "Croacia",
    "Ghana": "Ghana", "Panama": "Panamá",
}




# Grupos Mundial 2026 usados para simulación Monte Carlo aproximada.
WC_2026_GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

TEAM_ALIASES = {
    "USA": "United States",
    "United States of America": "United States",
    "USMNT": "United States",
    "México": "Mexico",
    "Mexico": "Mexico",
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "South Korea": "South Korea",
    "Czech Republic": "Czechia",
    "Czechia": "Czechia",
    "Bosnia": "Bosnia and Herzegovina",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Congo DR": "DR Congo",
    "Democratic Republic of Congo": "DR Congo",
    "DR Congo": "DR Congo",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Ivory Coast": "Ivory Coast",
    "Curacao": "Curacao",
    "Curaçao": "Curacao",
    "Netherlands": "Netherlands",
    "Holland": "Netherlands",
    "Türkiye": "Turkey",
    "Turkey": "Turkey",
    "Cape Verde Islands": "Cape Verde",
    "Cape Verde": "Cape Verde",
}

def normalize_team_name(name):
    """Normaliza nombres entre API, CSV histórico y app."""
    raw = safe_name(name)
    return TEAM_ALIASES.get(raw, raw)



TEAM_FLAGS = {
    "Mexico": "🇲🇽", "South Africa": "🇿🇦", "South Korea": "🇰🇷", "Czechia": "🇨🇿",
    "Canada": "🇨🇦", "Bosnia and Herzegovina": "🇧🇦", "Qatar": "🇶🇦", "Switzerland": "🇨🇭",
    "Brazil": "🇧🇷", "Morocco": "🇲🇦", "Haiti": "🇭🇹", "Scotland": "🏴",
    "United States": "🇺🇸", "Paraguay": "🇵🇾", "Australia": "🇦🇺", "Turkey": "🇹🇷",
    "Germany": "🇩🇪", "Curacao": "🇨🇼", "Ivory Coast": "🇨🇮", "Ecuador": "🇪🇨",
    "Netherlands": "🇳🇱", "Japan": "🇯🇵", "Sweden": "🇸🇪", "Tunisia": "🇹🇳",
    "Belgium": "🇧🇪", "Egypt": "🇪🇬", "Iran": "🇮🇷", "New Zealand": "🇳🇿",
    "Spain": "🇪🇸", "Cape Verde": "🇨🇻", "Saudi Arabia": "🇸🇦", "Uruguay": "🇺🇾",
    "France": "🇫🇷", "Senegal": "🇸🇳", "Iraq": "🇮🇶", "Norway": "🇳🇴",
    "Argentina": "🇦🇷", "Algeria": "🇩🇿", "Austria": "🇦🇹", "Jordan": "🇯🇴",
    "Portugal": "🇵🇹", "DR Congo": "🇨🇩", "Uzbekistan": "🇺🇿", "Colombia": "🇨🇴",
    "England": "🏴", "Croatia": "🇭🇷", "Ghana": "🇬🇭", "Panama": "🇵🇦",
}


# Perfil climático aproximado usado para el ajuste manual de clima.
# No pretende ser perfecto; sirve para que el modelo entienda que temperatura/humedad/altitud
# pueden afectar de forma distinta a cada selección.
TEAM_CLIMATE_PROFILE = {
    "Mexico": {"temp": 22, "humidity": 55, "altitude_tolerance": 1800},
    "South Africa": {"temp": 22, "humidity": 50, "altitude_tolerance": 1500},
    "South Korea": {"temp": 18, "humidity": 60, "altitude_tolerance": 800},
    "Czechia": {"temp": 15, "humidity": 55, "altitude_tolerance": 600},
    "Canada": {"temp": 10, "humidity": 55, "altitude_tolerance": 600},
    "Bosnia and Herzegovina": {"temp": 16, "humidity": 55, "altitude_tolerance": 700},
    "Qatar": {"temp": 30, "humidity": 55, "altitude_tolerance": 300},
    "Switzerland": {"temp": 12, "humidity": 55, "altitude_tolerance": 1200},
    "Brazil": {"temp": 25, "humidity": 70, "altitude_tolerance": 800},
    "Morocco": {"temp": 24, "humidity": 45, "altitude_tolerance": 900},
    "Haiti": {"temp": 27, "humidity": 75, "altitude_tolerance": 400},
    "Scotland": {"temp": 11, "humidity": 70, "altitude_tolerance": 500},
    "United States": {"temp": 19, "humidity": 55, "altitude_tolerance": 900},
    "Paraguay": {"temp": 25, "humidity": 65, "altitude_tolerance": 500},
    "Australia": {"temp": 23, "humidity": 55, "altitude_tolerance": 700},
    "Turkey": {"temp": 20, "humidity": 55, "altitude_tolerance": 800},
    "Germany": {"temp": 14, "humidity": 60, "altitude_tolerance": 500},
    "Curacao": {"temp": 28, "humidity": 75, "altitude_tolerance": 300},
    "Ivory Coast": {"temp": 27, "humidity": 78, "altitude_tolerance": 400},
    "Ecuador": {"temp": 20, "humidity": 65, "altitude_tolerance": 2500},
    "Netherlands": {"temp": 13, "humidity": 70, "altitude_tolerance": 400},
    "Japan": {"temp": 18, "humidity": 65, "altitude_tolerance": 800},
    "Sweden": {"temp": 10, "humidity": 60, "altitude_tolerance": 500},
    "Tunisia": {"temp": 25, "humidity": 45, "altitude_tolerance": 600},
    "Belgium": {"temp": 13, "humidity": 70, "altitude_tolerance": 500},
    "Egypt": {"temp": 28, "humidity": 45, "altitude_tolerance": 500},
    "Iran": {"temp": 23, "humidity": 40, "altitude_tolerance": 1200},
    "New Zealand": {"temp": 14, "humidity": 65, "altitude_tolerance": 600},
    "Spain": {"temp": 21, "humidity": 55, "altitude_tolerance": 700},
    "Cape Verde": {"temp": 26, "humidity": 70, "altitude_tolerance": 400},
    "Saudi Arabia": {"temp": 31, "humidity": 35, "altitude_tolerance": 500},
    "Uruguay": {"temp": 20, "humidity": 65, "altitude_tolerance": 500},
    "France": {"temp": 15, "humidity": 60, "altitude_tolerance": 600},
    "Senegal": {"temp": 28, "humidity": 70, "altitude_tolerance": 400},
    "Iraq": {"temp": 31, "humidity": 35, "altitude_tolerance": 500},
    "Norway": {"temp": 8, "humidity": 65, "altitude_tolerance": 500},
    "Argentina": {"temp": 20, "humidity": 55, "altitude_tolerance": 900},
    "Algeria": {"temp": 26, "humidity": 40, "altitude_tolerance": 700},
    "Austria": {"temp": 13, "humidity": 60, "altitude_tolerance": 900},
    "Jordan": {"temp": 27, "humidity": 40, "altitude_tolerance": 700},
    "Portugal": {"temp": 20, "humidity": 60, "altitude_tolerance": 500},
    "DR Congo": {"temp": 26, "humidity": 80, "altitude_tolerance": 500},
    "Uzbekistan": {"temp": 24, "humidity": 35, "altitude_tolerance": 700},
    "Colombia": {"temp": 22, "humidity": 65, "altitude_tolerance": 1800},
    "England": {"temp": 12, "humidity": 70, "altitude_tolerance": 400},
    "Croatia": {"temp": 18, "humidity": 60, "altitude_tolerance": 600},
    "Ghana": {"temp": 27, "humidity": 75, "altitude_tolerance": 400},
    "Panama": {"temp": 28, "humidity": 80, "altitude_tolerance": 400},
}



OUTCOME_LABELS = {
    0: "Gana local/equipo A",
    1: "Empate",
    2: "Gana visitante/equipo B"
}

OUTCOME_SHORT = {
    0: "A",
    1: "X",
    2: "B"
}


# ============================================================
# 2. UTILIDADES DE DATOS
# ============================================================

def safe_name(name):
    """Normaliza nombres básicos para evitar errores por espacios."""
    if pd.isna(name):
        return ""
    return str(name).strip()


def download_historical_results(force=False):
    """
    Descarga resultados históricos de selecciones.
    Si no hay internet o falla, intenta cargar datos/results.csv local.
    """
    local_file = DATA_DIR / "results.csv"

    if local_file.exists() and not force:
        df = pd.read_csv(local_file)
        return clean_results_df(df)

    try:
        df = pd.read_csv(HISTORICAL_RESULTS_URL)
        df.to_csv(local_file, index=False)
        return clean_results_df(df)
    except Exception as e:
        if local_file.exists():
            df = pd.read_csv(local_file)
            return clean_results_df(df)
        raise RuntimeError(
            "No pude descargar results.csv ni encontré datos/results.csv local. "
            "Revisa internet o descarga el CSV manualmente."
        ) from e


def clean_results_df(df):
    """Limpieza mínima del dataset histórico."""
    required = [
        "date", "home_team", "away_team", "home_score", "away_score",
        "tournament", "city", "country", "neutral"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en results.csv: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_team"] = df["home_team"].map(safe_name)
    df["away_team"] = df["away_team"].map(safe_name)
    df["tournament"] = df["tournament"].map(safe_name)
    df["city"] = df["city"].map(safe_name)
    df["country"] = df["country"].map(safe_name)

    # Neutral puede venir como bool o texto
    df["neutral"] = df["neutral"].astype(str).str.upper().isin(["TRUE", "1", "YES", "SI", "SÍ"])

    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["date", "home_score", "away_score", "home_team", "away_team"])

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.sort_values("date").reset_index(drop=True)
    return df



def internal_wc_calendar_seed():
    """
    Calendario interno gratuito del Mundial 2026.
    Sirve como respaldo cuando API-Football free bloquea la temporada 2026.
    Los fixture_id aquí se usan como identificadores internos/de referencia.
    La sede exacta puede completarse en datos/calendario_mundial_2026.csv cuando se confirme.
    """
    rows = [
        # Grupo A
        {"fixture_id": 66456904, "date": "2026-06-11", "home_team": "Mexico", "away_team": "South Africa", "group": "A", "status": "Complete", "home_score": 2, "away_score": 0, "sede_pais": "Mexico", "venue_key": ""},
        {"fixture_id": 66456906, "date": "2026-06-11", "home_team": "South Korea", "away_team": "Czechia", "group": "A", "status": "Complete", "home_score": 2, "away_score": 1, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66456910, "date": "2026-06-18", "home_team": "Czechia", "away_team": "South Africa", "group": "A", "status": "Complete", "home_score": 1, "away_score": 1, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66456908, "date": "2026-06-18", "home_team": "Mexico", "away_team": "South Korea", "group": "A", "status": "Complete", "home_score": 1, "away_score": 0, "sede_pais": "Mexico", "venue_key": ""},
        {"fixture_id": 66456912, "date": "2026-06-24", "home_team": "Czechia", "away_team": "Mexico", "group": "A", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "Mexico", "venue_key": ""},
        {"fixture_id": 66456914, "date": "2026-06-24", "home_team": "South Africa", "away_team": "South Korea", "group": "A", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},

        # Grupo B
        {"fixture_id": 66456916, "date": "2026-06-12", "home_team": "Canada", "away_team": "Bosnia and Herzegovina", "group": "B", "status": "Complete", "home_score": 1, "away_score": 1, "sede_pais": "Canada", "venue_key": ""},
        {"fixture_id": 66456918, "date": "2026-06-13", "home_team": "Qatar", "away_team": "Switzerland", "group": "B", "status": "Complete", "home_score": 1, "away_score": 1, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66456922, "date": "2026-06-18", "home_team": "Switzerland", "away_team": "Bosnia and Herzegovina", "group": "B", "status": "Complete", "home_score": 4, "away_score": 1, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66456920, "date": "2026-06-18", "home_team": "Canada", "away_team": "Qatar", "group": "B", "status": "Complete", "home_score": 6, "away_score": 0, "sede_pais": "Canada", "venue_key": ""},
        {"fixture_id": 66456924, "date": "2026-06-24", "home_team": "Switzerland", "away_team": "Canada", "group": "B", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "Canada", "venue_key": ""},
        {"fixture_id": 66456926, "date": "2026-06-24", "home_team": "Bosnia and Herzegovina", "away_team": "Qatar", "group": "B", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},

        # Grupo C
        {"fixture_id": 66456928, "date": "2026-06-13", "home_team": "Brazil", "away_team": "Morocco", "group": "C", "status": "Complete", "home_score": 1, "away_score": 1, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66456930, "date": "2026-06-13", "home_team": "Haiti", "away_team": "Scotland", "group": "C", "status": "Complete", "home_score": 0, "away_score": 1, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66456934, "date": "2026-06-19", "home_team": "Scotland", "away_team": "Morocco", "group": "C", "status": "Complete", "home_score": 0, "away_score": 1, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66456932, "date": "2026-06-19", "home_team": "Brazil", "away_team": "Haiti", "group": "C", "status": "Complete", "home_score": 3, "away_score": 0, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66456936, "date": "2026-06-24", "home_team": "Scotland", "away_team": "Brazil", "group": "C", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66456938, "date": "2026-06-24", "home_team": "Morocco", "away_team": "Haiti", "group": "C", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},

        # Grupo D
        {"fixture_id": 66456940, "date": "2026-06-12", "home_team": "United States", "away_team": "Paraguay", "group": "D", "status": "Complete", "home_score": 4, "away_score": 1, "sede_pais": "United States", "venue_key": ""},
        {"fixture_id": 66456942, "date": "2026-06-14", "home_team": "Australia", "away_team": "Turkey", "group": "D", "status": "Complete", "home_score": 2, "away_score": 0, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66456944, "date": "2026-06-19", "home_team": "United States", "away_team": "Australia", "group": "D", "status": "Complete", "home_score": 2, "away_score": 0, "sede_pais": "United States", "venue_key": ""},
        {"fixture_id": 66456946, "date": "2026-06-19", "home_team": "Turkey", "away_team": "Paraguay", "group": "D", "status": "Complete", "home_score": 0, "away_score": 1, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66456948, "date": "2026-06-25", "home_team": "Turkey", "away_team": "United States", "group": "D", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "United States", "venue_key": ""},
        {"fixture_id": 66456950, "date": "2026-06-25", "home_team": "Paraguay", "away_team": "Australia", "group": "D", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},

        # Grupo E/F
        {"fixture_id": 66457070, "date": "2026-06-14", "home_team": "Germany", "away_team": "Curacao", "group": "E", "status": "Complete", "home_score": 7, "away_score": 1, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66457072, "date": "2026-06-14", "home_team": "Ivory Coast", "away_team": "Ecuador", "group": "E", "status": "Complete", "home_score": 1, "away_score": 0, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66457074, "date": "2026-06-20", "home_team": "Germany", "away_team": "Ivory Coast", "group": "E", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66457076, "date": "2026-06-20", "home_team": "Ecuador", "away_team": "Curacao", "group": "E", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66457078, "date": "2026-06-25", "home_team": "Ecuador", "away_team": "Germany", "group": "E", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66457080, "date": "2026-06-25", "home_team": "Curacao", "away_team": "Ivory Coast", "group": "E", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},

        {"fixture_id": 66456968, "date": "2026-06-14", "home_team": "Netherlands", "away_team": "Japan", "group": "F", "status": "Complete", "home_score": 2, "away_score": 2, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66456970, "date": "2026-06-14", "home_team": "Sweden", "away_team": "Tunisia", "group": "F", "status": "Complete", "home_score": 5, "away_score": 1, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66456972, "date": "2026-06-20", "home_team": "Netherlands", "away_team": "Sweden", "group": "F", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66456974, "date": "2026-06-21", "home_team": "Tunisia", "away_team": "Japan", "group": "F", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66456976, "date": "2026-06-25", "home_team": "Tunisia", "away_team": "Netherlands", "group": "F", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66456978, "date": "2026-06-25", "home_team": "Japan", "away_team": "Sweden", "group": "F", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},

        # G/H/I/J/K/L complete and upcoming from project snapshot
        {"fixture_id": 66456982, "date": "2026-06-15", "home_team": "Belgium", "away_team": "Egypt", "group": "G", "status": "Complete", "home_score": 1, "away_score": 1, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66456984, "date": "2026-06-15", "home_team": "Iran", "away_team": "New Zealand", "group": "G", "status": "Complete", "home_score": 2, "away_score": 2, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66456986, "date": "2026-06-21", "home_team": "Belgium", "away_team": "Iran", "group": "G", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66456988, "date": "2026-06-21", "home_team": "New Zealand", "away_team": "Egypt", "group": "G", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66456990, "date": "2026-06-26", "home_team": "New Zealand", "away_team": "Belgium", "group": "G", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66456992, "date": "2026-06-26", "home_team": "Egypt", "away_team": "Iran", "group": "G", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},

        {"fixture_id": 66456994, "date": "2026-06-15", "home_team": "Spain", "away_team": "Cape Verde", "group": "H", "status": "Complete", "home_score": 0, "away_score": 0, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66456996, "date": "2026-06-15", "home_team": "Saudi Arabia", "away_team": "Uruguay", "group": "H", "status": "Complete", "home_score": 1, "away_score": 1, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66456998, "date": "2026-06-21", "home_team": "Spain", "away_team": "Saudi Arabia", "group": "H", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66457000, "date": "2026-06-21", "home_team": "Uruguay", "away_team": "Cape Verde", "group": "H", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66457002, "date": "2026-06-26", "home_team": "Uruguay", "away_team": "Spain", "group": "H", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66457004, "date": "2026-06-26", "home_team": "Cape Verde", "away_team": "Saudi Arabia", "group": "H", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},

        {"fixture_id": 66457006, "date": "2026-06-16", "home_team": "France", "away_team": "Senegal", "group": "I", "status": "Complete", "home_score": 3, "away_score": 1, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66457008, "date": "2026-06-16", "home_team": "Iraq", "away_team": "Norway", "group": "I", "status": "Complete", "home_score": 1, "away_score": 4, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66457010, "date": "2026-06-22", "home_team": "France", "away_team": "Iraq", "group": "I", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66457012, "date": "2026-06-22", "home_team": "Norway", "away_team": "Senegal", "group": "I", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66457014, "date": "2026-06-26", "home_team": "Norway", "away_team": "France", "group": "I", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66457016, "date": "2026-06-26", "home_team": "Senegal", "away_team": "Iraq", "group": "I", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},

        {"fixture_id": 66457018, "date": "2026-06-16", "home_team": "Argentina", "away_team": "Algeria", "group": "J", "status": "Complete", "home_score": 3, "away_score": 0, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66457020, "date": "2026-06-17", "home_team": "Austria", "away_team": "Jordan", "group": "J", "status": "Complete", "home_score": 3, "away_score": 1, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66457022, "date": "2026-06-22", "home_team": "Argentina", "away_team": "Austria", "group": "J", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66457024, "date": "2026-06-22", "home_team": "Jordan", "away_team": "Algeria", "group": "J", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66457026, "date": "2026-06-27", "home_team": "Jordan", "away_team": "Argentina", "group": "J", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66457028, "date": "2026-06-27", "home_team": "Algeria", "away_team": "Austria", "group": "J", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},

        {"fixture_id": 66457030, "date": "2026-06-17", "home_team": "Portugal", "away_team": "DR Congo", "group": "K", "status": "Complete", "home_score": 1, "away_score": 1, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66457032, "date": "2026-06-17", "home_team": "Uzbekistan", "away_team": "Colombia", "group": "K", "status": "Complete", "home_score": 1, "away_score": 3, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66457034, "date": "2026-06-23", "home_team": "Portugal", "away_team": "Uzbekistan", "group": "K", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66457036, "date": "2026-06-23", "home_team": "Colombia", "away_team": "DR Congo", "group": "K", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66457038, "date": "2026-06-27", "home_team": "Colombia", "away_team": "Portugal", "group": "K", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66457040, "date": "2026-06-27", "home_team": "DR Congo", "away_team": "Uzbekistan", "group": "K", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},

        {"fixture_id": 66457042, "date": "2026-06-17", "home_team": "England", "away_team": "Croatia", "group": "L", "status": "Complete", "home_score": 4, "away_score": 2, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66457044, "date": "2026-06-17", "home_team": "Ghana", "away_team": "Panama", "group": "L", "status": "Complete", "home_score": 1, "away_score": 0, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66457046, "date": "2026-06-23", "home_team": "England", "away_team": "Ghana", "group": "L", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66457048, "date": "2026-06-23", "home_team": "Panama", "away_team": "Croatia", "group": "L", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66457050, "date": "2026-06-27", "home_team": "Panama", "away_team": "England", "group": "L", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
        {"fixture_id": 66457052, "date": "2026-06-27", "home_team": "Croatia", "away_team": "Ghana", "group": "L", "status": "Scheduled", "home_score": None, "away_score": None, "sede_pais": "Sin ventaja de anfitrión", "venue_key": ""},
    ]
    return rows


def sync_internal_wc_calendar(force_seed_update=True):
    """
    Sincroniza datos/calendario_mundial_2026.csv con el calendario semilla del código.
    Esto corrige el problema de versiones viejas: si el CSV ya existía con partidos
    como Scheduled, se actualiza a Complete cuando la semilla trae marcador confirmado.
    """
    seed_df = pd.DataFrame(internal_wc_calendar_seed())

    if not INTERNAL_WC_CALENDAR_FILE.exists():
        seed_df.to_csv(INTERNAL_WC_CALENDAR_FILE, index=False)
        return

    current = pd.read_csv(INTERNAL_WC_CALENDAR_FILE)

    # Asegurar columnas
    for col in seed_df.columns:
        if col not in current.columns:
            current[col] = None

    if "fixture_id" not in current.columns:
        seed_df.to_csv(INTERNAL_WC_CALENDAR_FILE, index=False)
        return

    current["fixture_id"] = pd.to_numeric(current["fixture_id"], errors="coerce")
    seed_df["fixture_id"] = pd.to_numeric(seed_df["fixture_id"], errors="coerce")

    current = current.dropna(subset=["fixture_id"]).copy()
    seed_df = seed_df.dropna(subset=["fixture_id"]).copy()

    current = current.set_index("fixture_id", drop=False)
    seed_df = seed_df.set_index("fixture_id", drop=False)

    for fid, seed_row in seed_df.iterrows():
        if fid not in current.index:
            current.loc[fid, seed_df.columns] = seed_row
            continue

        # Completar columnas base si están vacías.
        for col in ["date", "home_team", "away_team", "group", "sede_pais"]:
            if pd.isna(current.loc[fid, col]) or str(current.loc[fid, col]).strip() == "":
                current.loc[fid, col] = seed_row[col]

        # Mantener venue_key del usuario si lo completó. Si está vacío, usar el de la semilla.
        if "venue_key" in seed_df.columns:
            cur_vk = "" if pd.isna(current.loc[fid, "venue_key"]) else str(current.loc[fid, "venue_key"]).strip()
            seed_vk = "" if pd.isna(seed_row.get("venue_key", "")) else str(seed_row.get("venue_key", "")).strip()
            if not cur_vk and seed_vk:
                current.loc[fid, "venue_key"] = seed_vk

        # Actualizar resultados confirmados incluidos en el código.
        seed_status = str(seed_row.get("status", "")).lower()
        cur_status = str(current.loc[fid, "status"]).lower() if "status" in current.columns else ""
        seed_has_score = pd.notna(seed_row.get("home_score")) and pd.notna(seed_row.get("away_score"))

        if force_seed_update and seed_status in ["complete", "final", "finished"] and seed_has_score:
            if cur_status not in ["complete", "final", "finished"] or pd.isna(current.loc[fid, "home_score"]) or pd.isna(current.loc[fid, "away_score"]):
                current.loc[fid, "status"] = seed_row["status"]
                current.loc[fid, "home_score"] = seed_row["home_score"]
                current.loc[fid, "away_score"] = seed_row["away_score"]

    # Evita error de pandas:
    # "'fixture_id' is both an index level and a column label, which is ambiguous"
    current = current.reset_index(drop=True)
    current = current.sort_values(["date", "fixture_id"]).reset_index(drop=True)
    current.to_csv(INTERNAL_WC_CALENDAR_FILE, index=False)


def create_internal_wc_calendar_template():
    """
    Crea/actualiza el calendario interno gratuito.
    El usuario puede completar venue_key con una sede exacta si la conoce.
    """
    sync_internal_wc_calendar(force_seed_update=True)


def load_internal_wc_calendar():
    """Carga calendario interno gratuito, lo sincroniza y normaliza nombres."""
    sync_internal_wc_calendar(force_seed_update=True)
    df = pd.read_csv(INTERNAL_WC_CALENDAR_FILE)
    for col in ["home_score", "away_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["home_team"] = df["home_team"].map(normalize_team_name)
    df["away_team"] = df["away_team"].map(normalize_team_name)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def update_internal_calendar_with_results(results_df, source_label="Fuente pública"):
    """
    Actualiza calendario interno con resultados confirmados que vengan de fuentes gratuitas
    o del CSV manual. Solo actualiza partidos encontrados por equipos y fecha aproximada.
    """
    if results_df is None or results_df.empty:
        return 0

    cal = load_internal_wc_calendar()
    if cal.empty:
        return 0

    updated = 0
    cal["date_only"] = pd.to_datetime(cal["date"], errors="coerce").dt.date

    for _, row in results_df.iterrows():
        try:
            h = normalize_team_name(row["home_team"])
            a = normalize_team_name(row["away_team"])
            hs = int(row["home_score"])
            aw = int(row["away_score"])
            d = pd.to_datetime(row["date"], errors="coerce")
            if pd.isna(d):
                continue

            mask = (
                (cal["date_only"] == d.date()) &
                (
                    ((cal["home_team"] == h) & (cal["away_team"] == a)) |
                    ((cal["home_team"] == a) & (cal["away_team"] == h))
                )
            )

            if not mask.any():
                # fallback por equipos sin fecha exacta
                mask = (
                    ((cal["home_team"] == h) & (cal["away_team"] == a)) |
                    ((cal["home_team"] == a) & (cal["away_team"] == h))
                )

            if not mask.any():
                continue

            idx = cal[mask].index[0]
            if cal.loc[idx, "home_team"] == h:
                cal.loc[idx, "home_score"] = hs
                cal.loc[idx, "away_score"] = aw
            else:
                cal.loc[idx, "home_score"] = aw
                cal.loc[idx, "away_score"] = hs

            cal.loc[idx, "status"] = "Complete"
            updated += 1
        except Exception:
            continue

    if updated > 0:
        cal = cal.drop(columns=["date_only"], errors="ignore")
        cal.to_csv(INTERNAL_WC_CALENDAR_FILE, index=False)

    return updated


def load_internal_wc_results():
    """Convierte partidos completos del calendario interno en resultados para entrenar/actualizar."""
    cal = load_internal_wc_calendar()
    if cal.empty:
        return pd.DataFrame()

    mask = (
        cal["status"].astype(str).str.lower().isin(["complete", "final", "finished"]) &
        cal["home_score"].notna() &
        cal["away_score"].notna()
    )
    df = cal.loc[mask].copy()
    if df.empty:
        return pd.DataFrame()

    df["tournament"] = "FIFA World Cup"
    df["city"] = ""
    df["country"] = df.get("sede_pais", "United States")
    df["neutral"] = True
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    return clean_results_df(df[["date", "home_team", "away_team", "home_score", "away_score", "tournament", "city", "country", "neutral"]])


def find_internal_fixture(team_a, team_b, match_date=None):
    """
    Busca el partido en el calendario interno.
    No depende de API-Football ni de plan pago.
    """
    try:
        cal = load_internal_wc_calendar()
        if cal.empty:
            return None

        a = normalize_team_name(team_a)
        b = normalize_team_name(team_b)

        m = cal[
            (
                ((cal["home_team"] == a) & (cal["away_team"] == b)) |
                ((cal["home_team"] == b) & (cal["away_team"] == a))
            )
        ].copy()

        if m.empty:
            return None

        if match_date is not None:
            md = pd.to_datetime(match_date, errors="coerce")
            if pd.notna(md):
                same_day = m[m["date"].dt.date == md.date()]
                if not same_day.empty:
                    m = same_day

        row = m.sort_values("date").iloc[0].to_dict()
        return {
            "source": "Calendario interno gratuito",
            "fixture_id": int(row.get("fixture_id")) if pd.notna(row.get("fixture_id")) else None,
            "date": str(row.get("date"))[:10],
            "home_team": row.get("home_team"),
            "away_team": row.get("away_team"),
            "group": row.get("group", ""),
            "status": row.get("status", ""),
            "home_score": None if pd.isna(row.get("home_score")) else int(row.get("home_score")),
            "away_score": None if pd.isna(row.get("away_score")) else int(row.get("away_score")),
            "sede_pais": row.get("sede_pais", "Sin ventaja de anfitrión"),
            "venue_key": "" if pd.isna(row.get("venue_key", "")) else str(row.get("venue_key", "")).strip(),
        }
    except Exception:
        return None


def fetch_espn_worldcup_scoreboard(match_date=None, force=False):
    """
    Fuente gratuita opcional: intenta consultar scoreboard público de ESPN.
    Si falla, la app continúa con calendario interno/manual.
    """
    try:
        if match_date is None:
            date_key = datetime.today().strftime("%Y%m%d")
        else:
            date_key = pd.to_datetime(match_date).strftime("%Y%m%d")

        cache_path = CACHE_DIR / f"espn_worldcup_{date_key}.json"
        if cache_path.exists() and not force:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

        urls = [
            f"https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?dates={date_key}",
            f"https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world.cup/scoreboard?dates={date_key}",
            f"https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard?dates={date_key}",
        ]

        last_error = ""
        for url in urls:
            try:
                r = requests.get(url, timeout=15)
                if r.status_code == 200:
                    data = r.json()
                    if isinstance(data, dict) and "events" in data:
                        with open(cache_path, "w", encoding="utf-8") as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                        return data
                last_error = f"HTTP {r.status_code}"
            except Exception as e:
                last_error = str(e)[:120]

        return {"events": [], "error": last_error}
    except Exception as e:
        return {"events": [], "error": str(e)[:120]}


def fetch_free_worldcup_results_range(start_date="2026-06-11", end_date=None, force=False):
    """
    Intenta traer marcadores de fuentes públicas gratuitas para varias fechas.
    No depende de API-Football.
    """
    if end_date is None:
        end_date = datetime.today().date()

    start_dt = pd.to_datetime(start_date, errors="coerce")
    end_dt = pd.to_datetime(end_date, errors="coerce")

    if pd.isna(start_dt) or pd.isna(end_dt):
        return pd.DataFrame(), "Rango de fechas inválido."

    all_results = []
    messages = []

    d = start_dt
    while d <= end_dt:
        data = fetch_espn_worldcup_scoreboard(match_date=d.date(), force=force)
        res = espn_scoreboard_to_results(data)
        if not res.empty:
            all_results.append(res)
            messages.append(f"{d.date()}: {len(res)} resultados")
        d += pd.Timedelta(days=1)

    if all_results:
        df = pd.concat(all_results, ignore_index=True).drop_duplicates(
            subset=["date", "home_team", "away_team", "home_score", "away_score"],
            keep="last"
        )
        return df, "; ".join(messages)

    return pd.DataFrame(), "No se encontraron resultados nuevos en fuentes públicas gratuitas."


def espn_scoreboard_to_results(data):
    """
    Convierte eventos de ESPN a resultados si están finalizados.
    """
    rows = []
    if not data or not isinstance(data, dict):
        return pd.DataFrame()

    for ev in data.get("events", []):
        try:
            comp = (ev.get("competitions") or [{}])[0]
            status = comp.get("status", {}).get("type", {})
            completed = bool(status.get("completed", False))
            if not completed:
                continue

            competitors = comp.get("competitors", [])
            if len(competitors) < 2:
                continue

            home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
            away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])

            home_team = normalize_team_name(home.get("team", {}).get("displayName") or home.get("team", {}).get("name"))
            away_team = normalize_team_name(away.get("team", {}).get("displayName") or away.get("team", {}).get("name"))

            hs = int(float(home.get("score", 0)))
            aw = int(float(away.get("score", 0)))

            rows.append({
                "date": str(ev.get("date", ""))[:10],
                "home_team": home_team,
                "away_team": away_team,
                "home_score": hs,
                "away_score": aw,
                "tournament": "FIFA World Cup",
                "city": "",
                "country": "",
                "neutral": True,
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()
    return clean_results_df(pd.DataFrame(rows))




def create_manual_wc_template():
    """
    Crea un archivo manual para agregar resultados del Mundial 2026 si no tienes API.
    Puedes editarlo en Excel o Google Sheets y guardarlo como CSV.
    """
    if MANUAL_WC_FILE.exists():
        return

    template = pd.DataFrame([
        {
            "date": "2026-06-17",
            "home_team": "Uzbekistan",
            "away_team": "Colombia",
            "home_score": 1,
            "away_score": 3,
            "tournament": "FIFA World Cup",
            "city": "Example City",
            "country": "United States",
            "neutral": True
        }
    ])
    template.to_csv(MANUAL_WC_FILE, index=False)


def load_manual_wc_results():
    """Carga resultados manuales del Mundial 2026 si existen."""
    create_manual_wc_template()
    if not MANUAL_WC_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(MANUAL_WC_FILE)
    return clean_results_df(df)


# ============================================================
# 3. API-FOOTBALL: ACTUALIZACIÓN WEB OPCIONAL
# ============================================================

def api_get(endpoint, api_key, params=None, cache_name=None, force=False):
    """
    Consulta API-Football.
    Necesitas API key de API-Sports/API-Football.
    No se consultan endpoints de cuotas.
    """
    if not api_key:
        return None

    if params is None:
        params = {}

    cache_path = None
    if cache_name:
        cache_path = CACHE_DIR / cache_name
        if cache_path.exists() and not force:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

    headers = {"x-apisports-key": api_key}
    url = f"{API_FOOTBALL_BASE}/{endpoint.lstrip('/')}"
    response = requests.get(url, headers=headers, params=params, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(f"Error API {response.status_code}: {response.text[:300]}")

    data = response.json()

    if cache_path:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return data


def fetch_worldcup_fixtures(api_key, force=False):
    """Descarga calendario/resultados del Mundial desde API-Football."""
    params = {"league": WORLD_CUP_LEAGUE_ID, "season": WORLD_CUP_SEASON}
    return api_get(
        "fixtures",
        api_key=api_key,
        params=params,
        cache_name=f"wc_{WORLD_CUP_SEASON}_fixtures.json",
        force=force
    )


def fetch_worldcup_standings(api_key, force=False):
    """Descarga tabla de posiciones del Mundial desde API-Football."""
    params = {"league": WORLD_CUP_LEAGUE_ID, "season": WORLD_CUP_SEASON}
    return api_get(
        "standings",
        api_key=api_key,
        params=params,
        cache_name=f"wc_{WORLD_CUP_SEASON}_standings.json",
        force=force
    )


def fetch_worldcup_injuries(api_key, force=False):
    """Descarga lesiones/sanciones reportadas si la API las tiene disponibles."""
    params = {"league": WORLD_CUP_LEAGUE_ID, "season": WORLD_CUP_SEASON}
    return api_get(
        "injuries",
        api_key=api_key,
        params=params,
        cache_name=f"wc_{WORLD_CUP_SEASON}_injuries.json",
        force=force
    )


def fetch_fixture_players(api_key, fixture_id, force=False):
    """
    Descarga ratings/estadísticas de jugadores por partido.
    Útil para el punto: promedio de 1 a 10 por jugador y promedio de equipo.
    """
    params = {"fixture": fixture_id}
    return api_get(
        "fixtures/players",
        api_key=api_key,
        params=params,
        cache_name=f"fixture_{fixture_id}_players.json",
        force=force
    )


def api_fixtures_to_results(fixtures_json):
    """
    Convierte fixtures finalizados de API-Football a formato results.csv.
    """
    if not fixtures_json or "response" not in fixtures_json:
        return pd.DataFrame()

    rows = []
    for item in fixtures_json["response"]:
        fixture = item.get("fixture", {})
        teams = item.get("teams", {})
        goals = item.get("goals", {})
        league = item.get("league", {})
        status = fixture.get("status", {})

        short_status = status.get("short")
        if short_status not in ["FT", "AET", "PEN"]:
            continue

        home_goals = goals.get("home")
        away_goals = goals.get("away")
        if home_goals is None or away_goals is None:
            continue

        date_txt = fixture.get("date", "")[:10]
        venue = fixture.get("venue", {}) or {}

        rows.append({
            "date": date_txt,
            "home_team": normalize_team_name((teams.get("home") or {}).get("name")),
            "away_team": normalize_team_name((teams.get("away") or {}).get("name")),
            "home_score": int(home_goals),
            "away_score": int(away_goals),
            "tournament": safe_name(league.get("name", "FIFA World Cup")),
            "city": safe_name(venue.get("city", "")),
            "country": "World Cup 2026",
            "neutral": True
        })

    if not rows:
        return pd.DataFrame()

    return clean_results_df(pd.DataFrame(rows))


# ============================================================
# 4. ELO PROPIO Y FEATURES HISTÓRICAS
# ============================================================

def tournament_weight(tournament):
    """
    Peso aproximado por importancia del partido.
    Esto NO es la fórmula oficial FIFA, es un peso propio para el modelo.
    """
    t = str(tournament).lower()

    if "fifa world cup" in t or "world cup" == t:
        return 1.35
    if "world cup qualification" in t or "qualification" in t:
        return 1.15
    if "copa américa" in t or "copa america" in t:
        return 1.20
    if "uefa euro" in t or "african cup" in t or "asian cup" in t or "gold cup" in t:
        return 1.18
    if "nations league" in t:
        return 1.08
    if "friendly" in t:
        return 0.75

    return 1.00


def expected_score_elo(rating_a, rating_b):
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo(r_a, r_b, goals_a, goals_b, tournament, neutral=True):
    """
    Actualiza Elo con:
    - K por importancia de torneo.
    - Factor por diferencia de gol moderado.
    - Pequeña ventaja local si no es neutral.
    """
    home_adv = 0 if neutral else 60
    r_a_adj = r_a + home_adv
    r_b_adj = r_b

    exp_a = expected_score_elo(r_a_adj, r_b_adj)

    if goals_a > goals_b:
        score_a = 1.0
    elif goals_a == goals_b:
        score_a = 0.5
    else:
        score_a = 0.0

    gd = abs(goals_a - goals_b)
    margin_factor = math.log(gd + 1) * (2.2 / ((abs(r_a - r_b) * 0.001) + 2.2)) if gd > 0 else 1.0

    k = 24 * tournament_weight(tournament)
    change = k * margin_factor * (score_a - exp_a)

    return r_a + change, r_b - change


class TeamMemory:
    """
    Guarda historial previo de cada selección para calcular features antes de cada partido.
    """
    def __init__(self):
        self.matches = {}
        self.elo = {}

    def ensure_team(self, team):
        if team not in self.matches:
            self.matches[team] = []
        if team not in self.elo:
            self.elo[team] = 1500.0

    def get_elo(self, team):
        self.ensure_team(team)
        return self.elo[team]

    def add_match(self, team, date, gf, ga, opponent, result_points, tournament, neutral):
        self.ensure_team(team)
        self.matches[team].append({
            "date": date,
            "gf": gf,
            "ga": ga,
            "opponent": opponent,
            "points": result_points,
            "tournament": tournament,
            "neutral": neutral
        })

    def rolling_stats(self, team, current_date=None, n=10):
        self.ensure_team(team)
        hist = self.matches[team]
        if current_date is not None:
            hist = [m for m in hist if m["date"] < current_date]

        last = hist[-n:]

        if len(last) == 0:
            return {
                f"gf_last{n}": 1.20,
                f"ga_last{n}": 1.20,
                f"points_last{n}": 1.00,
                f"win_rate_last{n}": 0.33,
                f"clean_sheet_last{n}": 0.20,
                f"scored_rate_last{n}": 0.70,
                f"matches_count_last{n}": 0,
            }

        gf = np.array([m["gf"] for m in last], dtype=float)
        ga = np.array([m["ga"] for m in last], dtype=float)
        pts = np.array([m["points"] for m in last], dtype=float)

        return {
            f"gf_last{n}": gf.mean(),
            f"ga_last{n}": ga.mean(),
            f"points_last{n}": pts.mean(),
            f"win_rate_last{n}": np.mean(pts == 3),
            f"clean_sheet_last{n}": np.mean(ga == 0),
            f"scored_rate_last{n}": np.mean(gf > 0),
            f"matches_count_last{n}": len(last),
        }

    def days_rest(self, team, current_date):
        self.ensure_team(team)
        hist = self.matches[team]
        if len(hist) == 0:
            return 7.0
        last_date = hist[-1]["date"]
        try:
            return max(0.0, float((current_date - last_date).days))
        except Exception:
            return 7.0

    def current_team_snapshot(self, team):
        """
        Stats actuales para predecir partidos futuros.
        """
        stats5 = self.rolling_stats(team, None, 5)
        stats10 = self.rolling_stats(team, None, 10)
        snap = {}
        snap.update(stats5)
        snap.update(stats10)
        snap["elo"] = self.get_elo(team)
        return snap


def h2h_stats(history_df_until_date, team_a, team_b, n=5):
    """Historial directo de últimos n enfrentamientos antes de una fecha."""
    if history_df_until_date.empty:
        return {"h2h_a_points": 1.0, "h2h_goal_diff": 0.0, "h2h_matches": 0}

    mask = (
        ((history_df_until_date["home_team"] == team_a) & (history_df_until_date["away_team"] == team_b)) |
        ((history_df_until_date["home_team"] == team_b) & (history_df_until_date["away_team"] == team_a))
    )
    h = history_df_until_date.loc[mask].tail(n)

    if h.empty:
        return {"h2h_a_points": 1.0, "h2h_goal_diff": 0.0, "h2h_matches": 0}

    points = []
    gd = []
    for _, row in h.iterrows():
        if row["home_team"] == team_a:
            gf, ga = row["home_score"], row["away_score"]
        else:
            gf, ga = row["away_score"], row["home_score"]

        gd.append(gf - ga)
        if gf > ga:
            points.append(3)
        elif gf == ga:
            points.append(1)
        else:
            points.append(0)

    return {
        "h2h_a_points": float(np.mean(points)),
        "h2h_goal_diff": float(np.mean(gd)),
        "h2h_matches": int(len(h))
    }


def build_features_from_results(results_df, min_year=2000):
    """
    Construye dataset de entrenamiento.
    Cada fila representa un partido y usa solo información disponible ANTES del partido.
    """
    df = results_df.copy()
    df = df[df["date"].dt.year >= min_year].sort_values("date").reset_index(drop=True)

    mem = TeamMemory()
    feature_rows = []

    # Para no recalcular h2h con toda la tabla cada vez, usamos acumulado simple.
    past_rows = []

    for _, row in df.iterrows():
        date = row["date"]
        home = row["home_team"]
        away = row["away_team"]
        hs = int(row["home_score"])
        aw = int(row["away_score"])
        tournament = row["tournament"]
        neutral = bool(row["neutral"])

        mem.ensure_team(home)
        mem.ensure_team(away)

        home_elo = mem.get_elo(home)
        away_elo = mem.get_elo(away)

        home5 = mem.rolling_stats(home, date, 5)
        away5 = mem.rolling_stats(away, date, 5)
        home10 = mem.rolling_stats(home, date, 10)
        away10 = mem.rolling_stats(away, date, 10)

        past_df = pd.DataFrame(past_rows) if past_rows else pd.DataFrame()
        h2h = h2h_stats(past_df, home, away, 5)

        if hs > aw:
            y = 0
            home_pts, away_pts = 3, 0
        elif hs == aw:
            y = 1
            home_pts, away_pts = 1, 1
        else:
            y = 2
            home_pts, away_pts = 0, 3

        feat = {
            "date": date,
            "home_team": home,
            "away_team": away,
            "home_score": hs,
            "away_score": aw,
            "target": y,
            "tournament": tournament,
            "neutral": int(neutral),

            # Para el Mundial 2026, la ventaja especial de anfitrión solo aplica a:
            # United States jugando en United States,
            # Mexico jugando en Mexico,
            # Canada jugando en Canada.
            # En el histórico se deja en 0 porque no conocemos automáticamente la sede anfitriona por torneo.
            "host_adv_home": 0,
            "host_adv_away": 0,
            "host_adv_diff": 0,

            # Variables avanzadas en histórico: se dejan neutrales porque no siempre están disponibles.
            "climate_score_home": 5.0,
            "climate_score_away": 5.0,
            "climate_adv_diff": 0.0,
            "mental_score_home": 5.0,
            "mental_score_away": 5.0,
            "mental_adv_diff": 0.0,
            "social_score_home": 5.0,
            "social_score_away": 5.0,
            "social_adv_diff": 0.0,
            "lineup_rating_home": 5.5,
            "lineup_rating_away": 5.5,
            "lineup_rating_diff": 0.0,

            # Contexto competitivo. En el histórico se deja neutral.
            "stage_group": 1,
            "stage_knockout": 0,
            "urgency_home": 5.0,
            "urgency_away": 5.0,
            "urgency_diff": 0.0,

            "tournament_weight": tournament_weight(tournament),

            "elo_home": home_elo,
            "elo_away": away_elo,
            "elo_diff": home_elo - away_elo,

            "rest_home": mem.days_rest(home, date),
            "rest_away": mem.days_rest(away, date),
            "rest_diff": mem.days_rest(home, date) - mem.days_rest(away, date),

            "h2h_a_points": h2h["h2h_a_points"],
            "h2h_goal_diff": h2h["h2h_goal_diff"],
            "h2h_matches": h2h["h2h_matches"],
        }

        # Diferencias de forma últimos 5 y 10
        for key, val in home5.items():
            feat["home_" + key] = val
        for key, val in away5.items():
            feat["away_" + key] = val
        for key, val in home10.items():
            feat["home_" + key] = val
        for key, val in away10.items():
            feat["away_" + key] = val

        # Variables comparativas más fuertes
        for n in [5, 10]:
            feat[f"gf_diff_last{n}"] = feat[f"home_gf_last{n}"] - feat[f"away_gf_last{n}"]
            feat[f"ga_diff_last{n}"] = feat[f"home_ga_last{n}"] - feat[f"away_ga_last{n}"]
            feat[f"points_diff_last{n}"] = feat[f"home_points_last{n}"] - feat[f"away_points_last{n}"]
            feat[f"win_rate_diff_last{n}"] = feat[f"home_win_rate_last{n}"] - feat[f"away_win_rate_last{n}"]
            feat[f"clean_sheet_diff_last{n}"] = feat[f"home_clean_sheet_last{n}"] - feat[f"away_clean_sheet_last{n}"]
            feat[f"scored_rate_diff_last{n}"] = feat[f"home_scored_rate_last{n}"] - feat[f"away_scored_rate_last{n}"]

        feature_rows.append(feat)

        # Actualizar memoria después del partido
        new_home_elo, new_away_elo = update_elo(
            home_elo, away_elo, hs, aw, tournament, neutral=neutral
        )
        mem.elo[home] = new_home_elo
        mem.elo[away] = new_away_elo

        mem.add_match(home, date, hs, aw, away, home_pts, tournament, neutral)
        mem.add_match(away, date, aw, hs, home, away_pts, tournament, neutral)

        past_rows.append(row.to_dict())

    features_df = pd.DataFrame(feature_rows)
    numeric_cols = [
        c for c in features_df.columns
        if c not in ["date", "home_team", "away_team", "home_score", "away_score", "target", "tournament"]
    ]

    return features_df, numeric_cols, mem


# ============================================================
# 5. MODELOS ML
# ============================================================

def get_models():
    """
    Modelos a comparar.
    XGBoost se intenta cargar solo si está instalado.
    """
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1500))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=350,
            max_depth=10,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        "Red neuronal simple": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=700,
                random_state=RANDOM_STATE,
                early_stopping=True
            ))
        ])
    }

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE
        )
    except Exception:
        pass

    return models


def multiclass_brier_score(y_true, proba, labels=[0, 1, 2]):
    """Brier score multiclase: menor es mejor."""
    try:
        y_true = np.array(y_true)
        y_onehot = np.zeros((len(y_true), len(labels)))
        for i, lab in enumerate(labels):
            y_onehot[:, i] = (y_true == lab).astype(float)
        return float(np.mean(np.sum((proba - y_onehot) ** 2, axis=1)))
    except Exception:
        return np.nan


def get_goal_regressors():
    """Modelos para aprender goles esperados de Equipo A y Equipo B."""
    regressors = {
        "Ridge goles": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MultiOutputRegressor(Ridge(alpha=1.0)))
        ]),
        "Random Forest goles": MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=250,
                max_depth=10,
                min_samples_leaf=3,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        ),
        "Gradient Boosting goles": MultiOutputRegressor(
            GradientBoostingRegressor(random_state=RANDOM_STATE)
        ),
        "HistGradientBoosting goles": MultiOutputRegressor(
            HistGradientBoostingRegressor(random_state=RANDOM_STATE)
        ),
    }

    try:
        from xgboost import XGBRegressor
        regressors["XGBoost goles"] = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=250,
                max_depth=4,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=RANDOM_STATE
            )
        )
    except Exception:
        pass

    return regressors


def train_goal_regressors(train_df, test_df, feature_cols):
    """
    Entrena modelos de regresión para goles home/away.
    Retorna el mejor por MAE promedio.
    """
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df[["home_score", "away_score"]]
    y_test = test_df[["home_score", "away_score"]]

    fitted = {}
    rows = []

    for name, model in get_goal_regressors().items():
        try:
            model.fit(X_train, y_train)
            pred = np.asarray(model.predict(X_test), dtype=float)
            pred = np.clip(pred, 0, 6)

            mae_home = float(np.mean(np.abs(pred[:, 0] - y_test["home_score"].values)))
            mae_away = float(np.mean(np.abs(pred[:, 1] - y_test["away_score"].values)))
            mae_mean = float((mae_home + mae_away) / 2)

            fitted[name] = model
            rows.append({
                "modelo_goles": name,
                "mae_home": mae_home,
                "mae_away": mae_away,
                "mae_promedio": mae_mean,
            })
        except Exception as e:
            rows.append({
                "modelo_goles": name,
                "mae_home": np.nan,
                "mae_away": np.nan,
                "mae_promedio": np.nan,
                "error": str(e)[:150],
            })

    metrics = pd.DataFrame(rows)
    valid = metrics.dropna(subset=["mae_promedio"])

    if valid.empty:
        return None, None, metrics

    best_name = valid.sort_values("mae_promedio").iloc[0]["modelo_goles"]
    return best_name, fitted[best_name], metrics


def make_calibrated_logistic():
    """Crea Logistic Regression calibrada, compatible con versiones distintas de sklearn."""
    base = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1500))
    ])
    try:
        return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=3)


def train_and_compare(features_df, feature_cols):
    """
    Entrena modelos con split temporal aproximado.
    Ahora incluye:
    - clasificadores W/D/L,
    - calibración de probabilidades,
    - Brier score,
    - regresores de goles.
    """
    df = features_df.dropna(subset=feature_cols + ["target"]).copy()

    # Split temporal: primeros 80% entrenan, últimos 20% prueban.
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * 0.80)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    models = get_models()

    # Agrega modelo calibrado. Si falla por versión, no rompe la app.
    try:
        models["Logistic Regression calibrada"] = make_calibrated_logistic()
    except Exception:
        pass

    fitted = {}
    metrics = []

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)
                ll = log_loss(y_test, proba, labels=[0, 1, 2])
                brier = multiclass_brier_score(y_test, proba, labels=[0, 1, 2])
            else:
                ll = np.nan
                brier = np.nan

            acc = accuracy_score(y_test, pred)
            f1 = f1_score(y_test, pred, average="weighted")

            fitted[name] = model
            metrics.append({
                "modelo": name,
                "accuracy": acc,
                "f1_weighted": f1,
                "log_loss": ll,
                "brier_score": brier,
            })
        except Exception as e:
            metrics.append({
                "modelo": name,
                "accuracy": np.nan,
                "f1_weighted": np.nan,
                "log_loss": np.nan,
                "brier_score": np.nan,
                "error": str(e)[:150]
            })

    metrics_df = pd.DataFrame(metrics)

    # Selección: menor log_loss si existe; si no, menor brier; si no, mayor f1.
    valid = metrics_df.dropna(subset=["log_loss"])
    if not valid.empty:
        best_name = valid.sort_values("log_loss").iloc[0]["modelo"]
    else:
        valid_b = metrics_df.dropna(subset=["brier_score"])
        if not valid_b.empty:
            best_name = valid_b.sort_values("brier_score").iloc[0]["modelo"]
        else:
            best_name = metrics_df.sort_values("f1_weighted", ascending=False).iloc[0]["modelo"]

    best_model = fitted[best_name]

    goal_best_name, goal_best_model, goal_metrics = train_goal_regressors(train_df, test_df, feature_cols)

    result = {
        "features_df": df,
        "feature_cols": feature_cols,
        "train_df": train_df,
        "test_df": test_df,
        "X_test": X_test,
        "y_test": y_test,
        "models": fitted,
        "metrics": metrics_df,
        "best_name": best_name,
        "best_model": best_model,
        "goal_best_name": goal_best_name,
        "goal_best_model": goal_best_model,
        "goal_metrics": goal_metrics,
    }

    return result


# ============================================================
# 6. PREDICCIÓN DE PARTIDO
# ============================================================
# 6. PREDICCIÓN DE PARTIDO
# ============================================================

def compute_worldcup_host_advantage(team_a, team_b, sede_pais="Sin ventaja de anfitrión"):
    """
    Calcula la ventaja de anfitrión del Mundial 2026.

    Regla estricta:
    - United States recibe ventaja SOLO si la sede_pais es United States.
    - Mexico recibe ventaja SOLO si la sede_pais es Mexico.
    - Canada recibe ventaja SOLO si la sede_pais es Canada.
    - Cualquier otro equipo NO recibe ventaja de casa, aunque juegue en United States, Mexico o Canada.

    Ejemplos:
    - United States vs Australia en United States => ventaja para United States.
    - United States vs Australia en Canada => sin ventaja para United States.
    - Mexico vs South Korea en Mexico => ventaja para Mexico.
    - Brazil vs Haiti en United States => sin ventaja para ambos.
    """
    sede_pais = str(sede_pais or "Sin ventaja de anfitrión").strip()

    host_adv_home = 0
    host_adv_away = 0

    if team_a in ["United States", "Mexico", "Canada"] and team_a == sede_pais:
        host_adv_home = 1

    if team_b in ["United States", "Mexico", "Canada"] and team_b == sede_pais:
        host_adv_away = 1

    return {
        "neutral": 1,  # En Mundial 2026 se modela como torneo neutral, con ajuste especial de anfitrión.
        "host_adv_home": int(host_adv_home),
        "host_adv_away": int(host_adv_away),
        "host_adv_diff": int(host_adv_home - host_adv_away),
        "sede_pais": sede_pais,
    }



def create_player_ratings_template():
    """
    Crea un CSV editable para ratings de jugadores y alineaciones.
    No inventa datos reales: deja ejemplos de estructura para que el usuario alimente la base.
    """
    if PLAYER_RATINGS_FILE.exists():
        return

    template = pd.DataFrame([
        {
            "team": "United States",
            "player": "Example Player",
            "position": "FWD",
            "career_rating": 6.5,
            "current_form_rating": 6.5,
            "expected_starter": True,
            "available": True,
            "lineup_type": "probable",
            "last_seen_date": "2026-06-18",
            "notes": "Reemplaza este ejemplo con jugadores reales."
        },
        {
            "team": "Australia",
            "player": "Example Player",
            "position": "DEF",
            "career_rating": 6.0,
            "current_form_rating": 6.0,
            "expected_starter": True,
            "available": True,
            "lineup_type": "probable",
            "last_seen_date": "2026-06-18",
            "notes": "Puedes cargar once titular oficial, probable o última alineación."
        }
    ])
    template.to_csv(PLAYER_RATINGS_FILE, index=False)


def load_player_ratings():
    """
    Carga jugadores_rating.csv.
    Columnas esperadas:
    team, player, position, career_rating, current_form_rating,
    expected_starter, available, lineup_type, last_seen_date, notes
    """
    create_player_ratings_template()

    if not PLAYER_RATINGS_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(PLAYER_RATINGS_FILE)
    required = [
        "team", "player", "position", "career_rating", "current_form_rating",
        "expected_starter", "available", "lineup_type", "last_seen_date"
    ]

    for col in required:
        if col not in df.columns:
            if col in ["career_rating", "current_form_rating"]:
                df[col] = 5.5
            elif col in ["expected_starter", "available"]:
                df[col] = True
            else:
                df[col] = ""

    df["team"] = df["team"].map(safe_name)
    df["player"] = df["player"].map(safe_name)
    df["position"] = df["position"].astype(str).str.upper().str.strip()
    df["career_rating"] = pd.to_numeric(df["career_rating"], errors="coerce").fillna(5.5).clip(0, 10)
    df["current_form_rating"] = pd.to_numeric(df["current_form_rating"], errors="coerce").fillna(df["career_rating"]).clip(0, 10)
    df["expected_starter"] = df["expected_starter"].astype(str).str.upper().isin(["TRUE", "1", "YES", "SI", "SÍ"])
    df["available"] = df["available"].astype(str).str.upper().isin(["TRUE", "1", "YES", "SI", "SÍ"])
    df["lineup_type"] = df["lineup_type"].astype(str).str.lower().str.strip()
    df["last_seen_date"] = pd.to_datetime(df["last_seen_date"], errors="coerce")

    return df


def calculate_team_lineup_rating(team, ratings_df=None):
    """
    Calcula rating promedio de alineación.
    Prioridad:
    1. jugadores available=True y expected_starter=True;
    2. si hay menos de 8, usa los mejores 11 disponibles;
    3. si no hay datos, devuelve 5.5.
    """
    if ratings_df is None:
        ratings_df = load_player_ratings()

    if ratings_df is None or ratings_df.empty:
        return {"rating": 5.5, "players_used": 0, "status": "Sin datos de jugadores"}

    d = ratings_df[(ratings_df["team"] == team) & (ratings_df["available"] == True)].copy()

    if d.empty:
        return {"rating": 5.5, "players_used": 0, "status": "Sin datos disponibles"}

    # Score jugador: carrera pesa más, forma reciente también importa.
    d["player_score"] = 0.65 * d["career_rating"] + 0.35 * d["current_form_rating"]

    pos_weight = {
        "GK": 1.00, "GKP": 1.00, "PORTERO": 1.00,
        "DEF": 1.00, "DF": 1.00,
        "MID": 1.05, "MF": 1.05,
        "FWD": 1.10, "FW": 1.10, "ST": 1.10,
    }
    d["pos_weight"] = d["position"].map(pos_weight).fillna(1.00)
    d["weighted_score"] = d["player_score"] * d["pos_weight"]

    starters = d[d["expected_starter"] == True].copy()

    if len(starters) >= 8:
        selected = starters.sort_values(["lineup_type", "last_seen_date"], ascending=[True, False]).head(11)
        status = "Usando alineación marcada como titular/probable"
    else:
        selected = d.sort_values("weighted_score", ascending=False).head(11)
        status = "Usando mejores 11 disponibles por rating"

    if selected.empty:
        return {"rating": 5.5, "players_used": 0, "status": "Sin datos suficientes"}

    rating = float(np.average(selected["player_score"], weights=selected["pos_weight"]))
    return {"rating": rating, "players_used": int(len(selected)), "status": status}


def climate_score_for_team(team, temperature_c=22, humidity_pct=55, altitude_m=300):
    """
    Puntuación 0-10 de adaptación climática aproximada.
    Considera temperatura, humedad y altitud.
    """
    profile = TEAM_CLIMATE_PROFILE.get(team, {"temp": 20, "humidity": 55, "altitude_tolerance": 700})

    temp_penalty = abs(float(temperature_c) - float(profile["temp"])) / 24.0
    humidity_penalty = abs(float(humidity_pct) - float(profile["humidity"])) / 60.0

    altitude_excess = max(0.0, float(altitude_m) - float(profile["altitude_tolerance"]))
    altitude_penalty = altitude_excess / 2800.0

    score = 10.0 - 4.0 * temp_penalty - 2.0 * humidity_penalty - 3.0 * altitude_penalty
    return clamp(score, 0.0, 10.0)



def fetch_open_meteo_weather(lat, lon, match_date=None):
    """
    Consulta Open-Meteo para temperatura y humedad.
    No requiere API key.
    """
    try:
        if match_date is None:
            match_date = datetime.today().date()
        if hasattr(match_date, "strftime"):
            date_txt = match_date.strftime("%Y-%m-%d")
        else:
            date_txt = str(match_date)[:10]

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": float(lat),
            "longitude": float(lon),
            "hourly": "temperature_2m,relative_humidity_2m",
            "forecast_days": 16,
            "timezone": "auto",
        }
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        hourly = data.get("hourly", {})

        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        hums = hourly.get("relative_humidity_2m", [])

        selected_temps = []
        selected_hums = []
        for t, temp, hum in zip(times, temps, hums):
            if str(t).startswith(date_txt):
                if temp is not None:
                    selected_temps.append(float(temp))
                if hum is not None:
                    selected_hums.append(float(hum))

        if not selected_temps and temps:
            selected_temps = [float(x) for x in temps if x is not None][:24]
        if not selected_hums and hums:
            selected_hums = [float(x) for x in hums if x is not None][:24]

        temp_avg = float(np.mean(selected_temps)) if selected_temps else 22.0
        hum_avg = float(np.mean(selected_hums)) if selected_hums else 55.0

        return {
            "ok": True,
            "temperature_c": round(temp_avg, 1),
            "humidity_pct": round(hum_avg, 1),
            "source": "Open-Meteo",
            "message": "Clima actualizado correctamente."
        }

    except Exception as e:
        return {
            "ok": False,
            "temperature_c": 22.0,
            "humidity_pct": 55.0,
            "source": "Open-Meteo",
            "message": f"No se pudo consultar clima: {str(e)[:120]}"
        }


def fetch_gdelt_team_news(team, days=3, max_records=12):
    """
    Consulta noticias recientes con GDELT DOC API.
    No requiere API key.
    """
    try:
        team_query = f'"{team}" ("World Cup" OR football OR soccer)'
        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {
            "query": team_query,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": int(max_records),
            "timespan": f"{int(days)}d",
            "sort": "DateDesc",
        }
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        articles = data.get("articles", [])
        return {"ok": True, "articles": articles, "message": f"{len(articles)} noticias encontradas"}
    except Exception as e:
        return {"ok": False, "articles": [], "message": f"No se pudo consultar GDELT: {str(e)[:120]}"}


def score_news_cohesion_from_articles(articles):
    """
    Convierte titulares/noticias en variables de cohesión y polémica.
    Es un análisis simple por palabras clave, no reemplaza lectura humana.
    """
    negative_keywords = [
        "injury", "injured", "doubt", "ruled out", "miss", "suspended", "ban",
        "conflict", "row", "crisis", "controversy", "criticized", "angry",
        "fight", "dispute", "illness", "fatigue", "problem", "setback",
        "lesion", "lesionado", "duda", "suspendido", "conflicto", "crisis",
        "polémica", "polemica", "pelea", "molesto", "molestia", "baja"
    ]
    positive_keywords = [
        "fit", "returns", "boost", "ready", "confident", "united", "available",
        "recovered", "training", "motivated", "strong", "optimistic",
        "regresa", "recuperado", "listo", "disponible", "confianza",
        "unido", "motivado", "optimista"
    ]

    text = " ".join([
        str(a.get("title", "")) + " " + str(a.get("seendate", "")) + " " + str(a.get("sourcecountry", ""))
        for a in articles
    ]).lower()

    neg = sum(text.count(k.lower()) for k in negative_keywords)
    pos = sum(text.count(k.lower()) for k in positive_keywords)

    controversy = clamp(neg * 1.2, 0, 10)
    cohesion = clamp(7 + 0.5 * pos - 0.8 * neg, 0, 10)

    return {
        "cohesion": float(cohesion),
        "controversy": float(controversy),
        "negative_hits": int(neg),
        "positive_hits": int(pos),
    }


def fetch_fixture_lineups(api_key, fixture_id, force=False):
    """
    Consulta alineaciones API-Football.
    Según cobertura, pueden aparecer 20-40 min antes del partido o después.
    """
    if not api_key or not fixture_id:
        return None
    return api_get(
        "fixtures/lineups",
        api_key=api_key,
        params={"fixture": int(fixture_id)},
        cache_name=f"fixture_{fixture_id}_lineups.json",
        force=force
    )


def find_fixture_for_match(fixtures_json, team_a, team_b, match_date=None):
    """
    Busca fixture del Mundial por equipos. Devuelve fixture_id y venue si lo encuentra.
    """
    if not fixtures_json or "response" not in fixtures_json:
        return None

    best = None
    for item in fixtures_json.get("response", []):
        teams = item.get("teams", {})
        fixture = item.get("fixture", {})
        home = normalize_team_name((teams.get("home") or {}).get("name"))
        away = normalize_team_name((teams.get("away") or {}).get("name"))

        same_match = {home, away} == {team_a, team_b}
        if not same_match:
            continue

        if match_date is not None:
            date_txt = str(match_date)[:10]
            fixture_date = str(fixture.get("date", ""))[:10]
            if fixture_date and fixture_date != date_txt:
                # No descartamos totalmente porque a veces hay zona horaria, pero lo dejamos como menor prioridad.
                pass

        best = item
        break

    if best is None:
        return None

    fixture = best.get("fixture", {})
    venue = fixture.get("venue", {}) or {}
    return {
        "fixture_id": fixture.get("id"),
        "date": fixture.get("date"),
        "venue_name": venue.get("name", ""),
        "venue_city": venue.get("city", ""),
    }


def lineup_rating_from_api_lineups(lineups_json, team, ratings_df=None):
    """
    Calcula rating de alineación con la alineación de API-Football si está disponible.
    Si el jugador no está en CSV, usa rating base 5.8.
    """
    if not lineups_json or "response" not in lineups_json:
        return None

    if ratings_df is None:
        ratings_df = load_player_ratings()

    target = None
    for item in lineups_json.get("response", []):
        api_team = normalize_team_name((item.get("team") or {}).get("name"))
        if api_team == normalize_team_name(team):
            target = item
            break

    if target is None:
        return None

    start_xi = target.get("startXI", []) or []
    if not start_xi:
        return None

    known = ratings_df[ratings_df["team"] == team].copy() if ratings_df is not None and not ratings_df.empty else pd.DataFrame()
    known["player_norm"] = known["player"].astype(str).str.lower().str.strip() if not known.empty else ""

    scores = []
    players = []
    for item in start_xi:
        player = item.get("player", {}) or {}
        name = safe_name(player.get("name"))
        if not name:
            continue

        base_score = 5.8
        if not known.empty:
            m = known[known["player_norm"] == name.lower().strip()]
            if not m.empty:
                base_score = float((0.65 * m.iloc[0]["career_rating"]) + (0.35 * m.iloc[0]["current_form_rating"]))

        scores.append(base_score)
        players.append(name)

    if not scores:
        return None

    return {
        "rating": float(np.mean(scores)),
        "players_used": len(scores),
        "status": "Alineación obtenida desde API-Football",
        "players": players,
    }


def count_injuries_for_team(injuries_json, team):
    if not injuries_json or "response" not in injuries_json:
        return 0

    count = 0
    for item in injuries_json.get("response", []):
        api_team = normalize_team_name((item.get("team") or {}).get("name"))
        if api_team == normalize_team_name(team):
            count += 1
    return count


def auto_fetch_advanced_context(api_key, team_a, team_b, sede_pais, venue_key, match_date=None, fixture_id_manual=None, force=False):
    """
    Construye variables avanzadas automáticamente sin depender de pagar API-Football:
    - detecta fixture desde calendario interno gratuito;
    - clima desde Open-Meteo;
    - noticias desde GDELT;
    - si API-Football sirve, usa alineaciones/lesiones; si no, continúa normal.
    """
    internal_fixture = find_internal_fixture(team_a, team_b, match_date=match_date)

    # Si el calendario interno tiene sede exacta, la usamos. Si no, se respeta la sede elegida por el usuario.
    if internal_fixture and internal_fixture.get("venue_key") in WORLD_CUP_2026_VENUES:
        venue_key = internal_fixture["venue_key"]

    venue = WORLD_CUP_2026_VENUES.get(venue_key, list(WORLD_CUP_2026_VENUES.values())[0])
    weather = fetch_open_meteo_weather(venue["lat"], venue["lon"], match_date=match_date)

    # Si calendario interno detecta sede_pais de anfitrión y el usuario dejó "sin ventaja", usamos la sede interna.
    sede_para_ventaja = sede_pais
    if internal_fixture and sede_pais == "Sin ventaja de anfitrión":
        sede_para_ventaja = internal_fixture.get("sede_pais", sede_pais) or sede_pais

    venue_info = compute_worldcup_host_advantage(team_a, team_b, sede_pais=sede_para_ventaja)
    crowd_support_a = 8 if venue_info["host_adv_home"] == 1 else 4 if venue_info["host_adv_away"] == 1 else 5
    crowd_support_b = 8 if venue_info["host_adv_away"] == 1 else 4 if venue_info["host_adv_home"] == 1 else 5
    pressure_a = 3 if venue_info["host_adv_home"] == 1 else 7 if venue_info["host_adv_away"] == 1 else 5
    pressure_b = 3 if venue_info["host_adv_away"] == 1 else 7 if venue_info["host_adv_home"] == 1 else 5

    news_a = fetch_gdelt_team_news(team_a, days=3, max_records=12)
    news_b = fetch_gdelt_team_news(team_b, days=3, max_records=12)
    news_score_a = score_news_cohesion_from_articles(news_a.get("articles", []))
    news_score_b = score_news_cohesion_from_articles(news_b.get("articles", []))

    ratings_df = load_player_ratings()
    lineup_a = calculate_team_lineup_rating(team_a, ratings_df)
    lineup_b = calculate_team_lineup_rating(team_b, ratings_df)

    fixture_info = internal_fixture
    lineups_json = None
    injuries_json = None
    api_error = None

    # API-Football es extra, no dependencia. Si plan free bloquea 2026, no rompe el flujo.
    if api_key:
        try:
            fixtures_json = fetch_worldcup_fixtures(api_key, force=force)
            api_fixture = find_fixture_for_match(fixtures_json, team_a, team_b, match_date=match_date)
            if api_fixture:
                fixture_info = api_fixture

            fixture_id = (str(fixture_id_manual).strip() if fixture_id_manual else "") or (api_fixture or {}).get("fixture_id") or (internal_fixture or {}).get("fixture_id")
            if fixture_id:
                lineups_json = fetch_fixture_lineups(api_key, fixture_id, force=force)
                api_lineup_a = lineup_rating_from_api_lineups(lineups_json, team_a, ratings_df)
                api_lineup_b = lineup_rating_from_api_lineups(lineups_json, team_b, ratings_df)
                if api_lineup_a:
                    lineup_a = api_lineup_a
                if api_lineup_b:
                    lineup_b = api_lineup_b

            injuries_json = fetch_worldcup_injuries(api_key, force=force)
            inj_a = count_injuries_for_team(injuries_json, team_a)
            inj_b = count_injuries_for_team(injuries_json, team_b)

            if inj_a > 0:
                lineup_a["rating"] = clamp(float(lineup_a["rating"]) - min(1.2, 0.18 * inj_a), 0, 10)
                news_score_a["controversy"] = clamp(news_score_a["controversy"] + 0.4 * inj_a, 0, 10)
            if inj_b > 0:
                lineup_b["rating"] = clamp(float(lineup_b["rating"]) - min(1.2, 0.18 * inj_b), 0, 10)
                news_score_b["controversy"] = clamp(news_score_b["controversy"] + 0.4 * inj_b, 0, 10)
        except Exception as e:
            api_error = str(e)[:180]

    context = {
        "temperature_c": weather["temperature_c"],
        "humidity_pct": weather["humidity_pct"],
        "altitude_m": float(venue["altitude_m"]),
        "crowd_support_a": crowd_support_a,
        "crowd_support_b": crowd_support_b,
        "pressure_a": pressure_a,
        "pressure_b": pressure_b,
        "cohesion_a": news_score_a["cohesion"],
        "cohesion_b": news_score_b["cohesion"],
        "controversy_a": news_score_a["controversy"],
        "controversy_b": news_score_b["controversy"],
        "lineup_rating_a": float(lineup_a["rating"]),
        "lineup_rating_b": float(lineup_b["rating"]),
    }

    sources = {
        "weather": weather,
        "venue": venue,
        "venue_key": venue_key,
        "internal_fixture": internal_fixture,
        "news_a": news_a.get("message", ""),
        "news_b": news_b.get("message", ""),
        "news_score_a": news_score_a,
        "news_score_b": news_score_b,
        "lineup_a": lineup_a,
        "lineup_b": lineup_b,
        "fixture_info": fixture_info,
        "api_error": api_error,
        "api_lineups_used": bool(lineups_json and lineups_json.get("response")),
        "mode": "Gratis: calendario interno + Open-Meteo + GDELT; API-Football solo como extra."
    }

    return context, sources



def build_advanced_context_features(team_a, team_b, advanced_context=None):
    """
    Convierte variables climáticas, mentales, sociales y de alineación en features.
    Todo queda en escalas moderadas para evitar que una noticia o slider destruya el modelo.
    """
    if advanced_context is None:
        advanced_context = {}

    temperature_c = float(advanced_context.get("temperature_c", 22))
    humidity_pct = float(advanced_context.get("humidity_pct", 55))
    altitude_m = float(advanced_context.get("altitude_m", 300))

    climate_a = climate_score_for_team(team_a, temperature_c, humidity_pct, altitude_m)
    climate_b = climate_score_for_team(team_b, temperature_c, humidity_pct, altitude_m)

    crowd_support_a = float(advanced_context.get("crowd_support_a", 5))
    crowd_support_b = float(advanced_context.get("crowd_support_b", 5))
    pressure_a = float(advanced_context.get("pressure_a", 5))
    pressure_b = float(advanced_context.get("pressure_b", 5))

    # Mental: apoyo ayuda, presión en contra castiga. Centro neutral = 5.
    mental_a = clamp(5 + 0.55 * (crowd_support_a - 5) - 0.45 * (pressure_a - 5), 0, 10)
    mental_b = clamp(5 + 0.55 * (crowd_support_b - 5) - 0.45 * (pressure_b - 5), 0, 10)

    cohesion_a = float(advanced_context.get("cohesion_a", 7))
    cohesion_b = float(advanced_context.get("cohesion_b", 7))
    controversy_a = float(advanced_context.get("controversy_a", 0))
    controversy_b = float(advanced_context.get("controversy_b", 0))

    # Social: cohesión ayuda; polémicas, peleas internas o crisis restan.
    social_a = clamp(0.75 * cohesion_a + 0.25 * 7 - 0.65 * controversy_a, 0, 10)
    social_b = clamp(0.75 * cohesion_b + 0.25 * 7 - 0.65 * controversy_b, 0, 10)

    lineup_rating_a = float(advanced_context.get("lineup_rating_a", 5.5))
    lineup_rating_b = float(advanced_context.get("lineup_rating_b", 5.5))

    stage = str(advanced_context.get("stage", "Fase de grupos"))
    is_knockout = 1 if stage != "Fase de grupos" else 0
    is_group = 1 - is_knockout

    urgency_a = float(advanced_context.get("urgency_a", 5))
    urgency_b = float(advanced_context.get("urgency_b", 5))

    return {
        "climate_score_home": climate_a,
        "climate_score_away": climate_b,
        "climate_adv_diff": (climate_a - climate_b) / 10.0,

        "mental_score_home": mental_a,
        "mental_score_away": mental_b,
        "mental_adv_diff": (mental_a - mental_b) / 10.0,

        "social_score_home": social_a,
        "social_score_away": social_b,
        "social_adv_diff": (social_a - social_b) / 10.0,

        "lineup_rating_home": clamp(lineup_rating_a, 0, 10),
        "lineup_rating_away": clamp(lineup_rating_b, 0, 10),
        "lineup_rating_diff": (clamp(lineup_rating_a, 0, 10) - clamp(lineup_rating_b, 0, 10)) / 10.0,

        "stage_group": is_group,
        "stage_knockout": is_knockout,
        "urgency_home": clamp(urgency_a, 0, 10),
        "urgency_away": clamp(urgency_b, 0, 10),
        "urgency_diff": (clamp(urgency_a, 0, 10) - clamp(urgency_b, 0, 10)) / 10.0,
    }



def build_future_match_features(mem, history_df, team_a, team_b, neutral=True, tournament="FIFA World Cup", sede_pais="Sin ventaja de anfitrión", advanced_context=None):
    """
    Construye una fila de features para un partido futuro.

    Para Mundial 2026:
    - No se usa una localía genérica tipo "casa del equipo A".
    - Solo se aplica ventaja de anfitrión cuando:
      United States juega en United States,
      Mexico juega en Mexico,
      Canada juega en Canada.
    """
    today = pd.Timestamp(datetime.today().date())

    mem.ensure_team(team_a)
    mem.ensure_team(team_b)

    home_elo = mem.get_elo(team_a)
    away_elo = mem.get_elo(team_b)

    home5 = mem.rolling_stats(team_a, None, 5)
    away5 = mem.rolling_stats(team_b, None, 5)
    home10 = mem.rolling_stats(team_a, None, 10)
    away10 = mem.rolling_stats(team_b, None, 10)

    h2h = h2h_stats(history_df, team_a, team_b, 5)
    venue = compute_worldcup_host_advantage(team_a, team_b, sede_pais=sede_pais)
    advanced_features = build_advanced_context_features(team_a, team_b, advanced_context)

    feat = {
        "neutral": venue["neutral"],
        "host_adv_home": venue["host_adv_home"],
        "host_adv_away": venue["host_adv_away"],
        "host_adv_diff": venue["host_adv_diff"],
        "tournament_weight": tournament_weight(tournament),

        "elo_home": home_elo,
        "elo_away": away_elo,
        "elo_diff": home_elo - away_elo,

        "rest_home": mem.days_rest(team_a, today),
        "rest_away": mem.days_rest(team_b, today),
        "rest_diff": mem.days_rest(team_a, today) - mem.days_rest(team_b, today),

        "h2h_a_points": h2h["h2h_a_points"],
        "h2h_goal_diff": h2h["h2h_goal_diff"],
        "h2h_matches": h2h["h2h_matches"],

        "climate_score_home": advanced_features["climate_score_home"],
        "climate_score_away": advanced_features["climate_score_away"],
        "climate_adv_diff": advanced_features["climate_adv_diff"],

        "mental_score_home": advanced_features["mental_score_home"],
        "mental_score_away": advanced_features["mental_score_away"],
        "mental_adv_diff": advanced_features["mental_adv_diff"],

        "social_score_home": advanced_features["social_score_home"],
        "social_score_away": advanced_features["social_score_away"],
        "social_adv_diff": advanced_features["social_adv_diff"],

        "lineup_rating_home": advanced_features["lineup_rating_home"],
        "lineup_rating_away": advanced_features["lineup_rating_away"],
        "lineup_rating_diff": advanced_features["lineup_rating_diff"],

        "stage_group": advanced_features["stage_group"],
        "stage_knockout": advanced_features["stage_knockout"],
        "urgency_home": advanced_features["urgency_home"],
        "urgency_away": advanced_features["urgency_away"],
        "urgency_diff": advanced_features["urgency_diff"],
    }

    for key, val in home5.items():
        feat["home_" + key] = val
    for key, val in away5.items():
        feat["away_" + key] = val
    for key, val in home10.items():
        feat["home_" + key] = val
    for key, val in away10.items():
        feat["away_" + key] = val

    for n in [5, 10]:
        feat[f"gf_diff_last{n}"] = feat[f"home_gf_last{n}"] - feat[f"away_gf_last{n}"]
        feat[f"ga_diff_last{n}"] = feat[f"home_ga_last{n}"] - feat[f"away_ga_last{n}"]
        feat[f"points_diff_last{n}"] = feat[f"home_points_last{n}"] - feat[f"away_points_last{n}"]
        feat[f"win_rate_diff_last{n}"] = feat[f"home_win_rate_last{n}"] - feat[f"away_win_rate_last{n}"]
        feat[f"clean_sheet_diff_last{n}"] = feat[f"home_clean_sheet_last{n}"] - feat[f"away_clean_sheet_last{n}"]
        feat[f"scored_rate_diff_last{n}"] = feat[f"home_scored_rate_last{n}"] - feat[f"away_scored_rate_last{n}"]

    return pd.DataFrame([feat])


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def dixon_coles_tau(goals_a, goals_b, lam_a, lam_b, rho=-0.06):
    """
    Ajuste Dixon-Coles simplificado para resultados bajos.
    rho negativo tiende a aumentar algunos marcadores bajos frecuentes.
    """
    if goals_a == 0 and goals_b == 0:
        tau = 1 - (lam_a * lam_b * rho)
    elif goals_a == 0 and goals_b == 1:
        tau = 1 + (lam_a * rho)
    elif goals_a == 1 and goals_b == 0:
        tau = 1 + (lam_b * rho)
    elif goals_a == 1 and goals_b == 1:
        tau = 1 - rho
    else:
        tau = 1.0
    return clamp(tau, 0.75, 1.30)


def poisson_score_matrix(future_row, lambda_override=None, use_dixon_coles=True):
    """
    Crea matriz de probabilidad de marcador usando goles esperados.
    Combina:
    - Forma ofensiva/defensiva últimos 10.
    - Diferencia Elo.
    - Descanso.
    - Sede/anfitrión.
    - Clima, presión, cohesión y alineaciones.
    - Contexto de fase/urgencia.
    - Ajuste Dixon-Coles para marcadores bajos.
    """
    r = future_row.iloc[0]

    base_goals = 1.28

    attack_a = clamp(r["home_gf_last10"] / base_goals, 0.45, 2.40)
    defense_b = clamp(r["away_ga_last10"] / base_goals, 0.45, 2.40)

    attack_b = clamp(r["away_gf_last10"] / base_goals, 0.45, 2.40)
    defense_a = clamp(r["home_ga_last10"] / base_goals, 0.45, 2.40)

    elo_factor_a = math.exp(clamp(r["elo_diff"], -500, 500) / 900)
    elo_factor_b = math.exp(clamp(-r["elo_diff"], -500, 500) / 900)

    rest_factor_a = clamp(1 + (r["rest_diff"] * 0.015), 0.90, 1.10)
    rest_factor_b = clamp(1 - (r["rest_diff"] * 0.015), 0.90, 1.10)

    host_factor_a = 1 + 0.10 * float(r.get("host_adv_home", 0))
    host_factor_b = 1 + 0.10 * float(r.get("host_adv_away", 0))

    climate_factor_a = 1 + 0.045 * ((float(r.get("climate_score_home", 5.0)) - 5.0) / 5.0)
    climate_factor_b = 1 + 0.045 * ((float(r.get("climate_score_away", 5.0)) - 5.0) / 5.0)

    mental_factor_a = 1 + 0.040 * ((float(r.get("mental_score_home", 5.0)) - 5.0) / 5.0)
    mental_factor_b = 1 + 0.040 * ((float(r.get("mental_score_away", 5.0)) - 5.0) / 5.0)

    social_factor_a = 1 + 0.035 * ((float(r.get("social_score_home", 5.0)) - 5.0) / 5.0)
    social_factor_b = 1 + 0.035 * ((float(r.get("social_score_away", 5.0)) - 5.0) / 5.0)

    lineup_factor_a = 1 + 0.070 * ((float(r.get("lineup_rating_home", 5.5)) - 5.5) / 4.5)
    lineup_factor_b = 1 + 0.070 * ((float(r.get("lineup_rating_away", 5.5)) - 5.5) / 4.5)

    # En eliminatorias suele haber más cautela; en fase de grupos la urgencia puede abrir el partido.
    knockout_factor = 0.94 if float(r.get("stage_knockout", 0)) == 1 else 1.00
    urgency_factor_a = 1 + 0.055 * ((float(r.get("urgency_home", 5.0)) - 5.0) / 5.0)
    urgency_factor_b = 1 + 0.055 * ((float(r.get("urgency_away", 5.0)) - 5.0) / 5.0)

    advanced_factor_a = clamp(
        climate_factor_a * mental_factor_a * social_factor_a * lineup_factor_a * urgency_factor_a * knockout_factor,
        0.74, 1.34
    )
    advanced_factor_b = clamp(
        climate_factor_b * mental_factor_b * social_factor_b * lineup_factor_b * urgency_factor_b * knockout_factor,
        0.74, 1.34
    )

    lam_a = base_goals * attack_a * defense_b * elo_factor_a * rest_factor_a * host_factor_a * advanced_factor_a
    lam_b = base_goals * attack_b * defense_a * elo_factor_b * rest_factor_b * host_factor_b * advanced_factor_b

    if lambda_override is not None:
        try:
            reg_a, reg_b = lambda_override
            # Ensamble moderado: Poisson heurístico + regresor de goles.
            lam_a = 0.58 * lam_a + 0.42 * float(reg_a)
            lam_b = 0.58 * lam_b + 0.42 * float(reg_b)
        except Exception:
            pass

    lam_a = clamp(lam_a, 0.15, 4.75)
    lam_b = clamp(lam_b, 0.15, 4.75)

    max_goals = 7
    rows = []
    for ga in range(max_goals + 1):
        for gb in range(max_goals + 1):
            p = poisson.pmf(ga, lam_a) * poisson.pmf(gb, lam_b)

            if use_dixon_coles:
                p *= dixon_coles_tau(ga, gb, lam_a, lam_b, rho=-0.06)

            if ga > gb:
                outcome = 0
            elif ga == gb:
                outcome = 1
            else:
                outcome = 2

            rows.append({
                "marcador": f"{ga}-{gb}",
                "goles_A": ga,
                "goles_B": gb,
                "prob_poisson": p,
                "outcome": outcome
            })

    score_df = pd.DataFrame(rows)
    score_df["prob_poisson"] = score_df["prob_poisson"] / score_df["prob_poisson"].sum()

    return score_df, lam_a, lam_b


def blend_poisson_with_ml(score_df, ml_probs, alpha=0.55):
    """
    Ajusta probabilidades de marcadores para que respeten la probabilidad W/D/L del ML.
    alpha = peso del ML.
    """
    df = score_df.copy()

    poisson_outcome_probs = df.groupby("outcome")["prob_poisson"].sum().to_dict()
    adjusted = []

    for _, row in df.iterrows():
        outcome = int(row["outcome"])
        p_pois = row["prob_poisson"]
        p_ml = ml_probs[outcome]
        p_pois_out = poisson_outcome_probs.get(outcome, 1e-9)

        # Ajuste por outcome
        p_adjusted = p_pois * ((1 - alpha) + alpha * (p_ml / max(p_pois_out, 1e-9)))
        adjusted.append(p_adjusted)

    df["prob_final"] = adjusted
    df["prob_final"] = df["prob_final"] / df["prob_final"].sum()
    return df


def predict_match(training_result, mem, history_df, team_a, team_b, neutral=True, sede_pais="Sin ventaja de anfitrión", advanced_context=None):
    """
    Devuelve:
    - probabilidades finales W/D/L
    - marcadores probables
    - lambdas de goles esperados
    - fila de features

    Nota importante:
    El modelo ML aprende ganador/empate/perdedor desde datos históricos.
    La ventaja de anfitrión del Mundial 2026 se refleja sobre todo en los goles esperados
    del modelo Poisson. Por eso las probabilidades finales W/D/L ahora se calculan desde
    la matriz final de marcadores, combinando ML + Poisson + sede.
    """
    feature_cols = training_result["feature_cols"]
    model = training_result["best_model"]

    row = build_future_match_features(mem, history_df, team_a, team_b, neutral=neutral, sede_pais=sede_pais, advanced_context=advanced_context)
    X = row.reindex(columns=feature_cols).fillna(0)

    if hasattr(model, "predict_proba"):
        ml_probs = model.predict_proba(X)[0]
    else:
        pred = model.predict(X)[0]
        ml_probs = np.array([0.0, 0.0, 0.0])
        ml_probs[pred] = 1.0

    lambda_override = None
    goal_model = training_result.get("goal_best_model")
    if goal_model is not None:
        try:
            goal_pred = np.asarray(goal_model.predict(X), dtype=float)[0]
            lambda_override = (clamp(goal_pred[0], 0.05, 5.5), clamp(goal_pred[1], 0.05, 5.5))
        except Exception:
            lambda_override = None

    score_df, lam_a, lam_b = poisson_score_matrix(row, lambda_override=lambda_override, use_dixon_coles=True)

    # Combina ML con Poisson. La sede/anfitrión ya afectó los lambdas de Poisson.
    score_df = blend_poisson_with_ml(score_df, ml_probs, alpha=0.55)

    # Las probabilidades mostradas arriba salen de la matriz final de marcadores,
    # no solo del ML. Así la sede cambia victoria/empate/derrota y no solo marcador exacto.
    outcome_final = (
        score_df.groupby("outcome")["prob_final"]
        .sum()
        .reindex([0, 1, 2], fill_value=0.0)
    )

    probs = {
        "victoria_A": float(outcome_final.loc[0]),
        "empate": float(outcome_final.loc[1]),
        "victoria_B": float(outcome_final.loc[2]),
    }

    probs_ml = {
        "victoria_A": float(ml_probs[0]),
        "empate": float(ml_probs[1]),
        "victoria_B": float(ml_probs[2]),
    }

    top_scores = score_df.sort_values("prob_final", ascending=False).head(10).copy()
    top_scores["probabilidad_%"] = top_scores["prob_final"] * 100

    return {
        "team_a": team_a,
        "team_b": team_b,
        "probs": probs,
        "probs_ml": probs_ml,
        "top_scores": top_scores,
        "lambda_a": lam_a,
        "lambda_b": lam_b,
        "lambda_override": lambda_override,
        "future_features": row
    }


# ============================================================
# 6B. PROBABILIDAD DE CAMPEÓN Y CURVA DE RENDIMIENTO
# ============================================================

def _is_worldcup_2026_match(row):
    try:
        year_ok = pd.to_datetime(row["date"]).year == 2026
    except Exception:
        year_ok = False
    tournament_txt = str(row.get("tournament", "")).lower()
    return year_ok and ("world cup" in tournament_txt)


def dedupe_results_by_match(results_df):
    """
    Deduplica por fecha + pareja de equipos, no por marcador.
    Así, si un resultado fue corregido, se conserva la última versión.
    """
    if results_df is None or results_df.empty:
        return pd.DataFrame()

    df = results_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_team"] = df["home_team"].map(normalize_team_name)
    df["away_team"] = df["away_team"].map(normalize_team_name)
    df["_date_key"] = df["date"].dt.date.astype(str)
    df["_pair_key"] = df.apply(lambda r: "||".join(sorted([str(r["home_team"]), str(r["away_team"])])), axis=1)

    df = df.drop_duplicates(subset=["_date_key", "_pair_key"], keep="last")
    df = df.drop(columns=["_date_key", "_pair_key"], errors="ignore")
    return df.sort_values("date").reset_index(drop=True)


def get_worldcup_2026_results(results_df, teams=None):
    """
    Fuente canónica de partidos jugados del Mundial 2026.
    Combina:
    - calendario interno actualizado,
    - resultados que ya estén cargados en results_df,
    - deduplicación por fecha + equipos.

    Esta función es la que deben usar campeón, mapa, Monte Carlo y precisión.
    """
    if teams is None:
        teams = WC_2026_TEAMS
    teams = set(teams)

    frames = []

    try:
        internal_wc = load_internal_wc_results()
        if internal_wc is not None and not internal_wc.empty:
            frames.append(internal_wc)
    except Exception:
        pass

    if results_df is not None and not results_df.empty:
        df = results_df.copy()
        mask = df.apply(_is_worldcup_2026_match, axis=1)
        df = df.loc[mask].copy()
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    wc = pd.concat(frames, ignore_index=True)
    wc = wc[wc["home_team"].isin(teams) & wc["away_team"].isin(teams)].copy()
    wc = wc.dropna(subset=["home_score", "away_score"])
    wc = dedupe_results_by_match(wc)

    return wc.sort_values("date").reset_index(drop=True)




def worldcup_team_table(results_df, teams=None):
    if teams is None:
        teams = WC_2026_TEAMS

    table = {
        t: {"equipo": t, "equipo_es": TEAM_NAME_ES.get(t, t), "PJ": 0, "PG": 0, "PE": 0, "PP": 0, "GF": 0, "GC": 0, "DG": 0, "PTS": 0}
        for t in teams
    }

    wc = get_worldcup_2026_results(results_df, teams)
    for _, row in wc.iterrows():
        h, a = row["home_team"], row["away_team"]
        hs, aw = int(row["home_score"]), int(row["away_score"])

        table[h]["PJ"] += 1
        table[a]["PJ"] += 1
        table[h]["GF"] += hs
        table[h]["GC"] += aw
        table[a]["GF"] += aw
        table[a]["GC"] += hs

        if hs > aw:
            table[h]["PG"] += 1
            table[h]["PTS"] += 3
            table[a]["PP"] += 1
        elif hs == aw:
            table[h]["PE"] += 1
            table[a]["PE"] += 1
            table[h]["PTS"] += 1
            table[a]["PTS"] += 1
        else:
            table[a]["PG"] += 1
            table[a]["PTS"] += 3
            table[h]["PP"] += 1

    rows = []
    for t in teams:
        r = table[t]
        r["DG"] = r["GF"] - r["GC"]
        rows.append(r)

    return pd.DataFrame(rows)


def estimate_champion_probabilities(mem, results_df, teams=None):
    """
    Estima una probabilidad normalizada de campeón para los 48 equipos.
    Suma total = 100%.
    """
    if teams is None:
        teams = WC_2026_TEAMS

    wc_table = worldcup_team_table(results_df, teams).set_index("equipo")

    rows = []
    snapshots = {}
    elos = []

    for team in teams:
        snap = mem.current_team_snapshot(team)
        snapshots[team] = snap
        elos.append(float(snap.get("elo", 1500)))

    mean_elo = float(np.mean(elos)) if elos else 1500.0

    for team in teams:
        snap = snapshots[team]
        wc = wc_table.loc[team] if team in wc_table.index else None

        elo = float(snap.get("elo", 1500))
        gf10 = float(snap.get("gf_last10", 1.2))
        ga10 = float(snap.get("ga_last10", 1.2))
        pts10 = float(snap.get("points_last10", 1.0))
        win10 = float(snap.get("win_rate_last10", 0.33))
        clean10 = float(snap.get("clean_sheet_last10", 0.20))
        scored10 = float(snap.get("scored_rate_last10", 0.70))

        wc_pts = float(wc["PTS"]) if wc is not None else 0.0
        wc_dg = float(wc["DG"]) if wc is not None else 0.0
        wc_pj = float(wc["PJ"]) if wc is not None else 0.0

        elo_factor = math.exp((elo - mean_elo) / 360.0)
        attack_factor = clamp(1 + 0.18 * (gf10 - 1.20), 0.70, 1.40)
        defense_factor = clamp(1 + 0.16 * (1.20 - ga10), 0.70, 1.40)
        points_factor = clamp(1 + 0.16 * (pts10 - 1.25), 0.72, 1.35)
        consistency_factor = clamp(1 + 0.22 * (win10 - 0.33) + 0.10 * (clean10 - 0.20) + 0.08 * (scored10 - 0.70), 0.75, 1.35)
        wc_factor = clamp(1 + 0.08 * wc_pts + 0.035 * wc_dg - 0.015 * max(0, wc_pj - 1), 0.70, 1.55)

        raw_strength = elo_factor * attack_factor * defense_factor * points_factor * consistency_factor * wc_factor

        rows.append({
            "equipo": team,
            "equipo_es": TEAM_NAME_ES.get(team, team),
            "elo": elo,
            "gf_last10": gf10,
            "ga_last10": ga10,
            "points_last10": pts10,
            "win_rate_last10": win10,
            "clean_sheet_last10": clean10,
            "mundial_PJ": wc_pj,
            "mundial_PTS": wc_pts,
            "mundial_DG": wc_dg,
            "fuerza_bruta": raw_strength,
        })

    df = pd.DataFrame(rows)
    total = float(df["fuerza_bruta"].sum())

    if total <= 0:
        df["prob_campeon_%"] = 100.0 / max(1, len(df))
    else:
        df["prob_campeon_%"] = df["fuerza_bruta"] / total * 100.0

    df["ranking"] = df["prob_campeon_%"].rank(ascending=False, method="first").astype(int)
    return df.sort_values("prob_campeon_%", ascending=False).reset_index(drop=True)


def build_worldcup_performance_timeline(results_df, teams=None):
    """
    Curva desde antes del primer partido.
    Rendimiento = Elo + bono por puntos y diferencia de gol en Mundial.
    Luego se normaliza 0-100.
    """
    if teams is None:
        teams = WC_2026_TEAMS

    teams = list(teams)
    df = results_df.copy().sort_values("date").reset_index(drop=True)
    wc_matches = get_worldcup_2026_results(df, teams)

    elo = {t: 1500.0 for t in teams}
    wc_stats = {t: {"PTS": 0, "DG": 0, "PJ": 0} for t in teams}

    for _, row in df.iterrows():
        if row["date"] >= pd.Timestamp("2026-06-11"):
            break

        h, a = row["home_team"], row["away_team"]
        if h not in elo:
            elo[h] = 1500.0
        if a not in elo:
            elo[a] = 1500.0

        new_h, new_a = update_elo(
            elo[h], elo[a],
            int(row["home_score"]), int(row["away_score"]),
            row["tournament"], neutral=bool(row["neutral"])
        )
        elo[h], elo[a] = new_h, new_a

    snapshots = []

    def add_snapshot(label, fecha):
        for t in teams:
            performance_raw = elo.get(t, 1500.0) + 14 * wc_stats[t]["PTS"] + 5 * wc_stats[t]["DG"]
            snapshots.append({
                "fecha": fecha,
                "momento": label,
                "equipo": t,
                "equipo_es": TEAM_NAME_ES.get(t, t),
                "elo": elo.get(t, 1500.0),
                "PTS": wc_stats[t]["PTS"],
                "DG": wc_stats[t]["DG"],
                "PJ": wc_stats[t]["PJ"],
                "rendimiento_bruto": performance_raw,
            })

    add_snapshot("Antes del Mundial", pd.Timestamp("2026-06-10"))

    wc_counter = 0
    for _, row in wc_matches.iterrows():
        wc_counter += 1
        h, a = row["home_team"], row["away_team"]
        hs, aw = int(row["home_score"]), int(row["away_score"])

        if h not in elo:
            elo[h] = 1500.0
        if a not in elo:
            elo[a] = 1500.0

        new_h, new_a = update_elo(elo[h], elo[a], hs, aw, row["tournament"], neutral=True)
        elo[h], elo[a] = new_h, new_a

        for t in [h, a]:
            if t in wc_stats:
                wc_stats[t]["PJ"] += 1

        if h in wc_stats:
            wc_stats[h]["DG"] += hs - aw
        if a in wc_stats:
            wc_stats[a]["DG"] += aw - hs

        if hs > aw:
            if h in wc_stats:
                wc_stats[h]["PTS"] += 3
        elif hs == aw:
            if h in wc_stats:
                wc_stats[h]["PTS"] += 1
            if a in wc_stats:
                wc_stats[a]["PTS"] += 1
        else:
            if a in wc_stats:
                wc_stats[a]["PTS"] += 3

        add_snapshot(f"{row['date'].date()} · Partido {wc_counter}", row["date"])

    timeline = pd.DataFrame(snapshots)
    if timeline.empty:
        return timeline

    min_v = timeline["rendimiento_bruto"].min()
    max_v = timeline["rendimiento_bruto"].max()

    if max_v == min_v:
        timeline["rendimiento_0_100"] = 50.0
    else:
        timeline["rendimiento_0_100"] = (timeline["rendimiento_bruto"] - min_v) / (max_v - min_v) * 100.0

    return timeline


def plot_champion_probabilities(champ_df, top_n=16):
    d = champ_df.head(top_n).copy().sort_values("prob_campeon_%", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(d["equipo_es"], d["prob_campeon_%"])
    ax.set_xlabel("Probabilidad estimada de campeón (%)")
    ax.set_title(f"Top {top_n} probabilidades estimadas de ganar el Mundial")
    for i, v in enumerate(d["prob_campeon_%"]):
        ax.text(v, i, f" {v:.2f}%", va="center")
    fig.tight_layout()
    return fig


def plot_performance_timeline(timeline_df, selected_teams):
    fig, ax = plt.subplots(figsize=(11, 6))

    if timeline_df is None or timeline_df.empty or not selected_teams:
        ax.set_title("Curva de rendimiento")
        ax.text(0.5, 0.5, "No hay datos suficientes para graficar.", ha="center", va="center")
        return fig

    d = timeline_df[timeline_df["equipo"].isin(selected_teams)].copy()
    moments = list(dict.fromkeys(timeline_df["momento"].tolist()))
    moment_to_x = {m: i for i, m in enumerate(moments)}
    d["x"] = d["momento"].map(moment_to_x)

    for team in selected_teams:
        dt = d[d["equipo"] == team].sort_values("x")
        if dt.empty:
            continue
        ax.plot(dt["x"], dt["rendimiento_0_100"], marker="o", label=TEAM_NAME_ES.get(team, team))

    ax.set_xlabel("Avance del Mundial")
    ax.set_ylabel("Rendimiento relativo (0-100)")
    ax.set_title("Evolución del rendimiento por equipo")
    ax.set_xticks(range(len(moments)))
    ax.set_xticklabels(moments, rotation=45, ha="right")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig



# ============================================================
# 6C. MONTE CARLO, BACKTESTING Y CALIDAD DE DATOS
# ============================================================

def get_played_worldcup_match_lookup(results_df):
    """Diccionario de partidos ya jugados en Mundial 2026."""
    wc = get_worldcup_2026_results(results_df, WC_2026_TEAMS)
    lookup = {}
    for _, row in wc.iterrows():
        key = tuple(sorted([row["home_team"], row["away_team"]]))
        lookup[key] = {
            "home": row["home_team"],
            "away": row["away_team"],
            "home_score": int(row["home_score"]),
            "away_score": int(row["away_score"]),
        }
    return lookup


def simulate_match_score_from_prediction(prediction, rng):
    """Muestrea un marcador desde la matriz final de marcadores."""
    scores = prediction["top_scores"].copy()
    # Para no sesgar por solo top 10, recomputamos con future_features y probs_ml si es posible.
    row = prediction["future_features"]
    score_df, _, _ = poisson_score_matrix(row)
    # Si existe prob_final en top, usamos la matriz Poisson base como fallback razonable.
    probs = score_df["prob_poisson"].values
    idx = rng.choice(len(score_df), p=probs / probs.sum())
    s = score_df.iloc[idx]
    return int(s["goles_A"]), int(s["goles_B"])


def predict_outcome_probs_fast(training_result, mem, results_df, team_a, team_b):
    """
    Probabilidades rápidas para simulación.
    Sin variables externas, para que Monte Carlo no haga consultas web.
    """
    pred = predict_match(
        training_result, mem, results_df,
        team_a, team_b,
        sede_pais="Sin ventaja de anfitrión",
        advanced_context={
            "temperature_c": 22,
            "humidity_pct": 55,
            "altitude_m": 300,
            "crowd_support_a": 5,
            "crowd_support_b": 5,
            "pressure_a": 5,
            "pressure_b": 5,
            "cohesion_a": 7,
            "cohesion_b": 7,
            "controversy_a": 0,
            "controversy_b": 0,
            "lineup_rating_a": 5.5,
            "lineup_rating_b": 5.5,
            "stage": "Fase de grupos",
            "urgency_a": 5,
            "urgency_b": 5,
        }
    )
    return pred


def simulate_group_stage_once(training_result, mem, results_df, rng):
    """
    Simula fase de grupos:
    - usa resultados reales ya cargados;
    - simula partidos faltantes dentro de cada grupo;
    - clasifica top 2 de cada grupo + 8 mejores terceros.
    """
    played = get_played_worldcup_match_lookup(results_df)
    group_tables = {}

    for group, teams in WC_2026_GROUPS.items():
        table = {
            t: {"team": t, "group": group, "PTS": 0, "GF": 0, "GA": 0, "GD": 0, "W": 0, "D": 0, "L": 0}
            for t in teams
        }

        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                a, b = teams[i], teams[j]
                key = tuple(sorted([a, b]))

                if key in played:
                    p = played[key]
                    # Convertir al orden a,b
                    if p["home"] == a:
                        ga, gb = p["home_score"], p["away_score"]
                    else:
                        ga, gb = p["away_score"], p["home_score"]
                else:
                    pred = predict_outcome_probs_fast(training_result, mem, results_df, a, b)
                    ga, gb = simulate_match_score_from_prediction(pred, rng)

                table[a]["GF"] += ga
                table[a]["GA"] += gb
                table[b]["GF"] += gb
                table[b]["GA"] += ga

                if ga > gb:
                    table[a]["PTS"] += 3
                    table[a]["W"] += 1
                    table[b]["L"] += 1
                elif ga == gb:
                    table[a]["PTS"] += 1
                    table[b]["PTS"] += 1
                    table[a]["D"] += 1
                    table[b]["D"] += 1
                else:
                    table[b]["PTS"] += 3
                    table[b]["W"] += 1
                    table[a]["L"] += 1

        for t in teams:
            table[t]["GD"] = table[t]["GF"] - table[t]["GA"]

        group_tables[group] = pd.DataFrame(table.values()).sort_values(
            ["PTS", "GD", "GF"], ascending=[False, False, False]
        ).reset_index(drop=True)

    qualifiers = []
    thirds = []

    for group, tbl in group_tables.items():
        qualifiers.extend(tbl.head(2)["team"].tolist())
        thirds.append(tbl.iloc[2].to_dict())

    thirds_df = pd.DataFrame(thirds).sort_values(
        ["PTS", "GD", "GF"], ascending=[False, False, False]
    ).head(8)

    qualifiers.extend(thirds_df["team"].tolist())

    return qualifiers, group_tables


def get_team_strength_for_knockout(mem, results_df, team):
    """Fuerza base para desempatar eliminatorias si hay empate simulado."""
    snap = mem.current_team_snapshot(team)
    wc = worldcup_team_table(results_df, WC_2026_TEAMS).set_index("equipo")
    wc_pts = float(wc.loc[team]["PTS"]) if team in wc.index else 0
    wc_dg = float(wc.loc[team]["DG"]) if team in wc.index else 0
    return float(snap.get("elo", 1500)) + 18 * wc_pts + 6 * wc_dg


def simulate_knockout_once(training_result, mem, results_df, qualifiers, rng):
    """
    Simulación aproximada de eliminatorias.
    No replica necesariamente el bracket FIFA oficial: ordena clasificados por fuerza y cruza 1 vs 32, 2 vs 31, etc.
    """
    seeded = sorted(
        qualifiers,
        key=lambda t: get_team_strength_for_knockout(mem, results_df, t),
        reverse=True
    )

    round_teams = seeded[:32]

    while len(round_teams) > 1:
        winners = []
        for i in range(len(round_teams) // 2):
            a = round_teams[i]
            b = round_teams[-(i + 1)]

            pred = predict_outcome_probs_fast(training_result, mem, results_df, a, b)
            probs = pred["probs"]
            p = np.array([probs["victoria_A"], probs["empate"], probs["victoria_B"]], dtype=float)
            p = p / p.sum()
            outcome = rng.choice([0, 1, 2], p=p)

            if outcome == 0:
                winners.append(a)
            elif outcome == 2:
                winners.append(b)
            else:
                # Empate en 90: avanza por fuerza + ruido.
                strength_a = get_team_strength_for_knockout(mem, results_df, a)
                strength_b = get_team_strength_for_knockout(mem, results_df, b)
                p_a = 1 / (1 + math.exp(-(strength_a - strength_b) / 220))
                winners.append(a if rng.random() < p_a else b)

        round_teams = winners

    return round_teams[0]


def monte_carlo_worldcup(training_result, mem, results_df, n_simulations=1000, seed=42):
    """
    Simulación Monte Carlo aproximada del Mundial completo.
    Devuelve probabilidad de campeón por equipo.
    """
    rng = np.random.default_rng(seed)
    champions = {t: 0 for t in WC_2026_TEAMS}

    for _ in range(int(n_simulations)):
        qualifiers, _ = simulate_group_stage_once(training_result, mem, results_df, rng)
        champion = simulate_knockout_once(training_result, mem, results_df, qualifiers, rng)
        champions[champion] += 1

    rows = []
    for team, count in champions.items():
        rows.append({
            "equipo": team,
            "equipo_es": TEAM_NAME_ES.get(team, team),
            "campeon_simulaciones": count,
            "prob_campeon_montecarlo_%": count / max(1, n_simulations) * 100,
        })

    df = pd.DataFrame(rows).sort_values("prob_campeon_montecarlo_%", ascending=False).reset_index(drop=True)
    df["ranking_mc"] = df["prob_campeon_montecarlo_%"].rank(ascending=False, method="first").astype(int)
    return df


def fit_best_backtest_model(train_df, feature_cols):
    """
    Selecciona el mejor modelo para backtesting usando validación temporal.
    Evita escoger a ciegas un solo modelo.
    """
    train_df = train_df.sort_values("date").reset_index(drop=True)
    if len(train_df) < 300:
        X_train = train_df[feature_cols]
        y_train = train_df["target"]
        model = make_calibrated_logistic()
        model.fit(X_train, y_train)
        return "Logistic Regression calibrada", model, pd.DataFrame()

    split_idx = int(len(train_df) * 0.82)
    sub_train = train_df.iloc[:split_idx].copy()
    val = train_df.iloc[split_idx:].copy()

    X_sub = sub_train[feature_cols]
    y_sub = sub_train["target"]
    X_val = val[feature_cols]
    y_val = val["target"]

    models = get_models()
    try:
        models["Logistic Regression calibrada"] = make_calibrated_logistic()
    except Exception:
        pass

    rows = []
    fitted_val = {}

    for name, model in models.items():
        try:
            model.fit(X_sub, y_sub)
            pred = model.predict(X_val)

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_val)
                ll = log_loss(y_val, proba, labels=[0, 1, 2])
                brier = multiclass_brier_score(y_val, proba, labels=[0, 1, 2])
            else:
                ll = np.nan
                brier = np.nan

            acc = accuracy_score(y_val, pred)
            f1 = f1_score(y_val, pred, average="weighted")

            rows.append({
                "modelo": name,
                "accuracy_val": acc,
                "f1_val": f1,
                "log_loss_val": ll,
                "brier_val": brier,
            })
            fitted_val[name] = model
        except Exception as e:
            rows.append({
                "modelo": name,
                "accuracy_val": np.nan,
                "f1_val": np.nan,
                "log_loss_val": np.nan,
                "brier_val": np.nan,
                "error": str(e)[:120]
            })

    metrics = pd.DataFrame(rows)

    valid = metrics.dropna(subset=["log_loss_val"])
    if not valid.empty:
        best_name = valid.sort_values(["log_loss_val", "brier_val", "f1_val"], ascending=[True, True, False]).iloc[0]["modelo"]
    else:
        valid = metrics.dropna(subset=["f1_val"])
        best_name = valid.sort_values(["f1_val", "accuracy_val"], ascending=[False, False]).iloc[0]["modelo"]

    # Reentrenar el mejor modelo con todo el bloque pre-Mundial.
    final_model = models[best_name]
    final_model.fit(train_df[feature_cols], train_df["target"])

    return best_name, final_model, metrics


def backtest_world_cups(features_df=None, feature_cols=None, min_year_backtest=2010):
    """
    Backtesting mejorado:
    - reconstruye una base histórica independiente desde 2010;
    - evalúa estrictamente partidos cuyo tournament contiene 'World Cup';
    - usa selección de modelo con validación temporal;
    - no depende del slider principal.
    """
    rows = []

    windows = {
        2018: ("2018-06-14", "2018-07-15"),
        2022: ("2022-11-20", "2022-12-18"),
    }

    try:
        hist = download_historical_results()
        hist = clean_results_df(hist)
        bt_features, bt_feature_cols, _ = build_features_from_results(hist, min_year=min_year_backtest)
    except Exception as e:
        return pd.DataFrame([{
            "torneo": "Error",
            "partidos_test": 0,
            "accuracy": np.nan,
            "log_loss": np.nan,
            "brier_score": np.nan,
            "mejor_modelo_bt": "",
            "nota": f"No se pudo reconstruir base histórica: {str(e)[:160]}"
        }])

    if bt_features.empty:
        return pd.DataFrame([{
            "torneo": "Sin datos",
            "partidos_test": 0,
            "accuracy": np.nan,
            "log_loss": np.nan,
            "brier_score": np.nan,
            "mejor_modelo_bt": "",
            "nota": "La base histórica reconstruida quedó vacía."
        }])

    for year, (start_txt, end_txt) in windows.items():
        start = pd.Timestamp(start_txt)
        end = pd.Timestamp(end_txt)

        train = bt_features[bt_features["date"] < start].copy()

        test = bt_features[
            (bt_features["date"] >= start) &
            (bt_features["date"] <= end) &
            (bt_features["tournament"].astype(str).str.contains("World Cup", case=False, na=False))
        ].copy()

        if len(train) < 100 or len(test) < 5:
            rows.append({
                "torneo": f"Mundial {year}",
                "partidos_test": len(test),
                "accuracy": np.nan,
                "log_loss": np.nan,
                "brier_score": np.nan,
                "mejor_modelo_bt": "",
                "nota": "No hay suficientes partidos exactos de Mundial para esta ventana."
            })
            continue

        try:
            best_name, model, val_metrics = fit_best_backtest_model(train, bt_feature_cols)

            X_test = test[bt_feature_cols]
            y_test = test["target"]

            pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)
                ll = log_loss(y_test, proba, labels=[0, 1, 2])
                brier = multiclass_brier_score(y_test, proba, labels=[0, 1, 2])
            else:
                ll = np.nan
                brier = np.nan

            rows.append({
                "torneo": f"Mundial {year}",
                "partidos_test": len(test),
                "accuracy": accuracy_score(y_test, pred),
                "log_loss": ll,
                "brier_score": brier,
                "mejor_modelo_bt": best_name,
                "nota": "Filtrado estricto por tournament = World Cup. Base histórica independiente desde 2010."
            })
        except Exception as e:
            rows.append({
                "torneo": f"Mundial {year}",
                "partidos_test": len(test),
                "accuracy": np.nan,
                "log_loss": np.nan,
                "brier_score": np.nan,
                "mejor_modelo_bt": "",
                "nota": str(e)[:160]
            })

    return pd.DataFrame(rows)




def compute_data_quality_index(auto_sources, advanced_context, prediction, mem, team_a, team_b):
    """
    Índice de calidad de datos 0-100.
    Mide qué tan alimentada está la predicción.
    """
    score = 0
    details = []

    weather_ok = bool(auto_sources.get("weather", {}).get("ok", False))
    score += 15 if weather_ok else 5
    details.append(("Clima", 15 if weather_ok else 5, "Actualizado" if weather_ok else "No confirmado"))

    api_lineups = bool(auto_sources.get("api_lineups_used", False))
    lineup_a_players = int(auto_sources.get("lineup_a", {}).get("players_used", 0))
    lineup_b_players = int(auto_sources.get("lineup_b", {}).get("players_used", 0))
    if api_lineups:
        lineup_score = 25
        lineup_msg = "Alineación API disponible"
    elif lineup_a_players >= 8 and lineup_b_players >= 8:
        lineup_score = 18
        lineup_msg = "Ratings/última alineación local suficiente"
    else:
        lineup_score = 8
        lineup_msg = "Alineaciones incompletas"
    score += lineup_score
    details.append(("Alineaciones", lineup_score, lineup_msg))

    news_a = auto_sources.get("news_a", "")
    news_b = auto_sources.get("news_b", "")
    news_score = 15 if ("noticias encontradas" in str(news_a) and "noticias encontradas" in str(news_b)) else 7
    score += news_score
    details.append(("Noticias/cohesión", news_score, "Noticias consultadas" if news_score == 15 else "Noticias limitadas"))

    snap_a = mem.current_team_snapshot(team_a)
    snap_b = mem.current_team_snapshot(team_b)
    recent_count = min(snap_a.get("matches_count_last10", 0), snap_b.get("matches_count_last10", 0))
    recent_score = 15 if recent_count >= 8 else 10 if recent_count >= 5 else 5
    score += recent_score
    details.append(("Histórico reciente", recent_score, f"Mínimo partidos recientes: {recent_count}"))

    fixture_ok = auto_sources.get("fixture_info") is not None and not isinstance(auto_sources.get("fixture_info"), str)
    internal_ok = auto_sources.get("internal_fixture") is not None
    fixture_score = 15 if fixture_ok else 12 if internal_ok else 7
    score += fixture_score
    if fixture_ok:
        fixture_msg = "Fixture detectado"
    elif internal_ok:
        fixture_msg = "Fixture detectado por calendario interno"
    else:
        fixture_msg = "Sede manual o no detectada"
    details.append(("Fixture/sede", fixture_score, fixture_msg))

    goal_model_ok = prediction.get("lambda_override") is not None
    score += 15 if goal_model_ok else 8
    details.append(("Modelo de goles", 15 if goal_model_ok else 8, "Regresor de goles activo" if goal_model_ok else "Solo Poisson/ML"))

    final_score = int(clamp(score, 0, 100))
    if final_score >= 80:
        label = "Alta"
    elif final_score >= 60:
        label = "Media"
    else:
        label = "Baja"

    details_df = pd.DataFrame(details, columns=["Componente", "Puntos", "Diagnóstico"])
    return final_score, label, details_df




# ============================================================
# 6D. MAPA PROFESIONAL DEL MUNDIAL Y PRECISIÓN DEL MODELO
# ============================================================

def actual_outcome_from_score(home_score, away_score):
    if int(home_score) > int(away_score):
        return 0
    if int(home_score) == int(away_score):
        return 1
    return 2


def outcome_text(outcome, team_a=None, team_b=None):
    if outcome == 0:
        return f"Gana {team_a}" if team_a else "Gana Equipo A"
    if outcome == 1:
        return "Empate"
    if outcome == 2:
        return f"Gana {team_b}" if team_b else "Gana Equipo B"
    return "Sin dato"


def get_worldcup_group_tables_from_calendar(results_df=None):
    """Tabla de grupos basada en calendario interno + resultados."""
    tables = {}
    cal = load_internal_wc_calendar()

    for group, teams in WC_2026_GROUPS.items():
        base = {
            t: {"grupo": group, "equipo": t, "equipo_es": TEAM_NAME_ES.get(t, t), "PJ": 0, "PG": 0, "PE": 0, "PP": 0, "GF": 0, "GC": 0, "DG": 0, "PTS": 0}
            for t in teams
        }

        if cal is not None and not cal.empty:
            gmatches = cal[cal.get("group", "") == group].copy()
            for _, row in gmatches.iterrows():
                if str(row.get("status", "")).lower() not in ["complete", "final", "finished"]:
                    continue
                if pd.isna(row.get("home_score")) or pd.isna(row.get("away_score")):
                    continue

                h, a = row["home_team"], row["away_team"]
                hs, aw = int(row["home_score"]), int(row["away_score"])
                if h not in base or a not in base:
                    continue

                base[h]["PJ"] += 1
                base[a]["PJ"] += 1
                base[h]["GF"] += hs
                base[h]["GC"] += aw
                base[a]["GF"] += aw
                base[a]["GC"] += hs

                if hs > aw:
                    base[h]["PG"] += 1
                    base[h]["PTS"] += 3
                    base[a]["PP"] += 1
                elif hs == aw:
                    base[h]["PE"] += 1
                    base[a]["PE"] += 1
                    base[h]["PTS"] += 1
                    base[a]["PTS"] += 1
                else:
                    base[a]["PG"] += 1
                    base[a]["PTS"] += 3
                    base[h]["PP"] += 1

        rows = []
        for team in teams:
            r = base[team]
            r["DG"] = r["GF"] - r["GC"]
            rows.append(r)

        tbl = pd.DataFrame(rows).sort_values(["PTS", "DG", "GF"], ascending=[False, False, False]).reset_index(drop=True)
        tbl["Pos"] = np.arange(1, len(tbl) + 1)

        max_pj = int(tbl["PJ"].max()) if not tbl.empty else 0
        if max_pj >= 3:
            tbl["Estado"] = tbl["Pos"].apply(lambda x: "Clasifica directo" if x <= 2 else "Tercero en disputa" if x == 3 else "Eliminado")
        else:
            tbl["Estado"] = tbl["Pos"].apply(lambda x: "Zona directa" if x <= 2 else "Tercero en disputa" if x == 3 else "En riesgo")

        tables[group] = tbl

    return tables


def build_projected_qualifiers(champion_df, results_df=None):
    """Proyección rápida de clasificados: top 2 + 8 mejores terceros."""
    tables = get_worldcup_group_tables_from_calendar(results_df)
    strength = champion_df.set_index("equipo")["prob_campeon_%"].to_dict() if champion_df is not None and not champion_df.empty else {}

    qualifiers = []
    third_rows = []

    for group, tbl in tables.items():
        t = tbl.copy()
        t["fuerza"] = t["equipo"].map(strength).fillna(0.0)
        t = t.sort_values(["PTS", "DG", "GF", "fuerza"], ascending=[False, False, False, False]).reset_index(drop=True)

        for _, row in t.head(2).iterrows():
            qualifiers.append({"equipo": row["equipo"], "grupo": group, "via": f"{int(row['Pos'])}° Grupo {group}", "PTS": row["PTS"], "DG": row["DG"], "fuerza": row["fuerza"]})

        if len(t) >= 3:
            third = t.iloc[2].copy()
            third_rows.append({"equipo": third["equipo"], "grupo": group, "via": f"3° Grupo {group}", "PTS": third["PTS"], "DG": third["DG"], "GF": third["GF"], "fuerza": third["fuerza"]})

    third_df = pd.DataFrame(third_rows)
    if not third_df.empty:
        third_df = third_df.sort_values(["PTS", "DG", "GF", "fuerza"], ascending=[False, False, False, False]).head(8)
        qualifiers.extend(third_df[["equipo", "grupo", "via", "PTS", "DG", "fuerza"]].to_dict("records"))

    qdf = pd.DataFrame(qualifiers)
    if qdf.empty:
        return qdf

    qdf["equipo_es"] = qdf["equipo"].map(lambda t: TEAM_NAME_ES.get(t, t))
    qdf["bandera"] = qdf["equipo"].map(lambda t: TEAM_FLAGS.get(t, "🏳️"))
    qdf = qdf.sort_values(["fuerza", "PTS", "DG"], ascending=[False, False, False]).reset_index(drop=True)
    qdf["seed_aprox"] = np.arange(1, len(qdf) + 1)
    return qdf


def build_projected_bracket_pairs(champion_df, results_df=None):
    qdf = build_projected_qualifiers(champion_df, results_df)
    if qdf.empty:
        return pd.DataFrame()

    teams = qdf.head(32).copy()
    rows = []
    n = len(teams)
    for i in range(n // 2):
        a = teams.iloc[i]
        b = teams.iloc[n - 1 - i]
        rows.append({
            "llave": i + 1,
            "equipo_a": a["equipo"],
            "equipo_b": b["equipo"],
            "equipo_a_es": a["equipo_es"],
            "equipo_b_es": b["equipo_es"],
            "flag_a": a["bandera"],
            "flag_b": b["bandera"],
            "prob_a": float(a["fuerza"]),
            "prob_b": float(b["fuerza"]),
            "via_a": a["via"],
            "via_b": b["via"],
        })
    return pd.DataFrame(rows)


def render_worldcup_map_html(champion_df, results_df, mc_df=None):
    """Renderiza un mapa/dashboard interactivo en HTML."""
    tables = get_worldcup_group_tables_from_calendar(results_df)
    bracket = build_projected_bracket_pairs(champion_df, results_df)

    fav_df = champion_df.copy()
    if mc_df is not None and not mc_df.empty:
        fav_df = fav_df.merge(mc_df[["equipo", "prob_campeon_montecarlo_%"]], on="equipo", how="left")
    else:
        fav_df["prob_campeon_montecarlo_%"] = np.nan

    fav_df = fav_df.head(8)

    def pct_bar(value, max_value=8):
        try:
            v = float(value)
        except Exception:
            v = 0
        width = max(4, min(100, v / max_value * 100))
        return f'<div class="bar"><span style="width:{width:.1f}%"></span></div>'

    group_html = []
    for group, tbl in tables.items():
        rows_html = []
        for _, r in tbl.iterrows():
            team = r["equipo"]
            estado = r["Estado"]
            status_class = "ok" if "directa" in estado.lower() or "clasifica" in estado.lower() else "warn" if "tercero" in estado.lower() else "risk"
            title = (
                f"{TEAM_NAME_ES.get(team, team)} | Grupo {group}\\n"
                f"PTS: {r['PTS']} | PJ: {r['PJ']} | DG: {r['DG']} | GF: {r['GF']} | GC: {r['GC']}\\n"
                f"Estado: {estado}"
            )
            rows_html.append(f"""
              <div class="team-row {status_class}" title="{title}">
                <div class="team-main"><span class="pos">{int(r["Pos"])}</span><span class="flag">{TEAM_FLAGS.get(team, "🏳️")}</span><b>{TEAM_NAME_ES.get(team, team)}</b></div>
                <div class="team-stats"><span>{int(r["PTS"])} pts</span><span>DG {int(r["DG"])}</span><span>PJ {int(r["PJ"])}</span></div>
              </div>
            """)
        group_html.append(f"""
          <div class="group-card">
            <div class="group-title">Grupo {group}</div>
            {''.join(rows_html)}
          </div>
        """)

    fav_html = []
    max_base = max(1.0, float(fav_df["prob_campeon_%"].max()) if not fav_df.empty else 1.0)
    for _, r in fav_df.iterrows():
        team = r["equipo"]
        mc_txt = "" if pd.isna(r.get("prob_campeon_montecarlo_%")) else f" · MC {float(r['prob_campeon_montecarlo_%']):.2f}%"
        title = f"{TEAM_NAME_ES.get(team, team)}\\nProb. base campeón: {float(r['prob_campeon_%']):.2f}%{mc_txt}\\nElo: {float(r.get('elo', 0)):.1f}"
        fav_html.append(f"""
          <div class="fav-row" title="{title}">
            <div><span class="flag">{TEAM_FLAGS.get(team, "🏳️")}</span><b>{TEAM_NAME_ES.get(team, team)}</b><small>{float(r["prob_campeon_%"]):.2f}%{mc_txt}</small></div>
            {pct_bar(float(r["prob_campeon_%"]), max_value=max_base)}
          </div>
        """)

    bracket_html = []
    for _, r in bracket.iterrows():
        total = max(0.0001, float(r["prob_a"]) + float(r["prob_b"]))
        pa = float(r["prob_a"]) / total * 100
        pb = 100 - pa
        title_a = f"{r['equipo_a_es']} | {r['via_a']} | fuerza base {r['prob_a']:.2f}%"
        title_b = f"{r['equipo_b_es']} | {r['via_b']} | fuerza base {r['prob_b']:.2f}%"
        bracket_html.append(f"""
          <div class="match-card">
            <div class="match-title">R32 · Llave {int(r["llave"])}</div>
            <div class="match-team" title="{title_a}"><span>{r["flag_a"]} <b>{r["equipo_a_es"]}</b></span><small>{pa:.1f}% fuerza relativa</small></div>
            <div class="versus">vs</div>
            <div class="match-team" title="{title_b}"><span>{r["flag_b"]} <b>{r["equipo_b_es"]}</b></span><small>{pb:.1f}% fuerza relativa</small></div>
          </div>
        """)

    html = f"""
    <div class="wc-map">
      <style>
        .wc-map {{
          font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: linear-gradient(135deg, #0f172a 0%, #111827 48%, #052e2b 100%);
          color: #eef2ff; border-radius: 24px; padding: 22px;
          box-shadow: 0 18px 45px rgba(15,23,42,.28);
        }}
        .wc-header {{ display:flex; justify-content:space-between; align-items:flex-start; gap:18px; margin-bottom:18px; }}
        .wc-header h2 {{ margin:0; font-size:28px; letter-spacing:.2px; }}
        .wc-header p {{ margin:6px 0 0; color:#cbd5e1; font-size:14px; }}
        .pill {{ background:rgba(34,197,94,.14); border:1px solid rgba(34,197,94,.35); color:#bbf7d0; padding:8px 12px; border-radius:999px; font-weight:700; white-space:nowrap; }}
        .section-title {{ margin:22px 0 10px; font-size:18px; font-weight:800; color:#fff; }}
        .groups-grid {{ display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap:12px; }}
        .group-card {{ background:rgba(255,255,255,.075); border:1px solid rgba(255,255,255,.14); border-radius:18px; padding:12px; backdrop-filter: blur(8px); }}
        .group-title {{ font-weight:900; margin-bottom:8px; color:#fef3c7; }}
        .team-row {{ padding:8px; border-radius:12px; margin:6px 0; background:rgba(15,23,42,.42); border-left:4px solid #64748b; cursor:help; transition:.16s; }}
        .team-row:hover {{ transform: translateY(-2px); background:rgba(255,255,255,.13); }}
        .team-row.ok {{ border-left-color:#22c55e; }} .team-row.warn {{ border-left-color:#f59e0b; }} .team-row.risk {{ border-left-color:#ef4444; }}
        .team-main {{ display:flex; align-items:center; gap:8px; font-size:13px; }}
        .pos {{ width:20px; height:20px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center; background:rgba(255,255,255,.12); color:#e2e8f0; font-size:11px; }}
        .flag {{ font-size:18px; }}
        .team-stats {{ display:flex; justify-content:space-between; color:#cbd5e1; font-size:11px; margin-top:4px; gap:6px; }}
        .dashboard-grid {{ display:grid; grid-template-columns: 1fr 1.4fr; gap:14px; }}
        .panel {{ background:rgba(255,255,255,.075); border:1px solid rgba(255,255,255,.14); border-radius:18px; padding:14px; }}
        .fav-row {{ margin:10px 0; cursor:help; }}
        .fav-row > div:first-child {{ display:flex; justify-content:space-between; gap:10px; align-items:center; font-size:13px; }}
        .fav-row small {{ color:#cbd5e1; }}
        .bar {{ height:8px; background:rgba(255,255,255,.13); border-radius:99px; overflow:hidden; margin-top:6px; }}
        .bar span {{ display:block; height:100%; background:linear-gradient(90deg,#22c55e,#38bdf8); border-radius:99px; }}
        .bracket-grid {{ display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap:10px; }}
        .match-card {{ background:rgba(2,6,23,.44); border:1px solid rgba(148,163,184,.24); border-radius:15px; padding:10px; }}
        .match-card:hover {{ border-color:#38bdf8; }}
        .match-title {{ color:#fef3c7; font-size:11px; font-weight:800; margin-bottom:8px; text-transform:uppercase; letter-spacing:.04em; }}
        .match-team {{ display:flex; justify-content:space-between; align-items:center; gap:8px; padding:7px; border-radius:11px; background:rgba(255,255,255,.07); cursor:help; font-size:12px; }}
        .match-team small {{ color:#cbd5e1; white-space:nowrap; }}
        .versus {{ text-align:center; color:#94a3b8; font-size:11px; margin:4px 0; font-weight:800; }}
        .legend {{ display:flex; gap:10px; flex-wrap:wrap; color:#cbd5e1; font-size:12px; margin-top:10px; }}
        .legend span {{ display:inline-flex; align-items:center; gap:6px; }}
        .dot {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}
        .dot.green {{ background:#22c55e; }} .dot.yellow {{ background:#f59e0b; }} .dot.red {{ background:#ef4444; }}
        @media (max-width: 1000px) {{
          .groups-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
          .dashboard-grid {{ grid-template-columns: 1fr; }}
          .bracket-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
        }}
      </style>
      <div class="wc-header">
        <div>
          <h2>🗺️ Mapa vivo del Mundial 2026</h2>
          <p>Posiciones, favoritos y ruta proyectada. Pasa el puntero sobre cada equipo o llave para ver más información.</p>
        </div>
        <div class="pill">Actualizable · Calendario interno + fuentes gratuitas</div>
      </div>
      <div class="section-title">Fase de grupos</div>
      <div class="groups-grid">{''.join(group_html)}</div>
      <div class="legend">
        <span><i class="dot green"></i>Zona directa</span>
        <span><i class="dot yellow"></i>Tercero / disputa</span>
        <span><i class="dot red"></i>Riesgo o eliminado</span>
      </div>
      <div class="section-title">Favoritos y ruta proyectada</div>
      <div class="dashboard-grid">
        <div class="panel"><b>🔥 Favoritos actuales</b>{''.join(fav_html)}</div>
        <div class="panel">
          <b>🧩 Eliminatoria proyectada aproximada</b>
          <p style="color:#cbd5e1;font-size:12px;margin-top:6px;">Proyección educativa: cruza 32 clasificados estimados por fuerza. No reemplaza el bracket oficial FIFA.</p>
          <div class="bracket-grid">{''.join(bracket_html)}</div>
        </div>
      </div>
    </div>
    """
    return html


def get_precision_start_datetime():
    """
    Devuelve la fecha desde la cual se empieza a medir precisión.
    Por defecto: Japan vs Tunisia / Tunisia vs Japan.
    """
    try:
        cal = load_internal_wc_calendar()
        if cal is not None and not cal.empty:
            m = cal[
                (
                    (cal["fixture_id"].astype(str) == str(PRECISION_START_FIXTURE_ID)) |
                    (
                        cal["home_team"].isin(PRECISION_START_TEAMS) &
                        cal["away_team"].isin(PRECISION_START_TEAMS)
                    )
                )
            ].copy()
            if not m.empty:
                d = pd.to_datetime(m.iloc[0]["date"], errors="coerce")
                if pd.notna(d):
                    return d
    except Exception:
        pass

    return pd.to_datetime(PRECISION_START_DATE)


def evaluate_worldcup_prediction_accuracy(training_result, results_df, min_year=2024, start_from_precision_match=True):
    """
    Evalúa acumulado de aciertos en partidos ya jugados del Mundial.
    Desde V15, por defecto comienza desde Japan vs Tunisia/Tunisia vs Japan.
    """
    wc = get_worldcup_2026_results(results_df, WC_2026_TEAMS)
    if wc.empty:
        return pd.DataFrame(), {}

    wc = wc.dropna(subset=["home_score", "away_score"]).copy()
    if wc.empty:
        return pd.DataFrame(), {}

    wc["date_dt"] = pd.to_datetime(wc["date"], errors="coerce")

    precision_start_dt = get_precision_start_datetime()
    if start_from_precision_match and pd.notna(precision_start_dt):
        wc = wc[wc["date_dt"] >= precision_start_dt].copy()

    if wc.empty:
        return pd.DataFrame(), {
            "partidos_evaluados": 0,
            "acierto_1x2_%": 0.0,
            "acierto_marcador_top1_%": 0.0,
            "acierto_marcador_top3_%": 0.0,
            "aciertos_1x2": 0,
            "precision_start": precision_start_dt.date().isoformat() if pd.notna(precision_start_dt) else PRECISION_START_DATE,
            "precision_label": PRECISION_START_LABEL,
            "mensaje": f"La medición de precisión empieza desde {PRECISION_START_LABEL}. Aún no hay partidos completados desde ese punto."
        }

    rows = []
    for _, match in wc.iterrows():
        try:
            h, a = match["home_team"], match["away_team"]
            hs, aw = int(match["home_score"]), int(match["away_score"])
            match_date = pd.to_datetime(match["date"])
            past = results_df[pd.to_datetime(results_df["date"]) < match_date].copy()

            mem_used = st.session_state.mem
            if len(past) > 200:
                try:
                    _, _, mem_before = build_features_from_results(past, min_year=min_year)
                    mem_used = mem_before
                except Exception:
                    pass

            pred = predict_match(
                training_result,
                mem_used,
                past if not past.empty else results_df,
                h,
                a,
                sede_pais="Sin ventaja de anfitrión",
                advanced_context={
                    "temperature_c": 22, "humidity_pct": 55, "altitude_m": 300,
                    "crowd_support_a": 5, "crowd_support_b": 5,
                    "pressure_a": 5, "pressure_b": 5,
                    "cohesion_a": 7, "cohesion_b": 7,
                    "controversy_a": 0, "controversy_b": 0,
                    "lineup_rating_a": 5.5, "lineup_rating_b": 5.5,
                    "stage": "Fase de grupos", "urgency_a": 5, "urgency_b": 5,
                }
            )

            probs = pred["probs"]
            pred_outcome = int(np.argmax([probs["victoria_A"], probs["empate"], probs["victoria_B"]]))
            actual_outcome = actual_outcome_from_score(hs, aw)

            top_scores = pred["top_scores"].copy()
            top1 = str(top_scores.iloc[0]["marcador"]) if not top_scores.empty else ""
            top3 = set(top_scores.head(3)["marcador"].astype(str).tolist())
            actual_score = f"{hs}-{aw}"

            rows.append({
                "Fecha": match_date.date().isoformat(),
                "Partido": f"{TEAM_FLAGS.get(h,'')} {TEAM_NAME_ES.get(h,h)} vs {TEAM_FLAGS.get(a,'')} {TEAM_NAME_ES.get(a,a)}",
                "Marcador real": actual_score,
                "Predicción 1X2": outcome_text(pred_outcome, TEAM_NAME_ES.get(h,h), TEAM_NAME_ES.get(a,a)),
                "Resultado real": outcome_text(actual_outcome, TEAM_NAME_ES.get(h,h), TEAM_NAME_ES.get(a,a)),
                "Acierto 1X2": pred_outcome == actual_outcome,
                "Top marcador": top1,
                "Acierto marcador top1": top1 == actual_score,
                "Acierto marcador top3": actual_score in top3,
                "Prob. A": probs["victoria_A"] * 100,
                "Prob. Empate": probs["empate"] * 100,
                "Prob. B": probs["victoria_B"] * 100,
            })
        except Exception as e:
            rows.append({
                "Fecha": str(match.get("date", ""))[:10],
                "Partido": f"{match.get('home_team')} vs {match.get('away_team')}",
                "Marcador real": "",
                "Predicción 1X2": "Error",
                "Resultado real": "",
                "Acierto 1X2": False,
                "Top marcador": "",
                "Acierto marcador top1": False,
                "Acierto marcador top3": False,
                "Error": str(e)[:100],
            })

    acc_df = pd.DataFrame(rows)
    if acc_df.empty:
        return acc_df, {}

    summary = {
        "partidos_evaluados": len(acc_df),
        "acierto_1x2_%": float(acc_df["Acierto 1X2"].mean() * 100),
        "acierto_marcador_top1_%": float(acc_df["Acierto marcador top1"].mean() * 100),
        "acierto_marcador_top3_%": float(acc_df["Acierto marcador top3"].mean() * 100),
        "aciertos_1x2": int(acc_df["Acierto 1X2"].sum()),
        "precision_start": precision_start_dt.date().isoformat() if pd.notna(precision_start_dt) else PRECISION_START_DATE,
        "precision_label": PRECISION_START_LABEL,
        "mensaje": ""
    }

    return acc_df, summary


# ============================================================
# 7. GRÁFICAS
# ============================================================

def plot_confusion_matrix(training_result):
    model = training_result["best_model"]
    X_test = training_result["X_test"]
    y_test = training_result["y_test"]

    pred = model.predict(X_test)
    cm = confusion_matrix(y_test, pred, labels=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm)
    ax.set_title(f"Matriz de confusión - {training_result['best_name']}")
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["Gana A", "Empate", "Gana B"])
    ax.set_yticklabels(["Gana A", "Empate", "Gana B"])
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    return fig


def plot_correlation(features_df, feature_cols, max_cols=18):
    """
    Muestra matriz de correlación de las variables más correlacionadas con target.
    """
    df = features_df[feature_cols + ["target"]].dropna()
    corr = df.corr(numeric_only=True)

    important = corr["target"].abs().sort_values(ascending=False).head(max_cols).index.tolist()
    if "target" not in important:
        important = ["target"] + important

    c = df[important].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(c.values, aspect="auto")
    ax.set_xticks(range(len(c.columns)))
    ax.set_yticks(range(len(c.index)))
    ax.set_xticklabels(c.columns, rotation=90)
    ax.set_yticklabels(c.index)
    ax.set_title("Matriz de correlación - variables principales")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def get_feature_importance(training_result):
    """
    Intenta extraer importancia de variables.
    Si no existe, usa correlación absoluta con target.
    """
    model = training_result["best_model"]
    feature_cols = training_result["feature_cols"]
    df = training_result["features_df"]

    real_model = model
    if isinstance(model, Pipeline):
        real_model = model.steps[-1][1]

    if hasattr(real_model, "feature_importances_"):
        imp = pd.DataFrame({
            "variable": feature_cols,
            "peso": real_model.feature_importances_
        })
    elif hasattr(real_model, "coef_"):
        coef = np.abs(real_model.coef_).mean(axis=0)
        imp = pd.DataFrame({
            "variable": feature_cols,
            "peso": coef
        })
    else:
        corr = df[feature_cols + ["target"]].corr(numeric_only=True)["target"].abs()
        imp = corr.drop("target").reset_index()
        imp.columns = ["variable", "peso"]

    imp = imp.sort_values("peso", ascending=False)
    if imp["peso"].sum() > 0:
        imp["peso_%"] = imp["peso"] / imp["peso"].sum() * 100
    else:
        imp["peso_%"] = 0
    return imp


def plot_feature_importance(training_result, top_n=15):
    imp = get_feature_importance(training_result).head(top_n).sort_values("peso_%")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp["variable"], imp["peso_%"])
    ax.set_xlabel("Peso relativo (%)")
    ax.set_title("Variables con más peso en el modelo")
    fig.tight_layout()
    return fig


def plot_team_comparison(prediction):
    r = prediction["future_features"].iloc[0]
    labels = [
        "Goles favor L10",
        "Goles contra L10",
        "Puntos L10",
        "Win rate L10",
        "Portería cero L10",
        "Elo normalizado"
    ]

    elo_a = r["elo_home"]
    elo_b = r["elo_away"]
    elo_min = min(elo_a, elo_b) - 100
    elo_max = max(elo_a, elo_b) + 100
    elo_a_norm = (elo_a - elo_min) / max(1, (elo_max - elo_min)) * 3
    elo_b_norm = (elo_b - elo_min) / max(1, (elo_max - elo_min)) * 3

    team_a_vals = [
        r["home_gf_last10"],
        r["home_ga_last10"],
        r["home_points_last10"],
        r["home_win_rate_last10"] * 3,
        r["home_clean_sheet_last10"] * 3,
        elo_a_norm
    ]
    team_b_vals = [
        r["away_gf_last10"],
        r["away_ga_last10"],
        r["away_points_last10"],
        r["away_win_rate_last10"] * 3,
        r["away_clean_sheet_last10"] * 3,
        elo_b_norm
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, team_a_vals, width, label=prediction["team_a"])
    ax.bar(x + width/2, team_b_vals, width, label=prediction["team_b"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title("Comparación de rendimiento reciente")
    ax.legend()
    fig.tight_layout()
    return fig


# ============================================================
# 8. CONSOLA SIMPLE PARA SPYDER
# ============================================================

def run_console():
    """
    Modo consola para Spyder:
    Ejecuta este archivo en Spyder si no quieres usar Streamlit.
    """
    print("\n=== Predictor Mundial 2026 - modo consola ===\n")

    api_key = os.getenv("API_FOOTBALL_KEY", "").strip()

    print("Cargando histórico...")
    results = download_historical_results(force=False)

    # Agregar resultados manuales
    manual_wc = load_manual_wc_results()
    if not manual_wc.empty:
        results = pd.concat([results, manual_wc], ignore_index=True)
        results = results.drop_duplicates(
            subset=["date", "home_team", "away_team", "home_score", "away_score"],
            keep="last"
        ).sort_values("date").reset_index(drop=True)

    # Si hay API key, intenta actualizar con resultados reales del Mundial.
    if api_key:
        print("Actualizando Mundial 2026 desde API-Football...")
        try:
            fixtures = fetch_worldcup_fixtures(api_key, force=False)
            wc_results = api_fixtures_to_results(fixtures)
            if not wc_results.empty:
                results = pd.concat([results, wc_results], ignore_index=True)
                results = results.drop_duplicates(
                    subset=["date", "home_team", "away_team", "home_score", "away_score"],
                    keep="last"
                ).sort_values("date").reset_index(drop=True)
                print(f"Resultados mundialistas agregados desde API: {len(wc_results)}")
        except Exception as e:
            print(f"No se pudo consultar la API: {e}")

    print("Construyendo variables...")
    features_df, feature_cols, mem = build_features_from_results(results, min_year=max(1980, datetime.today().year - 8))

    print("Entrenando modelos...")
    tr = train_and_compare(features_df, feature_cols)

    print("\nComparación de modelos:")
    print(tr["metrics"].sort_values(["log_loss", "f1_weighted"], ascending=[True, False]))
    print(f"\nMejor modelo: {tr['best_name']}")

    teams = sorted(set(results["home_team"]).union(set(results["away_team"])))
    team_a = input("\nEquipo A: ").strip()
    team_b = input("Equipo B: ").strip()

    if team_a not in teams:
        print(f"Advertencia: '{team_a}' no está escrito igual en la base.")
    if team_b not in teams:
        print(f"Advertencia: '{team_b}' no está escrito igual en la base.")

    print("\nPaís sede del partido en Mundial 2026:")
    print("1. Sin ventaja de anfitrión")
    print("2. United States")
    print("3. Mexico")
    print("4. Canada")
    sede_op = input("Elige opción [1]: ").strip()
    sede_map = {
        "1": "Sin ventaja de anfitrión",
        "2": "United States",
        "3": "Mexico",
        "4": "Canada",
    }
    sede_pais = sede_map.get(sede_op, "Sin ventaja de anfitrión")

    pred = predict_match(tr, mem, results, team_a, team_b, sede_pais=sede_pais)

    p = pred["probs"]
    print(f"\n{team_a} vs {team_b}")
    print(f"Probabilidad victoria {team_a}: {p['victoria_A']*100:.1f}%")
    print(f"Probabilidad empate: {p['empate']*100:.1f}%")
    print(f"Probabilidad victoria {team_b}: {p['victoria_B']*100:.1f}%")
    print(f"Goles esperados {team_a}: {pred['lambda_a']:.2f}")
    print(f"Goles esperados {team_b}: {pred['lambda_b']:.2f}")

    print("\nMarcadores más probables:")
    for _, row in pred["top_scores"].head(7).iterrows():
        print(f"{row['marcador']}: {row['probabilidad_%']:.2f}%")

    # Guardar gráficas
    out_dir = Path("salidas")
    out_dir.mkdir(exist_ok=True)

    fig1 = plot_confusion_matrix(tr)
    fig1.savefig(out_dir / "matriz_confusion.png", dpi=160, bbox_inches="tight")

    fig2 = plot_correlation(tr["features_df"], tr["feature_cols"])
    fig2.savefig(out_dir / "matriz_correlacion.png", dpi=160, bbox_inches="tight")

    fig3 = plot_feature_importance(tr)
    fig3.savefig(out_dir / "pesos_variables.png", dpi=160, bbox_inches="tight")

    fig4 = plot_team_comparison(pred)
    fig4.savefig(out_dir / "comparacion_equipos.png", dpi=160, bbox_inches="tight")

    print("\nGráficas guardadas en carpeta /salidas:")
    print("- matriz_confusion.png")
    print("- matriz_correlacion.png")
    print("- pesos_variables.png")
    print("- comparacion_equipos.png")


# ============================================================
# 9. INTERFAZ STREAMLIT
# ============================================================



# ============================================================
# 10B. ACCESO, MONETIZACIÓN Y MODO PÚBLICO SEGURO
# ============================================================

def streamlit_secrets_file_exists():
    """
    Evita que Streamlit muestre mensajes rojos cuando no existe secrets.toml.
    En local, si no hay archivo de secrets, usamos variables de entorno o valores por defecto.
    """
    try:
        candidates = [
            Path.cwd() / ".streamlit" / "secrets.toml",
            Path.home() / ".streamlit" / "secrets.toml",
        ]
        return any(p.exists() for p in candidates)
    except Exception:
        return False


def get_config_value(key, default=""):
    """
    Lee configuración sin generar errores visuales si no existe secrets.toml.
    Prioridad:
    1. Variables de entorno.
    2. st.secrets solo si existe archivo .streamlit/secrets.toml.
    3. Valor por defecto.
    """
    env_value = os.getenv(key)
    if env_value is not None:
        return env_value

    try:
        if STREAMLIT_OK and streamlit_secrets_file_exists():
            return st.secrets.get(key, default)
    except Exception:
        pass

    return default



def parse_access_codes(raw_codes):
    if raw_codes is None:
        return []
    if isinstance(raw_codes, (list, tuple, set)):
        return [str(x).strip() for x in raw_codes if str(x).strip()]
    return [x.strip() for x in str(raw_codes).replace("\n", ",").split(",") if x.strip()]


def monetization_settings():
    """
    Configuración por secrets/env:
    APP_REQUIRE_ACCESS=true/false
    APP_ACCESS_CODES=CODIGO1,CODIGO2
    PAYMENT_LINK=https://...
    PRICE_USD=4.99
    APP_BRAND_NAME=Nombre de marca
    SUPPORT_CONTACT=@usuario o correo de soporte
    """
    require_raw = str(get_config_value("APP_REQUIRE_ACCESS", "false")).lower().strip()
    require_access = require_raw in ["1", "true", "yes", "si", "sí"]

    access_codes = parse_access_codes(get_config_value("APP_ACCESS_CODES", ""))
    payment_link = str(get_config_value("PAYMENT_LINK", "")).strip()
    price_usd = str(get_config_value("PRICE_USD", "4.99")).strip()
    brand_name = str(get_config_value("APP_BRAND_NAME", "Mundial Predictor Pro")).strip()
    support_contact = str(get_config_value("SUPPORT_CONTACT", "")).strip()

    return {
        "require_access": require_access,
        "access_codes": access_codes,
        "payment_link": payment_link,
        "price_usd": price_usd,
        "brand_name": brand_name,
        "support_contact": support_contact,
    }


def render_public_landing(settings):
    """
    Pantalla de entrada para monetizar sin mostrar datos personales.
    Importante: no oculta identidad ante procesadores de pago, bancos o autoridades.
    """
    brand = settings["brand_name"]
    price = settings["price_usd"]
    payment_link = settings["payment_link"]
    support = settings["support_contact"]

    st.markdown(f"""
    <style>
      .pay-hero {{
        padding: 2rem;
        border-radius: 26px;
        background: linear-gradient(135deg, #0f172a 0%, #111827 55%, #052e2b 100%);
        color: #fff;
        box-shadow: 0 18px 45px rgba(15,23,42,.22);
        margin-bottom: 1rem;
      }}
      .pay-hero h1 {{
        font-size: 44px;
        margin: 0 0 .6rem;
        letter-spacing: -.04em;
      }}
      .pay-hero p {{ color: #cbd5e1; font-size: 16px; max-width: 850px; }}
      .pay-grid {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0,1fr));
        gap: 12px;
        margin-top: 1rem;
      }}
      .pay-card {{
        background: rgba(255,255,255,.08);
        border: 1px solid rgba(255,255,255,.14);
        border-radius: 18px;
        padding: 1rem;
      }}
      .pay-card b {{ color: #fef3c7; }}
      .price-pill {{
        display:inline-block;
        margin-top: 1rem;
        padding: .7rem 1rem;
        border-radius: 999px;
        background: rgba(34,197,94,.14);
        border: 1px solid rgba(34,197,94,.35);
        color: #bbf7d0;
        font-weight: 800;
      }}
      @media (max-width: 900px) {{
        .pay-grid {{ grid-template-columns: 1fr; }}
        .pay-hero h1 {{ font-size: 32px; }}
      }}
    </style>

    <div class="pay-hero">
      <h1>⚽ {brand}</h1>
      <p>
        Plataforma de análisis educativo del Mundial 2026 con modelos de probabilidad,
        Monte Carlo, mapa del torneo, seguimiento de precisión, clima, noticias y métricas de rendimiento.
      </p>
      <div class="price-pill">Acceso digital · USD ${price}</div>
      <div class="pay-grid">
        <div class="pay-card"><b>🗺️ Mapa vivo</b><br>Grupos, favoritos y ruta proyectada.</div>
        <div class="pay-card"><b>🎯 Precisión</b><br>Seguimiento acumulado de aciertos.</div>
        <div class="pay-card"><b>🤖 Modelo ML</b><br>Poisson, Dixon-Coles, regresores y Monte Carlo.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "Esta plataforma es de análisis deportivo y educativo. No es una casa de apuestas, "
        "no garantiza resultados y no promueve apostar."
    )

    col_buy, col_code = st.columns([1, 1])

    with col_buy:
        st.subheader("1. Compra el acceso")
        if payment_link:
            st.link_button("Comprar acceso", payment_link, type="primary", use_container_width=True)
        else:
            st.warning(
                "Aún no configuraste PAYMENT_LINK. Cuando tengas tu link de pago, agrégalo en Secrets."
            )

        if support:
            st.caption(f"Soporte: {support}")

    with col_code:
        st.subheader("2. Ingresa tu código")
        code_input = st.text_input("Código de acceso", type="password", placeholder="Ej: MUNDIAL-XXXX")
        if st.button("Entrar", use_container_width=True):
            valid_codes = settings["access_codes"]
            if code_input.strip() and code_input.strip() in valid_codes:
                st.session_state["access_granted"] = True
                st.success("Acceso aprobado. Cargando plataforma...")
                st.rerun()
            else:
                st.error("Código inválido. Revisa el código recibido después del pago.")

    with st.expander("Aviso de privacidad y pagos"):
        st.write(
            "Puedes mostrar una marca comercial o nombre de proyecto en la página, pero los procesadores de pago "
            "y entidades financieras requieren verificación de identidad. No se debe prometer anonimato financiero. "
            "Cumple impuestos, términos del procesador y leyes aplicables."
        )


def enforce_access_gate():
    """
    Bloquea la app si APP_REQUIRE_ACCESS=true y no hay código válido.
    Si no hay paywall configurado, deja pasar para pruebas.
    """
    settings = monetization_settings()

    if not settings["require_access"]:
        return settings

    if st.session_state.get("access_granted", False):
        return settings

    render_public_landing(settings)
    st.stop()


def streamlit_app():
    st.set_page_config(
        page_title="Predictor Mundial 2026",
        page_icon="⚽",
        layout="wide"
    )

    # Estilo visual simple para que la interfaz se vea más limpia.
    st.markdown("""
    <style>
        .main-title {
            font-size: 42px;
            font-weight: 800;
            line-height: 1.12;
            margin-bottom: 0.2rem;
        }
        .author-box {
            padding: 0.75rem 1rem;
            border-radius: 14px;
            background: #f3f6ff;
            border: 1px solid #dce5ff;
            margin-bottom: 1rem;
        }
        .step-box {
            padding: 0.85rem 1rem;
            border-radius: 14px;
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            margin-top: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">⚽ Predictor Mundial 2026 con Poisson + Machine Learning</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="author-box"><b>Autor:</b> Eder R Perez Herrera<br>'
        '<b>Objetivo:</b> estimar probabilidades, goles esperados y marcadores probables con datos recientes.</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Modelo educativo: estima probabilidades y marcadores probables. "
        "No usa cuotas ni servicios de apuestas."
    )

    access_settings = enforce_access_gate()

    current_year = datetime.today().year
    default_min_year = max(1980, current_year - 8)

    with st.sidebar:
        st.header("Configuración")

        api_key = st.text_input(
            "Clave API_FOOTBALL opcional",
            value=os.getenv("API_FOOTBALL_KEY", ""),
            type="password",
            help="Si no tienes clave API, el modelo usa histórico + datos manuales."
        )

        force_update = st.checkbox("Forzar actualización web", value=False)

        min_year = st.slider(
            "Usar partidos desde el año",
            min_value=1980,
            max_value=max(2026, current_year),
            value=default_min_year,
            help="Por defecto usa los últimos 8 años para que cargue más rápido y sea más relevante."
        )

        st.info(
            "Puedes editar datos/mundial_2026_manual.csv para añadir resultados recientes "
            "si no quieres usar API."
        )

        st.markdown("---")
        st.markdown("**Autor:** Eder R Perez Herrera")

    st.markdown("### Panel de ejecución")
    st.write(
        "El análisis no se ejecuta automáticamente. Presiona el botón cuando quieras cargar datos, "
        "entrenar modelos y generar resultados."
    )

    ejecutar = st.button("🚀 Ejecutar análisis y entrenar modelos", type="primary", use_container_width=True)

    if "training_result" not in st.session_state:
        st.session_state.training_result = None
        st.session_state.results = None
        st.session_state.mem = None
        st.session_state.api_status = None
        st.session_state.min_year = None

    if ejecutar:
        progress_bar = st.progress(0, text="0% · Preparando análisis...")
        status_box = st.empty()

        try:
            status_box.markdown('<div class="step-box">📥 Descargando/cargando histórico de partidos...</div>', unsafe_allow_html=True)
            progress_bar.progress(8, text="8% · Cargando histórico de partidos...")
            results = download_historical_results(force=force_update)

            status_box.markdown('<div class="step-box">📝 Revisando calendario interno gratuito y archivo manual del Mundial 2026...</div>', unsafe_allow_html=True)
            progress_bar.progress(20, text="20% · Revisando calendario interno y datos manuales...")

            # Primero sincroniza calendario interno con resultados incluidos en el código.
            sync_internal_wc_calendar(force_seed_update=True)

            # Luego intenta actualizar resultados desde fuentes públicas gratuitas en un rango de fechas.
            free_update_msg = ""
            try:
                status_box.markdown('<div class="step-box">🌐 Intentando actualización gratuita de marcadores públicos...</div>', unsafe_allow_html=True)
                free_results, free_update_msg = fetch_free_worldcup_results_range(
                    start_date="2026-06-11",
                    end_date=datetime.today().date(),
                    force=force_update
                )
                if not free_results.empty:
                    update_internal_calendar_with_results(free_results, source_label="Fuente pública gratuita")
            except Exception as e:
                free_update_msg = f"Fuentes públicas no disponibles: {str(e)[:100]}"

            internal_wc = load_internal_wc_results()
            manual_wc = load_manual_wc_results()

            if not internal_wc.empty:
                results = pd.concat([results, internal_wc], ignore_index=True)
            if not manual_wc.empty:
                results = pd.concat([results, manual_wc], ignore_index=True)

            api_status = (
                f"Modo económico activo: histórico + calendario interno gratuito "
                f"({len(internal_wc)} resultados del Mundial cargados) + CSV manual. "
                f"Actualización pública: {free_update_msg}"
            )

            if api_key:
                status_box.markdown('<div class="step-box">🌐 Consultando información actualizada en la web...</div>', unsafe_allow_html=True)
                progress_bar.progress(34, text="34% · Consultando API-Football...")
                try:
                    fixtures = fetch_worldcup_fixtures(api_key, force=force_update)
                    wc_results = api_fixtures_to_results(fixtures)
                    if not wc_results.empty:
                        results = pd.concat([results, wc_results], ignore_index=True)
                        api_status = f"API activa: {len(wc_results)} resultados mundialistas agregados."
                    else:
                        api_status = "API activa, pero no encontró partidos finalizados."
                except Exception as e:
                    api_status = f"{api_status} API-Football no disponible o bloqueada por plan: {e}"

            status_box.markdown('<div class="step-box">🧹 Limpiando datos y eliminando duplicados...</div>', unsafe_allow_html=True)
            progress_bar.progress(48, text="48% · Limpiando datos...")
            results = dedupe_results_by_match(results)

            status_box.markdown('<div class="step-box">🧠 Construyendo variables: Elo, forma reciente, goles y descanso...</div>', unsafe_allow_html=True)
            progress_bar.progress(65, text="65% · Construyendo variables del modelo...")
            features_df, feature_cols, mem = build_features_from_results(results, min_year=min_year)

            status_box.markdown('<div class="step-box">🤖 Entrenando modelos de Machine Learning...</div>', unsafe_allow_html=True)
            progress_bar.progress(82, text="82% · Entrenando modelos...")
            training_result = train_and_compare(features_df, feature_cols)

            status_box.markdown('<div class="step-box">📊 Preparando interfaz de resultados...</div>', unsafe_allow_html=True)
            progress_bar.progress(95, text="95% · Preparando resultados...")

            st.session_state.training_result = training_result
            st.session_state.results = results
            st.session_state.mem = mem
            st.session_state.api_status = api_status
            st.session_state.min_year = min_year

            progress_bar.progress(100, text="100% · Análisis completado.")
            status_box.success("✅ Análisis completado. Ya puedes revisar los resultados.")

        except Exception as e:
            progress_bar.progress(100, text="Proceso detenido por error.")
            st.error(f"Ocurrió un error durante la ejecución: {e}")
            return

    if st.session_state.training_result is None:
        st.warning("Presiona **🚀 Ejecutar análisis y entrenar modelos** para cargar la información.")
        st.stop()

    training_result = st.session_state.training_result
    results = st.session_state.results
    mem = st.session_state.mem
    api_status = st.session_state.api_status

    st.success(api_status)

    colA, colB, colC, colD = st.columns(4)
    colA.metric("Partidos cargados", f"{len(results):,}")
    colB.metric("Filas de entrenamiento", f"{len(training_result['features_df']):,}")
    colC.metric("Mejor modelo", training_result["best_name"])
    colD.metric("Desde el año", str(st.session_state.min_year))

    st.divider()

    teams = sorted(set(results["home_team"]).union(set(results["away_team"])))

    st.subheader("Analizar partido específico")

    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        team_a = st.selectbox("Equipo A", teams, index=teams.index("Colombia") if "Colombia" in teams else 0)
    with c2:
        team_b = st.selectbox("Equipo B", teams, index=teams.index("Portugal") if "Portugal" in teams else min(1, len(teams)-1))
    with c3:
        sede_pais = st.selectbox(
            "País sede del partido",
            ["Sin ventaja de anfitrión", "United States", "Mexico", "Canada"],
            help=(
                "Regla estricta: solo United States en United States, Mexico en Mexico "
                "y Canada en Canada reciben ventaja de anfitrión. Los demás equipos no reciben ventaja de casa."
            )
        )

    # Variables avanzadas automáticas desde internet y calendario interno gratuito.
    detected_fixture = find_internal_fixture(team_a, team_b)
    venue_options = list(WORLD_CUP_2026_VENUES.keys())

    if detected_fixture and detected_fixture.get("venue_key") in WORLD_CUP_2026_VENUES:
        default_venue = detected_fixture["venue_key"]
    else:
        detected_country = (detected_fixture or {}).get("sede_pais", sede_pais)
        default_venue = next(
            (v for v in venue_options if WORLD_CUP_2026_VENUES[v]["country"] == detected_country),
            next((v for v in venue_options if WORLD_CUP_2026_VENUES[v]["country"] == sede_pais), venue_options[0])
        )

    if detected_fixture:
        st.caption(
            f"Partido detectado en calendario interno gratuito: ID {detected_fixture.get('fixture_id')} · "
            f"Grupo {detected_fixture.get('group')} · Fecha {detected_fixture.get('date')} · "
            f"Estado {detected_fixture.get('status')}. No necesitas pagar API-Football para este identificador."
        )

    with st.expander("Variables automáticas: clima, noticias, presión, lesiones y alineaciones", expanded=False):
        st.caption(
            "Modo económico: usa calendario interno gratuito, clima desde Open-Meteo y noticias desde GDELT. "
            "API-Football queda como extra opcional si tu plan lo permite; si bloquea 2026, la app sigue funcionando."
        )

        auto_col1, auto_col2, auto_col3 = st.columns([2, 1, 1])
        with auto_col1:
            venue_key = st.selectbox(
                "Sede/estadio para clima y altitud",
                venue_options,
                index=venue_options.index(default_venue) if default_venue in venue_options else 0
            )
        with auto_col2:
            detected_date = pd.to_datetime((detected_fixture or {}).get("date"), errors="coerce")
            default_match_date = detected_date.date() if pd.notna(detected_date) else datetime.today().date()
            match_date = st.date_input("Fecha del partido", value=default_match_date)
        with auto_col3:
            detected_fixture_value = str((detected_fixture or {}).get("fixture_id") or "")
            if "fixture_id_manual_value" not in st.session_state:
                # Se conserva el fixture ID manual que estás usando. Puedes cambiarlo o borrarlo cuando quieras.
                st.session_state.fixture_id_manual_value = DEFAULT_MANUAL_FIXTURE_ID or detected_fixture_value
            fixture_id_manual = st.text_input(
                "Fixture ID opcional",
                key="fixture_id_manual_value",
                help=(
                    "Este campo ya no se sobrescribe automáticamente. "
                    "Si escribiste 236, la app lo mantiene. Si lo borras, usa el calendario interno."
                )
            )

        if detected_fixture and not detected_fixture.get("venue_key"):
            st.info(
                "El calendario interno detectó el partido, pero no tiene sede exacta guardada. "
                "La app usará la sede seleccionada para clima/altitud. Puedes completar venue_key en "
                "datos/calendario_mundial_2026.csv cuando tengas el estadio real."
            )

        usar_auto_web = st.checkbox("Usar actualización automática desde internet", value=True)
        force_context_update = st.checkbox("Forzar actualización de variables externas", value=False)

        if usar_auto_web:
            with st.spinner("Consultando clima, noticias, alineaciones y lesiones..."):
                advanced_context, auto_sources = auto_fetch_advanced_context(
                    api_key=api_key,
                    team_a=team_a,
                    team_b=team_b,
                    sede_pais=sede_pais,
                    venue_key=venue_key,
                    match_date=match_date,
                    fixture_id_manual=fixture_id_manual.strip() or None,
                    force=force_context_update
                )
        else:
            ratings_df = load_player_ratings()
            lineup_a = calculate_team_lineup_rating(team_a, ratings_df)
            lineup_b = calculate_team_lineup_rating(team_b, ratings_df)
            venue = WORLD_CUP_2026_VENUES.get(venue_key, list(WORLD_CUP_2026_VENUES.values())[0])
            venue_info = compute_worldcup_host_advantage(team_a, team_b, sede_pais=sede_pais)

            advanced_context = {
                "temperature_c": 22,
                "humidity_pct": 55,
                "altitude_m": float(venue["altitude_m"]),
                "crowd_support_a": 8 if venue_info["host_adv_home"] == 1 else 4 if venue_info["host_adv_away"] == 1 else 5,
                "crowd_support_b": 8 if venue_info["host_adv_away"] == 1 else 4 if venue_info["host_adv_home"] == 1 else 5,
                "pressure_a": 3 if venue_info["host_adv_home"] == 1 else 7 if venue_info["host_adv_away"] == 1 else 5,
                "pressure_b": 3 if venue_info["host_adv_away"] == 1 else 7 if venue_info["host_adv_home"] == 1 else 5,
                "cohesion_a": 7,
                "cohesion_b": 7,
                "controversy_a": 0,
                "controversy_b": 0,
                "lineup_rating_a": float(lineup_a["rating"]),
                "lineup_rating_b": float(lineup_b["rating"]),
            }
            auto_sources = {
                "weather": {"message": "Automático desactivado"},
                "venue": venue,
                "venue_key": venue_key,
                "news_a": "Automático desactivado",
                "news_b": "Automático desactivado",
                "lineup_a": lineup_a,
                "lineup_b": lineup_b,
                "fixture_info": None,
                "api_lineups_used": False,
            }

        st.markdown("**Contexto competitivo**")
        comp_col1, comp_col2, comp_col3 = st.columns(3)
        with comp_col1:
            stage = st.selectbox(
                "Fase del torneo",
                ["Fase de grupos", "Octavos", "Cuartos", "Semifinal", "Final"],
                index=0
            )
        with comp_col2:
            urgency_a = st.slider(f"Necesidad de ganar {team_a}", 0, 10, 5)
        with comp_col3:
            urgency_b = st.slider(f"Necesidad de ganar {team_b}", 0, 10, 5)

        advanced_context["stage"] = stage
        advanced_context["urgency_a"] = urgency_a
        advanced_context["urgency_b"] = urgency_b

        st.markdown("**Valores usados por el modelo**")
        values_df = pd.DataFrame([
            {"Variable": "Temperatura (°C)", "Valor": advanced_context["temperature_c"]},
            {"Variable": "Humedad (%)", "Valor": advanced_context["humidity_pct"]},
            {"Variable": "Altitud (m)", "Valor": advanced_context["altitude_m"]},
            {"Variable": f"Apoyo público {team_a}", "Valor": advanced_context["crowd_support_a"]},
            {"Variable": f"Apoyo público {team_b}", "Valor": advanced_context["crowd_support_b"]},
            {"Variable": f"Presión contra {team_a}", "Valor": advanced_context["pressure_a"]},
            {"Variable": f"Presión contra {team_b}", "Valor": advanced_context["pressure_b"]},
            {"Variable": f"Cohesión {team_a}", "Valor": round(advanced_context["cohesion_a"], 2)},
            {"Variable": f"Cohesión {team_b}", "Valor": round(advanced_context["cohesion_b"], 2)},
            {"Variable": f"Polémica/conflicto {team_a}", "Valor": round(advanced_context["controversy_a"], 2)},
            {"Variable": f"Polémica/conflicto {team_b}", "Valor": round(advanced_context["controversy_b"], 2)},
            {"Variable": f"Rating alineación {team_a}", "Valor": round(advanced_context["lineup_rating_a"], 2)},
            {"Variable": f"Rating alineación {team_b}", "Valor": round(advanced_context["lineup_rating_b"], 2)},
            {"Variable": "Fase del torneo", "Valor": advanced_context["stage"]},
            {"Variable": f"Necesidad de ganar {team_a}", "Valor": advanced_context["urgency_a"]},
            {"Variable": f"Necesidad de ganar {team_b}", "Valor": advanced_context["urgency_b"]},
        ])
        st.dataframe(values_df, use_container_width=True, hide_index=True)

        st.markdown("**Fuentes y diagnóstico de actualización**")
        diag_df = pd.DataFrame([
            {"Fuente": "Clima", "Detalle": auto_sources.get("weather", {}).get("message", "")},
            {"Fuente": f"Noticias {team_a}", "Detalle": auto_sources.get("news_a", "")},
            {"Fuente": f"Noticias {team_b}", "Detalle": auto_sources.get("news_b", "")},
            {
                "Fuente": f"Alineación {team_a}",
                "Detalle": f'{auto_sources.get("lineup_a", {}).get("status", "")} · jugadores: {auto_sources.get("lineup_a", {}).get("players_used", 0)}'
            },
            {
                "Fuente": f"Alineación {team_b}",
                "Detalle": f'{auto_sources.get("lineup_b", {}).get("status", "")} · jugadores: {auto_sources.get("lineup_b", {}).get("players_used", 0)}'
            },
            {"Fuente": "Modo de actualización", "Detalle": auto_sources.get("mode", "")},
            {"Fuente": "Fixture detectado", "Detalle": str(auto_sources.get("fixture_info"))},
            {"Fuente": "Calendario interno", "Detalle": str(auto_sources.get("internal_fixture"))},
            {"Fuente": "API-Football error/plan", "Detalle": str(auto_sources.get("api_error"))},
            {"Fuente": "Alineaciones API usadas", "Detalle": str(auto_sources.get("api_lineups_used"))},
        ])
        st.dataframe(diag_df, use_container_width=True, hide_index=True)

    prediction = predict_match(training_result, mem, results, team_a, team_b, sede_pais=sede_pais, advanced_context=advanced_context)
    probs = prediction["probs"]

    quality_score, quality_label, quality_details = compute_data_quality_index(
        auto_sources, advanced_context, prediction, mem, team_a, team_b
    )

    st.markdown(f"## {team_a} vs {team_b}")
    q1, q2 = st.columns([1, 3])
    q1.metric("Calidad de datos", f"{quality_score}/100", quality_label)
    with q2.expander("Ver diagnóstico de calidad de datos"):
        st.dataframe(quality_details, use_container_width=True, hide_index=True)
    st.info(
        f"País sede seleccionado: {sede_pais}. "
        "La ventaja de anfitrión solo se activa si el equipo es United States, Mexico o Canada "
        "y juega en su propio país."
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(f"Victoria {team_a}", f"{probs['victoria_A']*100:.1f}%")
    m2.metric("Empate", f"{probs['empate']*100:.1f}%")
    m3.metric(f"Victoria {team_b}", f"{probs['victoria_B']*100:.1f}%")
    m4.metric(f"Goles esperados {team_a}", f"{prediction['lambda_a']:.2f}")
    m5.metric(f"Goles esperados {team_b}", f"{prediction['lambda_b']:.2f}")

    st.caption(
        "Las probabilidades de victoria/empate/derrota se calculan con la matriz final de marcadores "
        "combinando Machine Learning + Poisson + sede/anfitrión + variables avanzadas "
        "como clima, presión mental, cohesión y rating de alineación."
    )

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Marcadores más probables")
        top = prediction["top_scores"][["marcador", "probabilidad_%"]].copy()
        top["probabilidad_%"] = top["probabilidad_%"].map(lambda x: f"{x:.2f}%")
        st.dataframe(top, use_container_width=True, hide_index=True)

    with right:
        st.subheader("Comparación de rendimiento")
        fig = plot_team_comparison(prediction)
        st.pyplot(fig, clear_figure=True)

    st.divider()

    tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Campeón Mundial",
        "Mapa del Mundial",
        "Precisión",
        "Modelos",
        "Matriz de confusión",
        "Matriz de correlación",
        "Peso de variables",
        "Datos del partido"
    ])

    with tab0:
        st.subheader("Probabilidad estimada de ganar el Mundial")

        champion_df = estimate_champion_probabilities(mem, results, WC_2026_TEAMS)
        timeline_df = build_worldcup_performance_timeline(results, WC_2026_TEAMS)

        st.markdown("### Ranking base de fuerza")
        total_prob = champion_df["prob_campeon_%"].sum()
        ctop1, ctop2, ctop3 = st.columns(3)
        ctop1.metric("Equipos evaluados", f"{len(champion_df)}")
        ctop2.metric("Suma de probabilidades", f"{total_prob:.2f}%")
        ctop3.metric("Favorito base", champion_df.iloc[0]["equipo_es"])

        st.caption(
            "Este ranking reparte 100% según fuerza actual. Debajo puedes correr una simulación Monte Carlo "
            "que considera fase de grupos y una ruta aproximada de eliminatorias."
        )

        col_prob_1, col_prob_2 = st.columns([1.05, 1])

        with col_prob_1:
            show_df = champion_df[[
                "ranking", "equipo_es", "prob_campeon_%", "elo",
                "mundial_PJ", "mundial_PTS", "mundial_DG",
                "gf_last10", "ga_last10", "points_last10"
            ]].copy()

            show_df.columns = [
                "Ranking", "Equipo", "Prob. campeón base (%)", "Elo",
                "PJ Mundial", "PTS Mundial", "DG Mundial",
                "GF últimos 10", "GC últimos 10", "Puntos prom. últimos 10"
            ]

            show_df["Prob. campeón base (%)"] = show_df["Prob. campeón base (%)"].map(lambda x: f"{x:.2f}%")
            show_df["Elo"] = show_df["Elo"].map(lambda x: round(float(x), 1))
            show_df["GF últimos 10"] = show_df["GF últimos 10"].map(lambda x: round(float(x), 2))
            show_df["GC últimos 10"] = show_df["GC últimos 10"].map(lambda x: round(float(x), 2))
            show_df["Puntos prom. últimos 10"] = show_df["Puntos prom. últimos 10"].map(lambda x: round(float(x), 2))

            st.dataframe(show_df, use_container_width=True, hide_index=True)

        with col_prob_2:
            top_n = st.slider("Cantidad de equipos en barras", 8, 32, 16)
            fig = plot_champion_probabilities(champion_df, top_n=top_n)
            st.pyplot(fig, clear_figure=True)

        st.divider()
        st.markdown("### Simulación Monte Carlo del torneo completo")

        mc_col1, mc_col2 = st.columns([1, 2])
        with mc_col1:
            n_mc = st.slider("Número de simulaciones", 200, 5000, 1000, step=200)
            run_mc = st.button("🎲 Ejecutar Monte Carlo", use_container_width=True)

        if run_mc:
            with st.spinner("Simulando fase de grupos y eliminatorias..."):
                mc_df = monte_carlo_worldcup(training_result, mem, results, n_simulations=n_mc, seed=RANDOM_STATE)
            st.session_state["mc_df"] = mc_df
            st.session_state["mc_n"] = n_mc

        if "mc_df" in st.session_state:
            mc_df = st.session_state["mc_df"]
            with mc_col2:
                st.metric("Favorito Monte Carlo", mc_df.iloc[0]["equipo_es"], f'{mc_df.iloc[0]["prob_campeon_montecarlo_%"]:.2f}%')
                st.caption(
                    "La simulación usa resultados reales cargados y simula partidos faltantes. "
                    "La llave de eliminatorias es aproximada por siembra de fuerza, no reemplaza el bracket oficial."
                )

            mc_show = mc_df[["ranking_mc", "equipo_es", "prob_campeon_montecarlo_%", "campeon_simulaciones"]].copy()
            mc_show.columns = ["Ranking MC", "Equipo", "Prob. campeón Monte Carlo (%)", "Veces campeón"]
            mc_show["Prob. campeón Monte Carlo (%)"] = mc_show["Prob. campeón Monte Carlo (%)"].map(lambda x: f"{x:.2f}%")
            st.dataframe(mc_show, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Curva de rendimiento durante el Mundial")

        default_teams = champion_df.head(8)["equipo"].tolist()
        selected_curve_teams = st.multiselect(
            "Equipos para mostrar en la curva",
            options=WC_2026_TEAMS,
            default=default_teams,
            format_func=lambda t: TEAM_NAME_ES.get(t, t),
            help="Puedes seleccionar más equipos, pero demasiadas líneas pueden volver la gráfica difícil de leer."
        )

        fig = plot_performance_timeline(timeline_df, selected_curve_teams)
        st.pyplot(fig, clear_figure=True)

        with st.expander("Ver datos de la curva de rendimiento"):
            curve_show = timeline_df.copy()
            curve_show = curve_show[curve_show["equipo"].isin(selected_curve_teams)]
            curve_show = curve_show[[
                "momento", "equipo_es", "rendimiento_0_100", "elo", "PTS", "DG", "PJ"
            ]]
            curve_show.columns = ["Momento", "Equipo", "Rendimiento 0-100", "Elo", "PTS", "DG", "PJ"]
            curve_show["Rendimiento 0-100"] = curve_show["Rendimiento 0-100"].map(lambda x: round(float(x), 2))
            curve_show["Elo"] = curve_show["Elo"].map(lambda x: round(float(x), 1))
            st.dataframe(curve_show, use_container_width=True, hide_index=True)



    with tab1:
        st.subheader("🗺️ Mapa profesional del Mundial")
        st.caption(
            "Visualiza grupos, favoritos, ruta proyectada y estados del torneo. "
            "Pasa el puntero sobre cada equipo o llave para ver información relevante."
        )

        champion_df_for_map = estimate_champion_probabilities(mem, results, WC_2026_TEAMS)
        mc_df_for_map = st.session_state.get("mc_df", None)
        map_html = render_worldcup_map_html(champion_df_for_map, results, mc_df=mc_df_for_map)
        components.html(map_html, height=1450, scrolling=True)

        st.info(
            "La ruta eliminatoria mostrada es una proyección educativa con los clasificados estimados. "
            "Cuando el calendario oficial de eliminatorias esté completamente definido, se puede conectar a una llave oficial."
        )

    with tab2:
        st.subheader("🎯 Precisión acumulada del modelo")
        st.caption(
            "Aquí se mide si la predicción más probable del modelo coincidió con el resultado real "
            "en partidos ya jugados del Mundial. Es una auditoría operativa acumulada, útil para saber cómo va funcionando."
        )

        acc_df, acc_summary = evaluate_worldcup_prediction_accuracy(
            training_result,
            results,
            min_year=st.session_state.min_year,
            start_from_precision_match=True
        )

        st.info(
            f"La precisión acumulada empieza desde **{PRECISION_START_LABEL}** "
            f"({acc_summary.get('precision_start', PRECISION_START_DATE)}). "
            "Los partidos anteriores no se cuentan en este porcentaje."
        )

        if not acc_summary:
            st.warning("Aún no hay partidos mundialistas suficientes para evaluar precisión.")
        elif acc_summary.get("partidos_evaluados", 0) == 0:
            st.warning(acc_summary.get("mensaje", "Aún no hay partidos completados desde el inicio definido."))
        else:
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Partidos evaluados", acc_summary["partidos_evaluados"])
            p2.metric("Acierto ganador/empate", f'{acc_summary["acierto_1x2_%"]:.1f}%')
            p3.metric("Marcador exacto top 1", f'{acc_summary["acierto_marcador_top1_%"]:.1f}%')
            p4.metric("Marcador en top 3", f'{acc_summary["acierto_marcador_top3_%"]:.1f}%')

            st.markdown("### Historial de predicciones evaluadas")
            show_acc = acc_df.copy()
            for col in ["Prob. A", "Prob. Empate", "Prob. B"]:
                if col in show_acc.columns:
                    show_acc[col] = show_acc[col].map(lambda x: f"{float(x):.1f}%" if pd.notna(x) else "")
            for col in ["Acierto 1X2", "Acierto marcador top1", "Acierto marcador top3"]:
                if col in show_acc.columns:
                    show_acc[col] = show_acc[col].map(lambda x: "✅" if bool(x) else "❌")
            st.dataframe(show_acc, use_container_width=True, hide_index=True)

            st.markdown("### Lectura rápida")
            st.write(
                f"El modelo lleva **{acc_summary['aciertos_1x2']} aciertos de {acc_summary['partidos_evaluados']}** "
                f"en ganador/empate/perdedor. El marcador exacto es más difícil, por eso se mide también si cayó dentro del top 3."
            )

    with tab3:
        st.subheader("Comparación de modelos")
        metrics = training_result["metrics"].copy()
        for col in ["accuracy", "f1_weighted", "log_loss", "brier_score"]:
            if col in metrics.columns:
                metrics[col] = metrics[col].map(lambda x: round(float(x), 4) if pd.notna(x) else x)
        st.dataframe(metrics.sort_values(["log_loss", "brier_score", "f1_weighted"], ascending=[True, True, False]), use_container_width=True)

        st.markdown("""
        **Lectura rápida:**
        - `accuracy`: porcentaje de aciertos en ganador/empate/perdedor.
        - `f1_weighted`: equilibrio entre precisión y sensibilidad para las tres clases.
        - `log_loss`: castiga probabilidades mal calibradas. Mientras más bajo, mejor.
        - `brier_score`: mide calidad/calibración de probabilidades. Mientras más bajo, mejor.
        """)

        st.subheader("Modelos de goles esperados")
        goal_metrics = training_result.get("goal_metrics")
        if goal_metrics is not None:
            gm = goal_metrics.copy()
            for col in ["mae_home", "mae_away", "mae_promedio"]:
                if col in gm.columns:
                    gm[col] = gm[col].map(lambda x: round(float(x), 4) if pd.notna(x) else x)
            st.dataframe(gm.sort_values("mae_promedio"), use_container_width=True)
            st.caption(f'Mejor modelo de goles: {training_result.get("goal_best_name")}')

        st.subheader("Backtesting Mundial 2018 / 2022")
        st.caption(
            "Esta prueba reconstruye una base histórica desde 2010. "
            "No depende del slider principal 'Usar partidos desde el año'."
        )
        if st.button("Ejecutar backtesting histórico", use_container_width=True):
            with st.spinner("Ejecutando backtesting..."):
                bt = backtest_world_cups(min_year_backtest=2010)
            st.session_state["backtest_df"] = bt

        if "backtest_df" in st.session_state:
            bt = st.session_state["backtest_df"].copy()
            for col in ["accuracy", "log_loss", "brier_score"]:
                if col in bt.columns:
                    bt[col] = bt[col].map(lambda x: round(float(x), 4) if pd.notna(x) else x)
            st.dataframe(bt, use_container_width=True, hide_index=True)
            st.caption(
                "Backtesting mejorado: ahora evalúa solo partidos exactos con tournament = World Cup "
                "y escoge el mejor modelo con validación temporal antes de probar cada Mundial."
            )

    with tab4:
        fig = plot_confusion_matrix(training_result)
        st.pyplot(fig, clear_figure=True)

    with tab5:
        fig = plot_correlation(training_result["features_df"], training_result["feature_cols"])
        st.pyplot(fig, clear_figure=True)

    with tab6:
        fig = plot_feature_importance(training_result, top_n=18)
        st.pyplot(fig, clear_figure=True)

        imp = get_feature_importance(training_result).head(25)
        imp_show = imp[["variable", "peso_%"]].copy()
        imp_show["peso_%"] = imp_show["peso_%"].map(lambda x: f"{x:.2f}%")
        st.dataframe(imp_show, use_container_width=True, hide_index=True)

    with tab7:
        st.subheader("Variables usadas para este partido")
        ff = prediction["future_features"].T.reset_index()
        ff.columns = ["Variable", "Valor"]
        st.dataframe(ff, use_container_width=True, hide_index=True)

    st.divider()
    st.caption(
        "Consejo: después de cada jornada, actualiza datos/mundial_2026_manual.csv o usa la API "
        "y vuelve a ejecutar. El modelo se alimenta con la información nueva."
    )


# ============================================================
# 10. ENTRADA PRINCIPAL
# ============================================================

if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
    except Exception:
        ctx = None

    if STREAMLIT_OK and ctx is not None:
        # Ejecutado correctamente con:
        # python -m streamlit run app_mundial.py
        streamlit_app()
    else:
        print("\nEsta es una aplicación de Streamlit.")
        print("No la ejecutes con el botón Run de Spyder.")
        print("\nEjecuta esto en CMD o Anaconda Prompt:\n")
        print('cd /d "C:\\Users\\USER\\Desktop\\Betplay\\Partidos del mundial"')
        print("python -m streamlit run app_mundial.py --server.port 8502\n")
