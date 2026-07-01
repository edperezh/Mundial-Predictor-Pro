from pathlib import Path
import re

files = [Path("app_mundial.py")]

# Si producción usa app_mundial_public.py, también lo actualizamos
if Path("app_mundial_public.py").exists():
    files.append(Path("app_mundial_public.py"))

helper_new = r'''
FASES_MUNDIAL_2026 = [
    "Fase de grupos",
    "Dieciseisavos de final",
    "Octavos de final",
    "Cuartos de final",
    "Semifinales",
    "Tercer puesto",
    "Final",
]

def detectar_fase_mundial_2026(match_number=None, round_name="", fecha=None):
    """
    Detecta la fase oficial del Mundial 2026.
    Prioridad:
    1. Número oficial del partido: P1-P104.
    2. Nombre de ronda: Round of 32, Round of 16, Quarter-final, etc.
    3. Fecha oficial del calendario.
    """

    # 1) Número oficial del partido
    if match_number is not None:
        try:
            n = int(str(match_number).replace("P", "").replace("Match", "").strip())
            if 1 <= n <= 72:
                return "Fase de grupos"
            if 73 <= n <= 88:
                return "Dieciseisavos de final"
            if 89 <= n <= 96:
                return "Octavos de final"
            if 97 <= n <= 100:
                return "Cuartos de final"
            if 101 <= n <= 102:
                return "Semifinales"
            if n == 103:
                return "Tercer puesto"
            if n == 104:
                return "Final"
        except Exception:
            pass

    # 2) Nombre de ronda recibido por API / ESPN / calendario
    r = str(round_name or "").lower()

    if "group" in r or "grupo" in r:
        return "Fase de grupos"
    if "round of 32" in r or "last 32" in r or "r32" in r or "dieciseis" in r or "1/16" in r:
        return "Dieciseisavos de final"
    if "round of 16" in r or "last 16" in r or "r16" in r or "octavos" in r or "1/8" in r:
        return "Octavos de final"
    if "quarter" in r or "cuartos" in r:
        return "Cuartos de final"
    if "semi" in r:
        return "Semifinales"
    if "third" in r or "tercer" in r or "3rd" in r:
        return "Tercer puesto"
    if "final" in r:
        return "Final"

    # 3) Fecha oficial como respaldo
    if fecha is not None:
        try:
            from datetime import datetime, date

            s = str(fecha)[:10]
            d = datetime.fromisoformat(s).date()

            if date(2026, 6, 11) <= d <= date(2026, 6, 27):
                return "Fase de grupos"
            if date(2026, 6, 28) <= d <= date(2026, 7, 3):
                return "Dieciseisavos de final"
            if date(2026, 7, 4) <= d <= date(2026, 7, 7):
                return "Octavos de final"
            if date(2026, 7, 9) <= d <= date(2026, 7, 11):
                return "Cuartos de final"
            if date(2026, 7, 14) <= d <= date(2026, 7, 15):
                return "Semifinales"
            if d == date(2026, 7, 18):
                return "Tercer puesto"
            if d == date(2026, 7, 19):
                return "Final"
        except Exception:
            pass

    return "Fase de grupos"
'''

for p in files:
    text = p.read_text(encoding="utf-8")
    backup = p.with_name(p.stem + "_backup_auto_fase.py")
    backup.write_text(text, encoding="utf-8")

    # Reemplazar función detectar_fase_mundial_2026 si ya existe
    pattern_helper = r'FASES_MUNDIAL_2026\s*=\s*\[[\s\S]*?def is_final_match_status\(status\):'
    if "def detectar_fase_mundial_2026" in text and "def is_final_match_status(status):" in text:
        text = re.sub(pattern_helper, helper_new + "\n\ndef is_final_match_status(status):", text)
    elif "def is_final_match_status(status):" in text:
        text = text.replace("def is_final_match_status(status):", helper_new + "\n\ndef is_final_match_status(status):")

    # Reemplazar selectbox de Fase del torneo para que tenga fase automática
    pattern_stage = r'(?P<indent>\s*)stage\s*=\s*st\.selectbox\(\s*\n\s*"Fase del torneo",\s*\n\s*\[[\s\S]*?\],\s*\n\s*index=0\s*\n\s*\)'

    def repl_stage(m):
        indent = m.group("indent")
        return f'''{indent}fase_opciones = FASES_MUNDIAL_2026
{indent}fixture_info_stage = auto_sources.get("fixture_info") if isinstance(auto_sources, dict) else None
{indent}if not isinstance(fixture_info_stage, dict):
{indent}    fixture_info_stage = {{}}

{indent}round_name_auto = (
{indent}    fixture_info_stage.get("round")
{indent}    or fixture_info_stage.get("stage")
{indent}    or fixture_info_stage.get("league_round")
{indent}    or fixture_info_stage.get("phase")
{indent}    or ""
{indent})

{indent}match_number_auto = (
{indent}    fixture_info_stage.get("match_number")
{indent}    or fixture_info_stage.get("match_no")
{indent}    or fixture_info_stage.get("number")
{indent}    or None
{indent})

{indent}stage_auto = detectar_fase_mundial_2026(
{indent}    match_number=match_number_auto,
{indent}    round_name=round_name_auto,
{indent}    fecha=match_date
{indent})

{indent}stage_index = fase_opciones.index(stage_auto) if stage_auto in fase_opciones else 0

{indent}stage = st.selectbox(
{indent}    "Fase del torneo",
{indent}    fase_opciones,
{indent}    index=stage_index,
{indent}    help=f"Fase sugerida automáticamente: {{stage_auto}}"
{indent})'''

    text, count = re.subn(pattern_stage, repl_stage, text, count=1)

    if count == 0:
        print(f"ADVERTENCIA: No encontré el selector de fase en {p}")
    else:
        print(f"Selector de fase actualizado en {p}")

    p.write_text(text, encoding="utf-8")
    print(f"Backup creado: {backup}")

print("Parche finalizado.")
