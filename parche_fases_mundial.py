from pathlib import Path
import re

p = Path("app_mundial.py")
text = p.read_text(encoding="utf-8")

backup = Path("app_mundial_backup_antes_fases.py")
backup.write_text(text, encoding="utf-8")

# 1. Normalizar nombres de fases en la llave
replacements = {
    '"fase": "Dieciseisavos"': '"fase": "Dieciseisavos de final"',
    '"fase": "Octavos"': '"fase": "Octavos de final"',
    '"fase": "Cuartos"': '"fase": "Cuartos de final"',
    '"fase": "Semifinal"': '"fase": "Semifinales"',

    '"Octavos": 2': '"Octavos de final": 2',
    '"Cuartos": 3': '"Cuartos de final": 3',
    '"Semifinal": 4': '"Semifinales": 4',

    '("Dieciseisavos", "left", "Dieciseisavos")': '("Dieciseisavos de final", "left", "Dieciseisavos de final")',
    '("Octavos", "left", "Octavos")': '("Octavos de final", "left", "Octavos de final")',
    '("Cuartos", "left", "Cuartos")': '("Cuartos de final", "left", "Cuartos de final")',
    '("Semifinal", "left", "Semifinal")': '("Semifinales", "left", "Semifinales")',
    '("Semifinal", "right", "Semifinal")': '("Semifinales", "right", "Semifinales")',
    '("Cuartos", "right", "Cuartos")': '("Cuartos de final", "right", "Cuartos de final")',
    '("Octavos", "right", "Octavos")': '("Octavos de final", "right", "Octavos de final")',
    '("Dieciseisavos", "right", "Dieciseisavos")': '("Dieciseisavos de final", "right", "Dieciseisavos de final")',

    '["Fase de grupos", "Octavos", "Cuartos", "Semifinal", "Final"]':
    '["Fase de grupos", "Dieciseisavos de final", "Octavos de final", "Cuartos de final", "Semifinales", "Tercer puesto", "Final"]',
}

for old, new in replacements.items():
    text = text.replace(old, new)

# 2. Agregar constantes y función de detección si no existen
helper = '''
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
    1. Número oficial del partido.
    2. Nombre de ronda recibido por API/calendario.
    """

    if match_number is not None:
        try:
            n = int(str(match_number).replace("P", "").strip())
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

    r = str(round_name or "").lower()

    if "group" in r or "grupo" in r:
        return "Fase de grupos"
    if "round of 32" in r or "r32" in r or "dieciseis" in r or "32" in r:
        return "Dieciseisavos de final"
    if "round of 16" in r or "r16" in r or "octavos" in r or "16" in r:
        return "Octavos de final"
    if "quarter" in r or "cuartos" in r:
        return "Cuartos de final"
    if "semi" in r:
        return "Semifinales"
    if "third" in r or "tercer" in r:
        return "Tercer puesto"
    if "final" in r:
        return "Final"

    return "Fase de grupos"
}
'''.replace("}\n", "\n")

if "def detectar_fase_mundial_2026" not in text:
    marker = "def is_final_match_status(status):"
    if marker in text:
        text = text.replace(marker, helper + "\n" + marker)
    else:
        text = helper + "\n" + text

p.write_text(text, encoding="utf-8")
print("Parche aplicado correctamente. Se creó backup:", backup)
