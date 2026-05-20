# NBA Playoffs 2025 - Organized Monte Carlo Simulation

from pathlib import Path
from collections import defaultdict, Counter
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except ModuleNotFoundError:
    HAS_SEABORN = False


# Configuration

BASE_DIR = Path(__file__).resolve().parent
EXCEL_FILE = BASE_DIR / "nba_2024_25_head_to_head.xlsx"
SHEET_NAME = "Detail"
N_SIMULATIONS = 10000
RANDOM_SEED = 42

OUTPUT_FOLDER = BASE_DIR / "resultados_montecarlo_nba_2025"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

np.random.seed(RANDOM_SEED)

if HAS_SEABORN:
    sns.set_theme(style="whitegrid")


# Actual 2025 NBA Playoff Bracket

BRACKET = {
    "East": [
        ("Cleveland Cavaliers", "Miami Heat"),
        ("Boston Celtics", "Orlando Magic"),
        ("New York Knicks", "Detroit Pistons"),
        ("Indiana Pacers", "Milwaukee Bucks")
    ],
    "West": [
        ("Oklahoma City Thunder", "Memphis Grizzlies"),
        ("Houston Rockets", "Golden State Warriors"),
        ("Los Angeles Lakers", "Minnesota Timberwolves"),
        ("Denver Nuggets", "Los Angeles Clippers")
    ]
}

PLAYOFF_TEAMS = sorted({
    team
    for conference in BRACKET.values()
    for matchup in conference
    for team in matchup
})

TEAM_ALIASES = {
    "OKC": "Oklahoma City Thunder",
    "Oklahoma City": "Oklahoma City Thunder",
    "GSW": "Golden State Warriors",
    "Golden State": "Golden State Warriors",
    "LA Clippers": "Los Angeles Clippers",
    "L.A. Clippers": "Los Angeles Clippers",
    "LA Lakers": "Los Angeles Lakers",
    "L.A. Lakers": "Los Angeles Lakers",
}


# Data Cleaning and Loading

def normalize_team_name(name):
    if pd.isna(name):
        return None

    name = str(name).strip()
    name = re.sub(r"\s+", " ", name)

    return TEAM_ALIASES.get(name, name)


def load_data():
    if not EXCEL_FILE.exists():
        raise FileNotFoundError(
            f"No encontré el Excel aquí:\n{EXCEL_FILE}\n\n"
            "Pon el archivo nba_2024_25_head_to_head.xlsx en la misma carpeta del .py"
        )

    df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)

    required_cols = [
        "Team",
        "Opponent",
        "Wins",
        "Losses",
        "Total Team Wins",
        "Total Team Losses"
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(
            f"Faltan columnas: {missing}\n"
            f"Columnas encontradas: {list(df.columns)}"
        )

    df = df[required_cols].copy()

    df["Team"] = df["Team"].apply(normalize_team_name)
    df["Opponent"] = df["Opponent"].apply(normalize_team_name)

    for col in ["Wins", "Losses", "Total Team Wins", "Total Team Losses"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    missing_teams = [team for team in PLAYOFF_TEAMS if team not in set(df["Team"])]

    if missing_teams:
        raise ValueError(
            "Estos equipos del bracket no aparecen en el Excel:\n"
            + "\n".join(missing_teams)
        )

    print("========================================")
    print("ARCHIVO LEÍDO CORRECTAMENTE")
    print("========================================")
    print(f"Archivo: {EXCEL_FILE.name}")
    print(f"Filas válidas: {len(df)}")
    print(f"Equipos encontrados: {df['Team'].nunique()}")

    return df


df = load_data()


# Probability Calculations

records = (
    df[["Team", "Total Team Wins", "Total Team Losses"]]
    .drop_duplicates("Team")
    .set_index("Team")
)

team_win_pct = (
    records["Total Team Wins"] /
    (records["Total Team Wins"] + records["Total Team Losses"])
).to_dict()


def get_h2h(team_a, team_b):
    row = df[(df["Team"] == team_a) & (df["Opponent"] == team_b)]

    if row.empty:
        raise ValueError(f"No hay head-to-head para {team_a} vs {team_b}")

    wins = int(row.iloc[0]["Wins"])
    losses = int(row.iloc[0]["Losses"])

    return wins, losses


def game_probability(team_a, team_b, alpha=4):
    h2h_wins, h2h_losses = get_h2h(team_a, team_b)
    h2h_games = h2h_wins + h2h_losses

    pct_a = team_win_pct[team_a]
    pct_b = team_win_pct[team_b]

    strength_prior = pct_a / (pct_a + pct_b)

    probability = (h2h_wins + alpha * strength_prior) / (h2h_games + alpha)

    return float(np.clip(probability, 0.05, 0.95))


# Best-of-Seven Series Simulation

def simulate_series(team_a, team_b):
    p_a = game_probability(team_a, team_b)

    wins_a = 0
    wins_b = 0

    while wins_a < 4 and wins_b < 4:
        if np.random.random() < p_a:
            wins_a += 1
        else:
            wins_b += 1

    if wins_a == 4:
        winner = team_a
        result = f"4-{wins_b}"
    else:
        winner = team_b
        result = f"4-{wins_a}"

    total_games = 4 + int(result.split("-")[1])

    return winner, result, total_games


# Monte Carlo Simulation

advance_counts = defaultdict(lambda: defaultdict(int))
champion_counts = Counter()
conference_champion_counts = Counter()

series_result_counts = defaultdict(Counter)
bracket_paths = Counter()

print("\nEjecutando simulación Monte Carlo...")
print(f"Simulaciones: {N_SIMULATIONS}")

for simulation in range(N_SIMULATIONS):

    current_path = []
    conference_winners = {}

    for conference in ["East", "West"]:

# First round
        r1_winners = []

        for team_a, team_b in BRACKET[conference]:
            winner, result, games = simulate_series(team_a, team_b)

            r1_winners.append(winner)
            advance_counts[winner]["Conference Semifinals"] += 1

            key = (conference, "First Round", team_a, team_b)
            series_result_counts[key][(winner, result, games)] += 1

            current_path.append((
                conference,
                "First Round",
                team_a,
                team_b,
                winner,
                result,
                games
            ))

# Conference semifinals
# Fixed bracket without reseeding
        semifinal_matchups = [
            (r1_winners[0], r1_winners[3]),
            (r1_winners[1], r1_winners[2])
        ]

        sf_winners = []

        for team_a, team_b in semifinal_matchups:
            winner, result, games = simulate_series(team_a, team_b)

            sf_winners.append(winner)
            advance_counts[winner]["Conference Finals"] += 1

            key = (conference, "Conference Semifinals", team_a, team_b)
            series_result_counts[key][(winner, result, games)] += 1

            current_path.append((
                conference,
                "Conference Semifinals",
                team_a,
                team_b,
                winner,
                result,
                games
            ))

# Conference finals
        team_a, team_b = sf_winners

        winner, result, games = simulate_series(team_a, team_b)

        advance_counts[winner]["NBA Finals"] += 1
        conference_champion_counts[winner] += 1
        conference_winners[conference] = winner

        key = (conference, "Conference Finals", team_a, team_b)
        series_result_counts[key][(winner, result, games)] += 1

        current_path.append((
            conference,
            "Conference Finals",
            team_a,
            team_b,
            winner,
            result,
            games
        ))

# NBA Finals
    east_champion = conference_winners["East"]
    west_champion = conference_winners["West"]

    champion, result, games = simulate_series(east_champion, west_champion)

    champion_counts[champion] += 1
    advance_counts[champion]["Champion"] += 1

    key = ("NBA", "NBA Finals", east_champion, west_champion)
    series_result_counts[key][(champion, result, games)] += 1

    current_path.append((
        "NBA",
        "NBA Finals",
        east_champion,
        west_champion,
        champion,
        result,
        games
    ))

    bracket_paths[tuple(current_path)] += 1


print("Simulación terminada.")


# Overall Probability Table

advance_table = pd.DataFrame(index=PLAYOFF_TEAMS)

advance_table["Conference Semifinals"] = [
    advance_counts[team]["Conference Semifinals"] / N_SIMULATIONS
    for team in PLAYOFF_TEAMS
]

advance_table["Conference Finals"] = [
    advance_counts[team]["Conference Finals"] / N_SIMULATIONS
    for team in PLAYOFF_TEAMS
]

advance_table["NBA Finals"] = [
    advance_counts[team]["NBA Finals"] / N_SIMULATIONS
    for team in PLAYOFF_TEAMS
]

advance_table["Conference Champion"] = [
    conference_champion_counts[team] / N_SIMULATIONS
    for team in PLAYOFF_TEAMS
]

advance_table["Champion Probability"] = [
    champion_counts[team] / N_SIMULATIONS
    for team in PLAYOFF_TEAMS
]

advance_table = advance_table.sort_values(
    "Champion Probability",
    ascending=False
)

advance_table.to_csv(OUTPUT_FOLDER / "probabilidades_avance.csv")


# Most Common Full Bracket

most_common_path, path_count = bracket_paths.most_common(1)[0]
path_probability = path_count / N_SIMULATIONS

organized_rows = []

for item in most_common_path:
    conference, round_name, team_a, team_b, winner, result, games = item
    loser = team_b if winner == team_a else team_a

    organized_rows.append({
        "Conference": conference,
        "Round": round_name,
        "Team A": team_a,
        "Team B": team_b,
        "Winner": winner,
        "Loser": loser,
        "Series Result": result,
        "Games": games
    })

organized_bracket = pd.DataFrame(organized_rows)
organized_bracket.to_csv(
    OUTPUT_FOLDER / "bracket_mas_frecuente.csv",
    index=False
)


# Print Organized Results

print("\n\n========================================")
print("BRACKET COMPLETO MÁS FRECUENTE")
print("========================================")
print(f"Veces que ocurrió: {path_count} de {N_SIMULATIONS}")
print(f"Probabilidad del camino completo: {path_probability:.2%}")
print("Nota: esta probabilidad es del bracket completo exacto, no solo del campeón.")

for conference in ["East", "West"]:

    conf_name = "ESTE" if conference == "East" else "OESTE"

    print(f"\n\nCONFERENCIA {conf_name}")
    print("=" * 60)

    for round_name in [
        "First Round",
        "Conference Semifinals",
        "Conference Finals"
    ]:

        round_spanish = {
            "First Round": "PRIMERA RONDA",
            "Conference Semifinals": "SEMIFINALES DE CONFERENCIA",
            "Conference Finals": "FINAL DE CONFERENCIA"
        }[round_name]

        print(f"\n{round_spanish}")
        print("-" * 60)

        rows = organized_bracket[
            (organized_bracket["Conference"] == conference)
            & (organized_bracket["Round"] == round_name)
        ]

        for _, row in rows.iterrows():
            print(f"{row['Team A']} vs {row['Team B']}")
            print(f"Ganador: {row['Winner']}")
            print(f"Resultado: {row['Winner']} {row['Series Result']}")
            print(f"Número de juegos: {row['Games']}")
            print()


print("\n\nFINALES NBA")
print("=" * 60)

nba_final = organized_bracket[
    organized_bracket["Round"] == "NBA Finals"
].iloc[0]

final_champion = nba_final["Winner"]
final_result = nba_final["Series Result"]
final_games = nba_final["Games"]
final_champion_probability = champion_counts[final_champion] / N_SIMULATIONS

print(f"{nba_final['Team A']} vs {nba_final['Team B']}")
print(f"Campeón NBA: {final_champion}")
print(f"Resultado: {final_champion} {final_result}")
print(f"Número de juegos: {final_games}")
print(f"Probabilidad de campeonato del equipo: {final_champion_probability:.2%}")
print(f"Probabilidad del bracket completo exacto: {path_probability:.2%}")


print("\n\nTOP 5 PROBABILIDAD GLOBAL DE CAMPEONATO")
print("=" * 60)

for team, row in advance_table.head(5).iterrows():
    print(f"{team}: {row['Champion Probability']:.2%}")


# Save Text Summary

report_path = OUTPUT_FOLDER / "resultado_organizado.txt"

with open(report_path, "w", encoding="utf-8") as f:
    f.write("NBA PLAYOFFS 2025 - RESULTADO ORGANIZADO\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Simulaciones: {N_SIMULATIONS}\n")
    f.write(f"Veces que ocurrió el bracket completo más frecuente: {path_count}\n")
    f.write(f"Probabilidad del bracket completo: {path_probability:.2%}\n\n")

    for conference in ["East", "West"]:
        conf_name = "ESTE" if conference == "East" else "OESTE"
        f.write(f"\nCONFERENCIA {conf_name}\n")
        f.write("=" * 60 + "\n")

        for round_name in [
            "First Round",
            "Conference Semifinals",
            "Conference Finals"
        ]:
            f.write(f"\n{round_name}\n")
            f.write("-" * 60 + "\n")

            rows = organized_bracket[
                (organized_bracket["Conference"] == conference)
                & (organized_bracket["Round"] == round_name)
            ]

            for _, row in rows.iterrows():
                f.write(f"{row['Team A']} vs {row['Team B']}\n")
                f.write(f"Ganador: {row['Winner']}\n")
                f.write(f"Resultado: {row['Winner']} {row['Series Result']}\n")
                f.write(f"Número de juegos: {row['Games']}\n\n")

    f.write("\nFINALES NBA\n")
    f.write("=" * 60 + "\n")
    f.write(f"{nba_final['Team A']} vs {nba_final['Team B']}\n")
    f.write(f"Campeón NBA: {final_champion}\n")
    f.write(f"Resultado: {final_champion} {final_result}\n")
    f.write(f"Número de juegos: {final_games}\n")
    f.write(f"Probabilidad de campeonato del equipo: {final_champion_probability:.2%}\n")
    f.write(f"Probabilidad del bracket completo exacto: {path_probability:.2%}\n")

    f.write("\n\nNOTA IMPORTANTE\n")
    f.write(
        "Esto es una simulación predictiva basada en datos históricos. "
        "No es una garantía del resultado real.\n"
    )


# Charts

def save_barh(series, title, xlabel, filename):
    plt.figure(figsize=(11, 8))
    series.sort_values().plot(kind="barh")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Equipo")
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / filename, dpi=300)
    plt.close()


save_barh(
    advance_table["Champion Probability"],
    "Probabilidad Global de Campeonato NBA 2025",
    "Probabilidad",
    "probabilidad_campeonato.png"
)

save_barh(
    advance_table["NBA Finals"],
    "Probabilidad de Llegar a Finales NBA",
    "Probabilidad",
    "probabilidad_finales_nba.png"
)

save_barh(
    advance_table["Conference Champion"],
    "Probabilidad de Ganar Conferencia",
    "Probabilidad",
    "probabilidad_conferencia.png"
)

plt.figure(figsize=(14, 8))
advance_table[
    [
        "Conference Semifinals",
        "Conference Finals",
        "NBA Finals",
        "Champion Probability"
    ]
].plot(kind="bar", figsize=(14, 8))
plt.title("Probabilidad de Avance por Ronda")
plt.ylabel("Probabilidad")
plt.xlabel("Equipo")
plt.xticks(rotation=75, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "avance_por_ronda.png", dpi=300)
plt.close()

prob_matrix = pd.DataFrame(
    index=PLAYOFF_TEAMS,
    columns=PLAYOFF_TEAMS,
    dtype=float
)

for team_a in PLAYOFF_TEAMS:
    for team_b in PLAYOFF_TEAMS:
        if team_a == team_b:
            prob_matrix.loc[team_a, team_b] = np.nan
        else:
            prob_matrix.loc[team_a, team_b] = game_probability(team_a, team_b)

plt.figure(figsize=(15, 11))

if HAS_SEABORN:
    sns.heatmap(
        prob_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0.5,
        linewidths=0.5
    )
else:
    plt.imshow(prob_matrix.astype(float), aspect="auto")
    plt.colorbar(label="Probabilidad")
    plt.xticks(range(len(PLAYOFF_TEAMS)), PLAYOFF_TEAMS, rotation=90)
    plt.yticks(range(len(PLAYOFF_TEAMS)), PLAYOFF_TEAMS)

plt.title("Heatmap de Probabilidades de Victoria por Juego")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "heatmap_probabilidades_juego.png", dpi=300)
plt.close()


# Final Output

print("\n\n========================================")
print("ARCHIVOS GENERADOS")
print("========================================")
print(f"Carpeta: {OUTPUT_FOLDER}")
print("- probabilidades_avance.csv")
print("- bracket_mas_frecuente.csv")
print("- resultado_organizado.txt")
print("- probabilidad_campeonato.png")
print("- probabilidad_finales_nba.png")
print("- probabilidad_conferencia.png")
print("- avance_por_ronda.png")
print("- heatmap_probabilidades_juego.png")

print("\nPROCESO COMPLETADO CORRECTAMENTE.")
print("Esto es una simulación predictiva, no una garantía del resultado real.")
