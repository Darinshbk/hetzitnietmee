# %%
import streamlit as st
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 

# %%
import requests
import streamlit as st

API_KEY = "703e38c5af704ea2b71e33878e34d5c4"
url = 'https://api.football-data.org/v4/competitions/PL/teams'
headers = {
   'X-Auth-Token': API_KEY
}

@st.cache_data
def fetch_teams():
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        for team in data['teams']:
            print(team['name'])
    else:
        print(f'Fout: {response.status_code}, {response.text}')

fetch_teams()


# %%
import requests
import pandas as pd

API_KEY = "703e38c5af704ea2b71e33878e34d5c4"
url = 'https://api.football-data.org/v4/competitions/PL/teams'
headers = {
   'X-Auth-Token': API_KEY
}

def fetch_teams():
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        
        # Zet de data om naar een Pandas DataFrame
        teams = data.get('teams', [])
        df = pd.DataFrame(teams)
        
        # Toon de eerste 5 rijen van de tabel
        print(df.head())
        
        return df
    else:
        print(f'Fout: {response.status_code}, {response.text}')
        return None

df = fetch_teams()


# %%
import requests
import pandas as pd

API_KEY = "703e38c5af704ea2b71e33878e34d5c4"
url = 'https://api.football-data.org/v4/competitions/PL/teams'
headers = {
   'X-Auth-Token': API_KEY
}

def fetch_columns():
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        
        # Zet de teams om in een DataFrame
        teams = data.get('teams', [])
        df = pd.DataFrame(teams)
        
        # Print alleen de kolomnamen
        print("Kolomnamen in de API-response:")
        print(df.columns.tolist())  # Lijst van kolomnamen tonen
    else:
        print(f'Fout: {response.status_code}, {response.text}')

fetch_columns()


# %%
import requests
import pandas as pd

API_KEY = "703e38c5af704ea2b71e33878e34d5c4"
url = 'https://api.football-data.org/v4/competitions/PL/teams'
headers = {
   'X-Auth-Token': API_KEY
}

def fetch_column_data():
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        
        # Zet de teams om in een DataFrame
        teams = data.get('teams', [])
        df = pd.DataFrame(teams)
        
        # Print de eerste paar waarden per kolom
        print("Eerste waarden per kolom:\n")
        for col in df.columns:
            print(f"🔹 {col}: {df[col].head(3).tolist()}\n")  # Toont de eerste 3 waarden per kolom
    else:
        print(f'Fout: {response.status_code}, {response.text}')

fetch_column_data()


# %%
england = pd.read_csv("England CSV.csv")
print(england.head())

# %%
import pandas as pd

# CSV inladen
england = pd.read_csv("England CSV.csv")

# Alleen seizoenen 2024/25 en 2023/24 behouden
england_filtered = england[england['Season'].isin(["2024/25", "2023/24"])]

# Duplicaten verwijderen (op basis van datum, teams en uitslag)
england_filtered = england_filtered.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam', 'FTH Goals', 'FTA Goals'])

# Nieuwe dataset bekijken
print(england_filtered['Season'].value_counts())  # Check of alleen 2024/25 en 2023/24 overblijven
print(england_filtered.head())  # Bekijk de eerste paar rijen

# Opslaan als nieuwe schone dataset
england_filtered.to_csv("England_Cleaned.csv", index=False)


# %% [markdown]
# Laten we de data opschonen door:
# 
# 1. Missende waarden (NaN) te verwijderen
# 
# 2. Duplicaten te controleren en te verwijderen
# 
# 3. Overbodige informatie te filteren

# %% [markdown]
# 
# 
# 1. Haal teamgegevens op
# 
# 2. Slaat alleen relevante velden op
# 
# 3. Verwijder missende waarden
# 
# 4. Verwijder duplicaten
# 
# 5. Opgeschoonde dataset

# %%



API_KEY = "703e38c5af704ea2b71e33878e34d5c4"
url = "https://api.football-data.org/v4/competitions/PL/teams"

headers = {
    "X-Auth-Token": API_KEY
}

@st.cache_data
def fetch_teams_data():
    
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()

        
        teams_data = []
        for team in data.get('teams', []):  
            teams_data.append({
                "id": team.get("id"),
                "name": team.get("name"),
                "shortName": team.get("shortName"),
                "tla": team.get("tla"),
                "founded": team.get("founded"),
                "venue": team.get("venue"),
            })

        
        df = pd.DataFrame(teams_data)

        
        df.dropna(inplace=True)

        
        df.drop_duplicates(inplace=True)

        
        print(df.head())

    else:
        print(f'Fout: {response.status_code}, {response.text}')

fetch_teams_data()


# %% [markdown]
# Punten per maand te berekenen

# %%



API_KEY = "703e38c5af704ea2b71e33878e34d5c4"
url = "https://api.football-data.org/v4/competitions/PL/matches"

headers = {
    "X-Auth-Token": API_KEY
}


@st.cache_data
def fetch_data(url, headers):
    response = requests.get(url, headers=headers)
    return response


response = fetch_data(url, headers)

if response.status_code == 200:
    data = response.json()

    
    matches_data = []
    for match in data.get('matches', []):  
        if match.get("status") == "FINISHED":  
            matches_data.append({
                "date": match.get("utcDate"),
                "home_team": match["homeTeam"]["name"],
                "away_team": match["awayTeam"]["name"],
                "home_score": match["score"]["fullTime"]["home"],
                "away_score": match["score"]["fullTime"]["away"]
            })

    
    df = pd.DataFrame(matches_data)

   
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")  

    
    def calculate_points(row):
        if row["home_score"] > row["away_score"]:
            return (row["home_team"], 3), (row["away_team"], 0)
        elif row["home_score"] < row["away_score"]:
            return (row["home_team"], 0), (row["away_team"], 3)
        else:
            return (row["home_team"], 1), (row["away_team"], 1)

    points = []
    for _, row in df.iterrows():
        home_points, away_points = calculate_points(row)
        points.append({"team": home_points[0], "points": home_points[1], "month": row["month"]})
        points.append({"team": away_points[0], "points": away_points[1], "month": row["month"]})

   
    points_df = pd.DataFrame(points)
    points_per_month = points_df.groupby(["team", "month"])["points"].sum().reset_index()

    
    print(points_per_month)

else:
    print(f'Fout: {response.status_code}, {response.text}')


# %% [markdown]
# stand per team per maand toont

# %% [markdown]
# 
# 
# 1. Wedstrijddata ophalen van de Premier League.
# 
# 2. Punten per wedstrijd berekenen (3 voor winst, 1 voor gelijkspel, 0 voor verlies).
# 
# 3. Punten per maand groeperen voor elk team.
# 
# 4. Totale punten per team berekenen door alle maanden op te tellen.
# 
# 5. Ranglijst opstellen door de teams te sorteren op het totaal aantal punten.
# 
# 6. Resultaat tonen als de uiteindelijke stand van de Premier League

# %%



API_KEY = "703e38c5af704ea2b71e33878e34d5c4"
url = "https://api.football-data.org/v4/competitions/PL/matches"

headers = {
    "X-Auth-Token": API_KEY
}


@st.cache_data
def fetch_data(url, headers):
    response = requests.get(url, headers=headers)
    return response


response = fetch_data(url, headers)

if response.status_code == 200:
    data = response.json()

    
    matches_data = []
    for match in data.get('matches', []):  
        if match.get("status") == "FINISHED":  
            matches_data.append({
                "date": match.get("utcDate"),
                "home_team": match["homeTeam"]["name"],
                "away_team": match["awayTeam"]["name"],
                "home_score": match["score"]["fullTime"]["home"],
                "away_score": match["score"]["fullTime"]["away"]
            })

    
    df = pd.DataFrame(matches_data)

    
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")  

    
    def calculate_points(row):
        if row["home_score"] > row["away_score"]:
            return (row["home_team"], 3), (row["away_team"], 0)
        elif row["home_score"] < row["away_score"]:
            return (row["home_team"], 0), (row["away_team"], 3)
        else:
            return (row["home_team"], 1), (row["away_team"], 1)

    points = []
    for _, row in df.iterrows():
        home_points, away_points = calculate_points(row)
        points.append({"team": home_points[0], "points": home_points[1], "month": row["month"]})
        points.append({"team": away_points[0], "points": away_points[1], "month": row["month"]})

    
    points_df = pd.DataFrame(points)
    points_per_month = points_df.groupby(["team", "month"])["points"].sum().reset_index()

    
    total_points = points_per_month.groupby("team")["points"].sum().reset_index()

    
    total_points = total_points.sort_values(by="points", ascending=False).reset_index(drop=True)

    
    print("Premier League Stand:")
    print(total_points)

else:
    print(f'Fout: {response.status_code}, {response.text}')



# %%

from datetime import datetime


url = "https://api.football-data.org/v4/competitions/PL/matches"
headers = {"X-Auth-Token": "703e38c5af704ea2b71e33878e34d5c4"}  


@st.cache_data
def fetch_data(url, headers):
    response = requests.get(url, headers=headers)
    return response


response = fetch_data(url, headers)


if response.status_code == 200:
    data = response.json()
else:
    print("Er is iets mis gegaan met het ophalen van de data.")
    exit()


matches = data['matches']


filtered_matches = {
    "Augustus": [],
    "September": [],
    "Oktober": [],
    "November": [],
    "December": [],
    "Januari": [],
    "Februari": []
}

for match in matches:
    match_date = datetime.strptime(match['utcDate'], '%Y-%m-%dT%H:%M:%SZ')
    if match_date.year == 2024:
        if match_date.month == 8:
            filtered_matches["Augustus"].append(match)
        elif match_date.month == 9:
            filtered_matches["September"].append(match)
        elif match_date.month == 10:
            filtered_matches["Oktober"].append(match)
        elif match_date.month == 11:
            filtered_matches["November"].append(match)
        elif match_date.month == 12:
            filtered_matches["December"].append(match)
    elif match_date.year == 2025:
        if match_date.month == 1:
            filtered_matches["Januari"].append(match)
        elif match_date.month == 2:
            filtered_matches["Februari"].append(match)


teams = [
    "Arsenal FC", "Chelsea FC", "Liverpool FC", "Tottenham Hotspur FC", "Manchester City FC", 
    "Manchester United FC", "Leicester City FC", "Everton FC", "West Ham United FC", 
    "Aston Villa FC", "Newcastle United FC", "Brighton & Hove Albion FC", "Brentford FC", 
    "Crystal Palace FC", "Wolverhampton Wanderers FC", "Southampton FC", "AFC Bournemouth", 
    "Nottingham Forest FC", "Fulham FC", "Ipswich Town FC"
]


team_stats = {team: {"Augustus": 0, "September": 0, "Oktober": 0, "November": 0, "December": 0, "Januari": 0, "Februari": 0, "Totaal": 0} for team in teams}


def update_points(matches, maand):
    for match in matches:
        home_team = match['homeTeam']['name']
        away_team = match['awayTeam']['name']
        home_score = match['score']['fullTime']['home']
        away_score = match['score']['fullTime']['away']

        
        if home_score is not None and away_score is not None:
            if home_score > away_score:
                team_stats[home_team][maand] += 3
            elif away_score > home_score:
                team_stats[away_team][maand] += 3
            else:
                team_stats[home_team][maand] += 1
                team_stats[away_team][maand] += 1


for maand, matches in filtered_matches.items():
    update_points(matches, maand)


for team in teams:
    team_stats[team]["Totaal"] = sum(team_stats[team][maand] for maand in filtered_matches.keys())


stand_data = []
for team, stats in team_stats.items():
    stand_data.append({
        "Team": team,
        "Augustus": stats["Augustus"],
        "September": stats["September"],
        "Oktober": stats["Oktober"],
        "November": stats["November"],
        "December": stats["December"],
        "Januari": stats["Januari"],
        "Februari": stats["Februari"],
        "Totaal": stats["Totaal"]
    })

stand_df = pd.DataFrame(stand_data)


stand_df = stand_df.sort_values(by="Totaal", ascending=False).reset_index(drop=True)


print(stand_df)


stand_df.to_csv("premier_league_stand_aug-feb_2024_2025.csv", index=False)


# %%
# Tabs

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([ "Oude Dashbaord", "Verschil", "Premier League Stand", "Eerste Helft vs Tweede Helft", "Doelpuntenanalyse", "Thuis- en Uitprestaties", "Verloren Wedstrijden Thuis en Uit", "Kaart", "Overtredingen & Kaarten per team", "Conclusie", "Bronverwijzing"])

# %%
import plotly.express as px
import pandas as pd
import requests
from datetime import datetime
import streamlit as st


API_KEY = "703e38c5af704ea2b71e33878e34d5c4"
url = "https://api.football-data.org/v4/competitions/PL/matches"
headers = {"X-Auth-Token": API_KEY}

@st.cache_data
def fetch_data():
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Er is iets mis gegaan met het ophalen van de data.")
        return None

data = fetch_data()

matches = data['matches'] if data else []

teams = set()
for match in matches:
    teams.add(match['homeTeam']['name'].replace(" FC", ""))
    teams.add(match['awayTeam']['name'].replace(" FC", ""))

filtered_matches = {maand: [] for maand in ["Augustus", "September", "Oktober", "November", "December", "Januari", "Februari"]}

for match in matches:
    match_date = datetime.strptime(match['utcDate'], '%Y-%m-%dT%H:%M:%SZ')
    maand_namen = ["Augustus", "September", "Oktober", "November", "December", "Januari", "Februari"]
    if 8 <= match_date.month <= 12 and match_date.year == 2024:
        filtered_matches[maand_namen[match_date.month - 8]].append(match)
    elif 1 <= match_date.month <= 2 and match_date.year == 2025:
        filtered_matches[maand_namen[match_date.month + 4]].append(match)

team_stats = {team: {maand: 0 for maand in filtered_matches.keys()} for team in teams}
for team in teams:
    team_stats[team]["Totaal"] = 0

def update_points(matches, maand):
    for match in matches:
        home_team = match['homeTeam']['name'].replace(" FC", "")
        away_team = match['awayTeam']['name'].replace(" FC", "")
        home_score = match['score']['fullTime']['home']
        away_score = match['score']['fullTime']['away']
        if home_score is not None and away_score is not None:
            if home_score > away_score:
                team_stats[home_team][maand] += 3
            elif away_score > home_score:
                team_stats[away_team][maand] += 3
            else:
                team_stats[home_team][maand] += 1
                team_stats[away_team][maand] += 1

def filter_matches_by_type(matches, match_type, selected_team):
    if match_type == "Alle wedstrijden":
        return matches
    elif match_type == "Thuiswedstrijden":
        return [match for match in matches if match['homeTeam']['name'].replace(" FC", "") == selected_team]
    elif match_type == "Uitwedstrijden":
        return [match for match in matches if match['awayTeam']['name'].replace(" FC", "") == selected_team]
    return matches





with tab1:
    # Interactieve elementen boven de plot
    selected_month = st.selectbox("Kies een maand", ["All"] + list(filtered_matches.keys()))
    selected_team = st.selectbox("Kies een team", ["Alle teams"] + list(teams))
    match_type = st.selectbox("Kies type wedstrijd", ["Alle wedstrijden", "Thuiswedstrijden", "Uitwedstrijden"])

    # Unieke keys voor de checkboxen
    show_lowest_5 = st.checkbox("Toon top 5 teams met de minste punten", key="lowest_5")
    show_highest_5 = st.checkbox("Toon top 5 teams met de meeste punten", key="highest_5")

    points_filter = st.slider("Minimum aantal punten", 0, 67, 0)

    filtered_matches_for_team = filter_matches_by_type(matches, match_type, selected_team)

    filtered_matches_by_month = {maand: [] for maand in filtered_matches.keys()}

    for match in filtered_matches_for_team:
        match_date = datetime.strptime(match['utcDate'], '%Y-%m-%dT%H:%M:%SZ')
        maand_namen = ["Augustus", "September", "Oktober", "November", "December", "Januari", "Februari"]
        if 8 <= match_date.month <= 12 and match_date.year == 2024:
            filtered_matches_by_month[maand_namen[match_date.month - 8]].append(match)
        elif 1 <= match_date.month <= 2 and match_date.year == 2025:
            filtered_matches_by_month[maand_namen[match_date.month + 4]].append(match)

    team_stats = {team: {maand: 0 for maand in filtered_matches.keys()} for team in teams}
    for team in teams:
        team_stats[team]["Totaal"] = 0

    for maand, matches in filtered_matches_by_month.items():
        update_points(matches, maand)

    for team in teams:
        team_stats[team]["Totaal"] = sum(team_stats[team][maand] for maand in filtered_matches.keys())

    stand_df = pd.DataFrame.from_dict(team_stats, orient="index").reset_index().rename(columns={"index": "Team"})

    filtered_df = stand_df.copy()

    if selected_month != "All":
        filtered_df = filtered_df[filtered_df[selected_month] > 0]

    if selected_team != "Alle teams":
        filtered_df = filtered_df[filtered_df["Team"] == selected_team]

    filtered_df = filtered_df[filtered_df[selected_month if selected_month != "All" else "Totaal"] >= points_filter]

    if show_lowest_5:
        filtered_df = filtered_df.nsmallest(5, selected_month if selected_month != "All" else "Totaal")
    elif show_highest_5:
        filtered_df = filtered_df.nlargest(5, selected_month if selected_month != "All" else "Totaal")

    if filtered_df.empty:
        st.write("Er zijn geen teams die voldoen aan de geselecteerde criteria.")
    else:
        fig = px.bar(filtered_df, x="Team", y=selected_month if selected_month != "All" else "Totaal",
                     title=f"Premier League Stand: {selected_month} (Totaal)",
                     labels={selected_month if selected_month != "All" else "Totaal": "Punten", "Team": "Team"},
                     color="Team", color_discrete_map={team: px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)] for i, team in enumerate(filtered_df["Team"].unique())})

        max_y_value = filtered_df[selected_month if selected_month != "All" else "Totaal"].max()
        fig.update_layout(
            yaxis=dict(range=[0, max_y_value + 5]),
            width=1000,
            height=600,
            bargap=0.5,
            margin=dict(l=50, r=50, t=50, b=50)
        )

        fig.update_layout(
            legend_title="Teams",
            showlegend=True
        )

        st.plotly_chart(fig)



# %%
with tab2:
    # Uitleg over de verschillen tussen het oude en het nieuwe dashboard
    st.header("Verschillen tussen het oude en nieuwe dashboard")
    
    st.markdown("""
    **In dit nieuwe dashboard zijn de volgende verbeteringen aangebracht:**

    - **Extra data toegevoegd:** 
      In het nieuwe dashboard is extra data toegevoegd, waaronder wedstrijden van de Premier League voor het seizoen 2023/2024. 
    
    - **Diepere analyses:** 
      Er zijn diepgaandere analyses toegevoegd om meer inzicht te geven.

    - **Nieuwe variabelen:** 
      Er zijn nieuwe variabelen toegevoegd, zoals rode kaart, doelpunten, eerste helft etc...

    - **lineair regressiemodel:** 
      Een nieuw lineair regressiemodel is toegevoegd om de relatie tussen verschillende variabelen zoals het aantal gescoorde doelpunten te begrijpen. Dit biedt gebruikers de mogelijkheid om voorspellingen te doen over de toekomstige prestaties van teams.

    - **Een kaart:** 
      Een interactieve kaart wordt toegevoegd om geografische data te visualiseren. Zoals de locaties van de stadions in de Premier League tonen.

    - **Conclusie:** 
      Het bevat een conclusie.

    - **Bronverwijzing:** 
      Het bevat een bronverwijzing.
    """)


# %%
import streamlit as st
import plotly.express as px
import pandas as pd
import requests
from datetime import datetime

# API-configuratie
API_KEY = "703e38c5af704ea2b71e33878e34d5c4"
url = "https://api.football-data.org/v4/competitions/PL/matches"
headers = {"X-Auth-Token": API_KEY}

@st.cache_data
def fetch_data():
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Er is iets mis gegaan met het ophalen van de data.")
        return None

data = fetch_data()
matches = data['matches'] if data else []

# 🔹 Stap 1: Mapping van API-teamnamen naar CSV-teamnamen
team_name_mapping = {
    "Manchester United FC": "Man United",
    "Manchester City FC": "Man City",
    "Brighton & Hove Albion FC": "Brighton",
    "West Ham United FC": "West Ham",
    "Wolverhampton Wanderers FC": "Wolves",
    "Tottenham Hotspur FC": "Tottenham",
    "Newcastle United FC": "Newcastle",
    "Sheffield United FC": "Sheffield Utd",
    "Nottingham Forest FC": "Nott'm Forest",
    "AFC Bournemouth": "Bournemouth",
    "Luton Town FC": "Luton",
    "Burnley FC": "Burnley",
    "Liverpool FC": "Liverpool",
    "Chelsea FC": "Chelsea",
    "Arsenal FC": "Arsenal", 
    "Leicester City FC": "Leicester",
    "Everton FC": "Everton",
    "Aston Villa FC": "Aston Villa",
    "Brentford FC": "Brentford", 
    "Crystal Palace FC": "Crystal Palace",
    "Southampton FC": "Southampton", 
    "Fulham FC": "Fulham",
    "Ipswich Town FC": "Ipswich"
}

# 🔹 Stap 2: Functie om teamnamen te normaliseren
def normalize_team_name(team_name):
    return team_name_mapping.get(team_name, team_name)

# Data inladen van het seizoen 2023/24 uit CSV
england = pd.read_csv("England CSV.csv")
previous_season = england[england["Season"] == "2023/24"]

# 🔹 Stap 3: Teams per seizoen filteren
teams_2023_24 = set(previous_season['HomeTeam']).union(set(previous_season['AwayTeam']))
teams_2024_25 = set()
if data:
    for match in matches:
        teams_2024_25.add(normalize_team_name(match['homeTeam']['name']))
        teams_2024_25.add(normalize_team_name(match['awayTeam']['name']))


with tab3:
    st.title("Premier League Stand Vergelijking (2023/24 vs. 2024/25)")

    with st.sidebar:
        selected_season = st.radio("Kies een seizoen", ["2023/24", "2024/25"])
        selected_team = st.selectbox("Kies een team", ["Alle teams"] + sorted(teams_2023_24 if selected_season == "2023/24" else teams_2024_25))
        show_lowest_5 = st.checkbox("Toon top 5 teams met de minste punten")
        show_highest_5 = st.checkbox("Toon top 5 teams met de meeste punten")
        points_filter = st.slider("Minimum aantal punten", 0, 100, 0)

    # 🔹 Stap 4: Functie om punten te berekenen
    def calculate_points(df, home_col, away_col, home_score_col, away_score_col, valid_teams):
        team_stats = {team: 0 for team in valid_teams}
        for _, row in df.iterrows():
            home_team, away_team = row[home_col], row[away_col]
            home_score, away_score = row[home_score_col], row[away_score_col]
            if home_team in team_stats and away_team in team_stats:
                if home_score > away_score:
                    team_stats[home_team] += 3
                elif away_score > home_score:
                    team_stats[away_team] += 3
                else:
                    team_stats[home_team] += 1
                    team_stats[away_team] += 1
        return team_stats

    # 🔹 Stap 5: Punten berekenen voor het gekozen seizoen
    if selected_season == "2023/24":
        team_points = calculate_points(previous_season, "HomeTeam", "AwayTeam", "FTH Goals", "FTA Goals", teams_2023_24)
    else:
        team_points = {team: 0 for team in teams_2024_25}
        for match in matches:
            home_team = normalize_team_name(match['homeTeam']['name'])
            away_team = normalize_team_name(match['awayTeam']['name'])
            home_score = match['score']['fullTime']['home']
            away_score = match['score']['fullTime']['away']
            if home_team in team_points and away_team in team_points:
                if home_score is not None and away_score is not None:
                    if home_score > away_score:
                        team_points[home_team] += 3
                    elif away_score > home_score:
                        team_points[away_team] += 3
                    else:
                        team_points[home_team] += 1
                        team_points[away_team] += 1

    # 🔹 Stap 6: DataFrame maken en filters toepassen
    stand_df = pd.DataFrame(list(team_points.items()), columns=["Team", "Punten"])
    stand_df = stand_df[stand_df["Punten"] >= points_filter]

    if selected_team != "Alle teams":
        stand_df = stand_df[stand_df["Team"] == selected_team]

    if show_lowest_5:
        stand_df = stand_df.nsmallest(5, "Punten")
    elif show_highest_5:
        stand_df = stand_df.nlargest(5, "Punten")

    # 🔹 Stap 7: Visualisatie
    if stand_df.empty:
        st.write("Geen data beschikbaar voor de geselecteerde filters.")
    else:
        fig = px.bar(
            stand_df, x="Team", y="Punten", 
            title=f"Premier League Stand {selected_season}", 
            labels={"Punten": "Punten", "Team": "Team"},
            color="Team", 
            color_discrete_map={team: px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)] for i, team in enumerate(stand_df["Team"].unique())}
        )
        fig.update_layout(width=1000, height=600, bargap=0.5)
        st.plotly_chart(fig)

# %%
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go

with tab4:
     # Titel
    st.title("Premier League Analyse: Eerste Helft vs Tweede Helft")

     # Data laden
    df = pd.read_csv("England_Cleaned.csv")

     # Filter voor het huidige seizoen (2024/25)
    current_season = "2024/25"
    df = df[df['Season'] == current_season]

    # Bereken doelpunten per helft
    df['Goals_First_Half'] = df['HTH Goals'] + df['HTA Goals']
    df['Goals_Second_Half'] = (df['FTH Goals'] + df['FTA Goals']) - (df['HTH Goals'] + df['HTA Goals'])

    # Scatterplot met regressielijn
    st.header("Regressie-analyse: Eerste Helft vs Tweede Helft")

    # Bereid data voor
    X = df[['Goals_First_Half']]
    y = df['Goals_Second_Half']

    # Train simpele lineaire regressie
    model = LinearRegression()
    model.fit(X, y)

    # Bereken R² en coëfficiënten
    r2 = model.score(X, y)
    coef = model.coef_[0]
    intercept = model.intercept_

    # Dynamische assen instellen met extra marge
    x_max = int(np.ceil(df['Goals_First_Half'].max())) + 1  # Extra marge op de x-as
    y_max = int(np.ceil(df['Goals_Second_Half'].max())) + 1  # Extra marge op de y-as

    # Voorspellingen maken voor de trendlijn
    x_range = np.linspace(0, x_max, 100)
    y_pred = model.predict(x_range.reshape(-1, 1))

    # Maak interactieve plot met Plotly
    fig = go.Figure()

    # Voeg scatterplot toe
    fig.add_trace(go.Scatter(
        x=df['Goals_First_Half'],
        y=df['Goals_Second_Half'],
        mode='markers',
        name='Wedstrijden',
        hovertext=df.apply(lambda row: f"Wedstrijd: {row['HomeTeam']} vs {row['AwayTeam']}<br>"
                                  f"Eerste helft: {row['Goals_First_Half']}<br>"
                                  f"Tweede helft: {row['Goals_Second_Half']}", axis=1),
        marker=dict(size=10, opacity=0.6, color='#636EFA')
))

    # Voeg regressielijn toe
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_pred,
        mode='lines',
        name='Regressielijn',
        line=dict(color='red', width=3, dash='dash')
))

    # Layout aanpassen met extra assenruimte
    fig.update_layout(
        title='Relatie tussen doelpunten in Eerste en Tweede Helft',
        xaxis_title='Doelpunten in Eerste Helft',
        yaxis_title='Doelpunten in Tweede Helft',
        xaxis=dict(range=[-0.5, x_max]),  # Extra marge toegevoegd
        yaxis=dict(range=[-0.5, y_max]),  # Extra marge toegevoegd
        hovermode='closest',
        showlegend=True
)

    # Toon de plot
    st.plotly_chart(fig, use_container_width=True)

    # Uitleg van de resultaten
    st.subheader("Resultaten")
    st.write(f"**Regressievergelijking:** Doelpunten Tweede Helft = {coef:.2f} × Doelpunten Eerste Helft + {intercept:.2f}")
    st.write(f"**R²:** {r2:.2f}")

    if coef > 0:
        st.write("Er is een positieve relatie tussen doelpunten in de eerste en tweede helft.")
    elif coef < 0:
        st.write("Er is een negatieve relatie tussen doelpunten in de eerste en tweede helft.")
    else:
        st.write("Er lijkt geen relatie te zijn tussen doelpunten in de eerste en tweede helft.")

    # Statistieken
    st.subheader("Gemiddeld aantal doelpunten per helft")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Eerste Helft", f"{df['Goals_First_Half'].mean():.2f}")
    with col2:
        st.metric("Tweede Helft", f"{df['Goals_Second_Half'].mean():.2f}")


# %%
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go


with tab5:
    # 📌 Titel
    st.title("Premier League: Doelpuntenanalyse & Lineair Model")

    # 🔹 Stap 1: Mapping van API-teamnamen naar CSV-teamnamen
    team_name_mapping = {
        "Manchester United FC": "Man United",
        "Manchester City FC": "Man City",
        "Brighton & Hove Albion FC": "Brighton",
        "West Ham United FC": "West Ham",
        "Wolverhampton Wanderers FC": "Wolves",
        "Tottenham Hotspur FC": "Tottenham",
        "Newcastle United FC": "Newcastle",
        "Sheffield United FC": "Sheffield Utd",
        "Nottingham Forest FC": "Nott'm Forest",
        "AFC Bournemouth": "Bournemouth",
        "Luton Town FC": "Luton",
        "Burnley FC": "Burnley",
        "Liverpool FC": "Liverpool",
        "Chelsea FC": "Chelsea",
        "Arsenal FC": "Arsenal",
        "Leicester City FC": "Leicester",
        "Everton FC": "Everton",
        "Aston Villa FC": "Aston Villa",
        "Brentford FC": "Brentford",
        "Crystal Palace FC": "Crystal Palace",
        "Southampton FC": "Southampton",
        "Fulham FC": "Fulham",
        "Ipswich Town FC": "Ipswich"
    }

    # 📥 Data laden
    df = pd.read_csv("England_Cleaned.csv")

    # 🔍 Filter voor huidig seizoen (2024/25)
    df = df[df['Season'] == "2024/25"]

    # 🛠️ Pas de mapping toe op HomeTeam en AwayTeam
    df['HomeTeam'] = df['HomeTeam'].str.strip().replace(team_name_mapping)
    df['AwayTeam'] = df['AwayTeam'].str.strip().replace(team_name_mapping)

    # 🎯 Verzamel statistieken voor elk team
    teams = {}

    for _, row in df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        home_scored = row['FTH Goals']
        home_conceded = row['FTA Goals']
        away_scored = row['FTA Goals']
        away_conceded = row['FTH Goals']

        # Werk thuisploeg bij
        if home not in teams:
            teams[home] = {"Goals_Scored": 0, "Goals_Conceded": 0}
        teams[home]["Goals_Scored"] += home_scored
        teams[home]["Goals_Conceded"] += home_conceded

        # Werk uitploeg bij
        if away not in teams:
            teams[away] = {"Goals_Scored": 0, "Goals_Conceded": 0}
        teams[away]["Goals_Scored"] += away_scored
        teams[away]["Goals_Conceded"] += away_conceded

    # Converteer naar DataFrame
    team_stats = pd.DataFrame.from_dict(teams, orient='index').reset_index()
    team_stats.columns = ["Team", "Goals_Scored", "Goals_Conceded"]
    team_stats["Goal_Difference"] = team_stats["Goals_Scored"] - team_stats["Goals_Conceded"]

    # ✅ 🔘 Checkbox om de huidige stand te tonen
    if st.checkbox("📊 Huidige Stand Weergeven"):
        # 📋 Sorteer teams op doelsaldo (zoals een echte ranglijst)
        team_stats_sorted = team_stats.sort_values(by=["Goal_Difference", "Goals_Scored"], ascending=[False, False])

        # 🔢 Voeg een rangnummer toe
        team_stats_sorted.insert(0, "Positie", range(1, len(team_stats_sorted) + 1))

        # 🎯 Toon als tabel in Streamlit
        st.dataframe(
            team_stats_sorted,
            column_config={
                "Positie": "🏆 Pos",
                "Team": "Team",
                "Goals_Scored": "⚽ Gescoorde Goals",
                "Goals_Conceded": "🥅 Doelpunten Tegen",
                "Goal_Difference": "🔄 Doelsaldo"
            },
            hide_index=True
        )

    # 📊 Lineaire Regressie
    X = team_stats[["Goals_Scored"]]
    y = team_stats["Goal_Difference"]

    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)

    # 📈 Scatterplot met Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=team_stats["Goals_Scored"],
        y=team_stats["Goal_Difference"],
        mode='markers',
        text=team_stats["Team"],
        marker=dict(size=10, color="#636EFA"),
        name="Teams"
    ))
    
    # Regressielijn tekenen
    x_range = np.linspace(X.min().values[0], X.max().values[0], 100)  # Correcte schaal
    y_pred = model.predict(x_range.reshape(-1, 1))

    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_pred,
        mode='lines',
        line=dict(color="red", dash="dash"),
        name="Regressielijn"
    ))

    fig.update_layout(
        title="Doelpunten vs. Doelsaldo",
        xaxis_title="Totaal Gescoorde Goals",
        yaxis_title="Doelsaldo",
        hovermode="closest"
    )

    # 📊 Toon resultaten
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"**R²-waarde:** {r2:.2f}")

    st.markdown("""
    🔹 **Interpretatie van R²:**  
     - **Dicht bij 1** → Teams die meer scoren hebben een hoger doelsaldo.  
     - **Dicht bij 0** → Gescoorde goals voorspellen het doelsaldo niet goed.  
     """)



# %%
import streamlit as st
import plotly.express as px
import pandas as pd
import requests



with tab6:
    # API-configuratie
    API_KEY = "703e38c5af704ea2b71e33878e34d5c4"
    url = "https://api.football-data.org/v4/competitions/PL/matches"
    headers = {"X-Auth-Token": API_KEY}

    @st.cache_data
    def fetch_data():
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Er is iets mis gegaan met het ophalen van de data.")
            return None

    data = fetch_data()
    matches = data['matches'] if data else []

    # Teamnaam mapping
    team_name_mapping = {"Manchester United FC": "Man United", "Manchester City FC": "Man City", "Brighton & Hove Albion FC": "Brighton", "West Ham United FC": "West Ham", "Wolverhampton Wanderers FC": "Wolves", "Tottenham Hotspur FC": "Tottenham", "Newcastle United FC": "Newcastle", "Sheffield United FC": "Sheffield Utd", "Nottingham Forest FC": "Nott'm Forest", "AFC Bournemouth": "Bournemouth", "Luton Town FC": "Luton", "Burnley FC": "Burnley", "Liverpool FC": "Liverpool", "Chelsea FC": "Chelsea", "Arsenal FC": "Arsenal", "Leicester City FC": "Leicester", "Everton FC": "Everton", "Aston Villa FC": "Aston Villa", "Brentford FC": "Brentford", "Crystal Palace FC": "Crystal Palace", "Southampton FC": "Southampton", "Fulham FC": "Fulham", "Ipswich Town FC": "Ipswich"}

    def normalize_team_name(team_name):
        return team_name_mapping.get(team_name, team_name)

    england = pd.read_csv("England CSV.csv")
    season_2023 = england[england["Season"] == "2023/24"]

    teams_2023 = set(season_2023['HomeTeam']).union(set(season_2023['AwayTeam']))
    teams_2024 = set()
    if data:
        for match in matches:
            teams_2024.add(normalize_team_name(match['homeTeam']['name']))
            teams_2024.add(normalize_team_name(match['awayTeam']['name']))

    st.title("Vergelijking Thuis- en Uitprestaties 2023 vs. 2024")

    col1, col2 = st.columns([1, 1])
    with col1:
        selected_season = st.radio("Kies een seizoen", ["2023/24", "2024/25"], key="season_radio")

    col3, col4 = st.columns([1, 1])
    with col3:
        show_top_home = st.checkbox("Toon top 5 thuis teams", key="top_home_checkbox")
    with col4:
        show_top_away = st.checkbox("Toon top 5 uit teams", key="top_away_checkbox")

    def calculate_home_away_wins(df, home_col, away_col, home_score_col, away_score_col, valid_teams):
        home_wins = {team: 0 for team in valid_teams}
        away_wins = {team: 0 for team in valid_teams}

        for _, row in df.iterrows():
            home_team, away_team = row[home_col], row[away_col]
            home_score, away_score = row[home_score_col], row[away_score_col]

            if home_team in home_wins and away_team in away_wins:
                if home_score > away_score:
                    home_wins[home_team] += 1
                elif away_score > home_score:
                    away_wins[away_team] += 1

        return home_wins, away_wins

    if selected_season == "2023/24":
        home_wins, away_wins = calculate_home_away_wins(
            season_2023, "HomeTeam", "AwayTeam", "FTH Goals", "FTA Goals", teams_2023
        )
    else:
        home_wins = {team: 0 for team in teams_2024}
        away_wins = {team: 0 for team in teams_2024}

        for match in matches:
            home_team = normalize_team_name(match['homeTeam']['name'])
            away_team = normalize_team_name(match['awayTeam']['name'])
            home_score = match['score']['fullTime']['home']
            away_score = match['score']['fullTime']['away']

            if home_team in home_wins and away_team in away_wins:
                if home_score is not None and away_score is not None:
                    if home_score > away_score:
                        home_wins[home_team] += 1
                    elif away_score > home_score:
                        away_wins[away_team] += 1

    performance_df = pd.DataFrame({
        "Team": list(home_wins.keys()),
        "Thuis Gewonnen": list(home_wins.values()),
        "Uit Gewonnen": list(away_wins.values())
    })

    if show_top_home:
        filtered_df = performance_df.nlargest(5, "Thuis Gewonnen")[["Team", "Thuis Gewonnen"]]
        fig = px.bar(
            filtered_df, x="Team", y="Thuis Gewonnen",
            title=f"Top 5 teams met meeste thuisoverwinningen {selected_season}",
            labels={"Thuis Gewonnen": "Aantal Thuisoverwinningen"},
            color_discrete_sequence=["blue"]
        )
        st.plotly_chart(fig)
    elif show_top_away:
        filtered_df = performance_df.nlargest(5, "Uit Gewonnen")[["Team", "Uit Gewonnen"]]
        fig = px.bar(
            filtered_df, x="Team", y="Uit Gewonnen",
            title=f"Top 5 teams met meeste uitoverwinningen {selected_season}",
            labels={"Uit Gewonnen": "Aantal Uitoverwinningen"},
            color_discrete_sequence=["red"]
        )
        st.plotly_chart(fig)
    else:
        fig = px.bar(
            performance_df, x="Team", y=["Thuis Gewonnen", "Uit Gewonnen"],
            title=f"Vergelijking thuis- en uitprestaties {selected_season}",
            labels={"value": "Aantal Gewonnen Wedstrijden", "variable": "Type"},
            barmode="group",
            color_discrete_map={"Thuis Gewonnen": "blue", "Uit Gewonnen": "red"}
        )
        st.plotly_chart(fig)


# %%
import streamlit as st
import plotly.express as px
import pandas as pd
import requests

with tab7:

    # API-configuratie
    API_KEY = "703e38c5af704ea2b71e33878e34d5c4"
    url = "https://api.football-data.org/v4/competitions/PL/matches"
    headers = {"X-Auth-Token": API_KEY}

    @st.cache_data
    def fetch_data():
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Er is iets mis gegaan met het ophalen van de data.")
            return None

    data = fetch_data()
    matches = data['matches'] if data else []

    # 🔹 Data van 2023 inladen
    england = pd.read_csv("England CSV.csv")
    season_2023 = england[england["Season"] == "2023/24"]

    def calculate_losses(df, home_col, away_col, home_score_col, away_score_col):
        home_losses = {}
        away_losses = {}

        for _, row in df.iterrows():
            home_team, away_team = row[home_col], row[away_col]
            home_score, away_score = row[home_score_col], row[away_score_col]

            # Verlies thuisploeg
            if home_score < away_score:
                home_losses[home_team] = home_losses.get(home_team, 0) + 1
            # Verlies uitploeg
            elif away_score < home_score:
                away_losses[away_team] = away_losses.get(away_team, 0) + 1

        return home_losses, away_losses

    # 🔹 Titel en opties
    st.title("Aantal Verloren Wedstrijden Thuis en Uit")

    # Unieke key voor de radio-button
    selected_season = st.radio(
        "Kies een seizoen", 
        ["2023/24", "2024/25"], 
        key="season_radio_verliezen"
    )

    # 🔹 Bereken verliezen per seizoen
    if selected_season == "2023/24":
        home_losses, away_losses = calculate_losses(
            season_2023, "HomeTeam", "AwayTeam", "FTH Goals", "FTA Goals"
        )
    else:
        home_losses, away_losses = {}, {}
        for match in matches:
            home_team, away_team = match['homeTeam']['name'], match['awayTeam']['name']
            home_score, away_score = match['score']['fullTime']['home'], match['score']['fullTime']['away']
            if home_score is not None and away_score is not None:
                if home_score < away_score:
                    home_losses[home_team] = home_losses.get(home_team, 0) + 1
                elif away_score < home_score:
                    away_losses[away_team] = away_losses.get(away_team, 0) + 1

    # 🔹 DataFrame voor visualisatie
    losses_df = pd.DataFrame({
        "Team": list(home_losses.keys()),
        "Thuis Verloren": list(home_losses.values()),
        "Uit Verloren": [away_losses.get(team, 0) for team in home_losses.keys()]
    })

    # Vervang NaN-waarden door 0 (indien nodig)
    losses_df.fillna(0, inplace=True)

    # 🔹 Checkboxes voor top 5 teams
    col1, col2 = st.columns(2)
    with col1:
        show_top_home_losses = st.checkbox("Toon top 5 teams met meeste thuisverliezen", key="top_home_losses_checkbox")
    with col2:
        show_top_away_losses = st.checkbox("Toon top 5 teams met meeste uitverliezen", key="top_away_losses_checkbox")

    # 🔹 Algemene visualisatie: Barplot voor verliezen
    if not show_top_home_losses and not show_top_away_losses:
        fig = px.bar(
            losses_df,
            x="Team",
            y=["Thuis Verloren", "Uit Verloren"],
            title=f"Aantal Verloren Wedstrijden Thuis en Uit ({selected_season})",
            labels={"value": "Aantal Verloren Wedstrijden", "variable": "Type"},
            barmode="group",
            color_discrete_map={"Thuis Verloren": "blue", "Uit Verloren": "red"}
        )
        fig.update_xaxes(tickangle=90)  # Labels verticaal uitlijnen
        st.plotly_chart(fig)

    # 🔹 Top 5 thuisverliezen
    if show_top_home_losses:
        top_home_losses_df = losses_df.nlargest(5, "Thuis Verloren")[["Team", "Thuis Verloren"]]
        fig_top_home = px.bar(
            top_home_losses_df,
            x="Team",
            y="Thuis Verloren",
            title=f"Top 5 Teams met Meeste Thuisverliezen ({selected_season})",
            labels={"Thuis Verloren": "Aantal Thuisverliezen"},
            color_discrete_sequence=["blue"]
        )
        fig_top_home.update_xaxes(tickangle=90)  # Labels verticaal uitlijnen
        st.plotly_chart(fig_top_home)

    # 🔹 Top 5 uitverliezen
    if show_top_away_losses:
        top_away_losses_df = losses_df.nlargest(5, "Uit Verloren")[["Team", "Uit Verloren"]]
        fig_top_away = px.bar(
            top_away_losses_df,
            x="Team",
            y="Uit Verloren",
            title=f"Top 5 Teams met Meeste Uitverliezen ({selected_season})",
            labels={"Uit Verloren": "Aantal Uitverliezen"},
            color_discrete_sequence=["red"]
        )
        fig_top_away.update_xaxes(tickangle=90)  # Labels verticaal uitlijnen
        st.plotly_chart(fig_top_away)


# %%
import streamlit as st
import folium
from streamlit_folium import folium_static
import requests
import pandas as pd

# API ophalen
API_KEY = "703e38c5af704ea2b71e33878e34d5c4"
url = 'https://api.football-data.org/v4/competitions/PL/teams'
headers = {'X-Auth-Token': API_KEY}

with tab8:
    # Handmatige locatiegegevens gekoppeld aan shortName uit de API
    manual_data = {
        "Liverpool": {"latitude": 53.4308, "longitude": -2.9608},
        "Arsenal": {"latitude": 51.5549, "longitude": -0.1084},
        "Manchester City": {"latitude": 53.4831, "longitude": -2.2004},
        "Aston Villa": {"latitude": 52.5090, "longitude": -1.8847},
        "Tottenham": {"latitude": 51.6043, "longitude": -0.0664},
        "Manchester United": {"latitude": 53.4631, "longitude": -2.2913},
        "West Ham United": {"latitude": 51.5390, "longitude": -0.0158},
        "Brighton": {"latitude": 50.8616, "longitude": -0.0837},
        "Newcastle United": {"latitude": 54.9756, "longitude": -1.6216},
        "Wolverhampton": {"latitude": 52.5903, "longitude": -2.1302},
        "Chelsea": {"latitude": 51.4817, "longitude": -0.1910},
        "Fulham": {"latitude": 51.4746, "longitude": -0.2216},
        "Crystal Palace": {"latitude": 51.3983, "longitude": -0.0856},
        "Bournemouth": {"latitude": 50.7352, "longitude": -1.8381},
        "Brentford": {"latitude": 51.4900, "longitude": -0.2889},
        "Nottingham Forest": {"latitude": 52.9399, "longitude": -1.1321},
        "Everton": {"latitude": 53.4388, "longitude": -2.9663},
        "Leicester City": {"latitude": 52.6204, "longitude": -1.1422},
        "Southampton": {"latitude": 50.9058, "longitude": -1.3911},
        "Ipswich Town": {"latitude": 52.0551, "longitude": 1.1453}
    }

    # Mapping van API shortNames naar gewenste namen
    name_corrections = {
        "Man City": "Manchester City",
        "Man United": "Manchester United",
        "West Ham": "West Ham United",
        "Brighton Hove": "Brighton",
        "Newcastle": "Newcastle United",
        "Nottingham": "Nottingham Forest"
    }

    # API-data ophalen
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        teams = data.get('teams', [])
        df_api = pd.DataFrame(teams)
    else:
        st.error(f'Fout: {response.status_code}, {response.text}')
        df_api = pd.DataFrame()

    # Data combineren: locatie uit handmatige data, de rest uit API
    combined_data = []
    for _, api_entry in df_api.iterrows():
        short_name = api_entry["shortName"]
        corrected_name = name_corrections.get(short_name, short_name)  # Pas naam aan als nodig
        if corrected_name in manual_data:
            combined_data.append({
                "naam": corrected_name,
                "latitude": manual_data[corrected_name]["latitude"],
                "longitude": manual_data[corrected_name]["longitude"],
                "stadion": api_entry["venue"],
                "website": api_entry["website"],
                "logo_url": api_entry["crest"]
            })
        else:
            print(f"⚠️ Club niet gevonden in handmatige data: {corrected_name}")

    # 🎯 Streamlit-app begint hier
    st.title("Premier League Clubs Kaart")

    # Dropdown menu
    club_namen = ["Toon alle teams"] + [club["naam"] for club in combined_data]
    geselecteerde_club = st.selectbox("Selecteer een club", club_namen)

    # 🔍 Inzoomen op de geselecteerde club
    if geselecteerde_club == "Toon alle teams":
        m = folium.Map(location=[53.0, -1.5], zoom_start=6)  # Algemeen overzicht
    else:
        club_info = next((club for club in combined_data if club["naam"] == geselecteerde_club), None)
        if club_info:
            m = folium.Map(location=[club_info["latitude"], club_info["longitude"]], zoom_start=12)
        else:
            m = folium.Map(location=[53.0, -1.5], zoom_start=6)  # Fallback-optie

    # 📍 Markers toevoegen met clublogo als icoon
    for club in combined_data:
        popup_html = f"""
        <div style="text-align: center;">
        <img src="{club['logo_url']}" width="50"><br>
        <b>{club['naam']}</b><br>
        Stadion: {club['stadion']}<br>
        <a href="{club['website']}" target="_blank">Website</a>
        </div>
        """

        icon = folium.CustomIcon(
            icon_image=club['logo_url'],
            icon_size=(35, 35)  # Grootte clublogo als marker
        )

        folium.Marker(
            location=[club["latitude"], club["longitude"]],
            popup=folium.Popup(popup_html, max_width=250),
            icon=icon
        ).add_to(m)

    # 🗺️ Kaart tonen in Streamlit
    folium_static(m)


# %%
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

with tab9:
    df = pd.read_csv("England CSV.csv")

    st.title("Overtredingen & Kaarten per team")

    # Alleen seizoen 2023/24 en 2024/25
    df = df[df["Season"].isin(["2023/24", "2024/25"])]

    # Seizoenselectie
    seizoenen = sorted(df["Season"].unique(), reverse=True)
    geselecteerd_seizoen = st.selectbox("Kies een seizoen", seizoenen)

    # Filter op gekozen seizoen
    df = df[df["Season"] == geselecteerd_seizoen]

    # Teamselectie
    alle_teams = sorted(pd.unique(df["HomeTeam"].tolist() + df["AwayTeam"].tolist()))
    geselecteerd_team = st.selectbox("Kies een team (optioneel)", ["Alle teams"] + alle_teams)

    # Data transformeren per team
    teams_data = []

    for _, row in df.iterrows():
        teams_data.append({
            'Team': row['HomeTeam'],
            'Fouls': row['H Fouls'],
            'Yellow': row['H Yellow'],
            'Red': row['H Red'],
            'Match': 1
        })
        teams_data.append({
            'Team': row['AwayTeam'],
            'Fouls': row['A Fouls'],
            'Yellow': row['A Yellow'],
            'Red': row['A Red'],
            'Match': 1
        })

    team_stats_df = pd.DataFrame(teams_data)

    # Optioneel filteren op geselecteerd team
    if geselecteerd_team != "Alle teams":
        team_stats_df = team_stats_df[team_stats_df["Team"] == geselecteerd_team]

    # Samenvatten per team
    team_summary = team_stats_df.groupby("Team").agg({
        "Fouls": "sum",
        "Yellow": "sum",
        "Red": "sum",
        "Match": "sum"
    }).reset_index()

    team_summary["Fouls per Match"] = team_summary["Fouls"] / team_summary["Match"]

    # Interactieve barplot
    st.subheader("Gemiddeld aantal overtredingen per wedstrijd")

    foul_fig = px.bar(
        team_summary.sort_values("Fouls per Match", ascending=False),
        x="Team",
        y="Fouls per Match",
        color="Fouls per Match",
        color_continuous_scale="Purples",
        title="Gemiddelde overtredingen per team",
        labels={"Fouls per Match": "Gemiddelde overtredingen"},
        hover_data=["Fouls", "Match"]
    )
    foul_fig.update_layout(xaxis_tickangle=-45, coloraxis_showscale=False)
    st.plotly_chart(foul_fig, use_container_width=True)

    # Selectieoptie voor kaartenweergave
    kaart_filter = st.selectbox("Selecteer kaart-weergave", [
        "Alle teams",
        "Top 5 meeste gele kaarten",
        "Top 5 meeste rode kaarten",
        "Top 5 minste gele kaarten",
        "Top 5 minste rode kaarten"
    ])

    # Filter toepassen
    if kaart_filter == "Top 5 meeste gele kaarten":
        kaarten_df = team_summary.sort_values("Yellow", ascending=False).head(5)
    elif kaart_filter == "Top 5 meeste rode kaarten":
        kaarten_df = team_summary.sort_values("Red", ascending=False).head(5)
    elif kaart_filter == "Top 5 minste gele kaarten":
        kaarten_df = team_summary.sort_values("Yellow", ascending=True).head(5)
    elif kaart_filter == "Top 5 minste rode kaarten":
        kaarten_df = team_summary.sort_values("Red", ascending=True).head(5)
    else:
        kaarten_df = team_summary.sort_values("Yellow", ascending=False)

    # Stacked barplot
    st.subheader("Aantal kaarten per team")

    cards_fig = go.Figure(data=[
        go.Bar(name='Gele kaarten', x=kaarten_df["Team"], y=kaarten_df["Yellow"], marker_color='gold'),
        go.Bar(name='Rode kaarten', x=kaarten_df["Team"], y=kaarten_df["Red"], marker_color='crimson')
    ])

    cards_fig.update_layout(
        barmode='stack',
        xaxis_tickangle=-45,
        title="Aantal kaarten per team",
        yaxis_title="Aantal kaarten",
        xaxis_title="Team"
    )

    st.plotly_chart(cards_fig, use_container_width=True)


# %%
with tab10:
    st.title("Conclusie")

    conclusies = [
        "• Man City een mindere seizoen dan vorig jaar",
        "• Nottingham Forest FC tot nu een geweldige seizoen",
        "• Geen verband tussen de doelpunten in de eerste en tweede helft",
        "• Niet helemaal een verband tussen doelpunten en doelsaldo, vanwege doeltegen.",
        "• Liverpool veel sterker thuis dan uit in 2023/24, nu juist gelijk",
        "• Newcastle 11 keer verloren uit, 3 maar thuis in 2023/24"
    ]

    for conclusie in conclusies:
        st.write(conclusie)


# %%
with tab11:
    st.title("Bronverwijzing")

    st.write("De data in deze applicatie is afkomstig van de volgende bronnen:")

    st.markdown("- [Football Data API](https://www.football-data.org/)")
    st.markdown("- [Kaggle: English Premier League and Championship Full Dataset](https://www.kaggle.com/datasets/panaaaaa/english-premier-league-and-championship-full-dataset)")



