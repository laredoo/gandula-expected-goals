from typing import List, Tuple
import ast
import pandas as pd


def explode_competitions(competitions: pd.DataFrame) -> pd.DataFrame:
    return competitions.explode("games").rename(columns={"id": "competition_id"})


def format_matches(matches: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [matches.drop(columns=["games"]), matches["games"].apply(pd.Series)], axis=1
    ).rename(columns={"id": "game_id"})

def format_competitions(competitions: pd.DataFrame) -> pd.DataFrame:

    competitions['games'] = competitions['games'].apply(ast.literal_eval)

    games_exploded = competitions.explode('games', ignore_index=True)

    games = pd.concat([
        games_exploded[['id', 'name']].rename(columns={'id': 'competition_id', 'name': 'competition_name'}),
        pd.json_normalize(games_exploded['games'])
    ], axis=1)

    games = games.rename(columns={'id': 'game_id'})

    competitions = competitions.drop(columns=['games'])

    games_list = games['game_id'].unique().tolist() 

    return competitions, games, games_list

