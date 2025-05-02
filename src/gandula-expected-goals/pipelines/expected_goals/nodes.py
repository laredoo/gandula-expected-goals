import pandas as pd


def explode_competitions(competitions: pd.DataFrame) -> pd.DataFrame:
    return competitions.explode("games").rename(columns={"id": "competition_id"})


def format_matches(matches: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [matches.drop(columns=["games"]), matches["games"].apply(pd.Series)], axis=1
    ).rename(columns={"id": "game_id"})

def format_competitions(competitions: pd.DataFrame) -> pd.DataFrame:
    return competitions.drop(columns=["games"])
