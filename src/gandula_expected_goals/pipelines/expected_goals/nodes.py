import ast
import logging
import pandas as pd
import numpy as np
import sys

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


from typing import Any, Callable, Dict, List
from kloppy import pff

logger = logging.getLogger(__name__)
sys.setrecursionlimit(10000)

PATH = "/home/laredo/work/thesis/gandula-expected-goals/data/01_raw"

event_data_prefix = PATH
meta_data_prefix = PATH + "/metadata"
roster_data_prefix = PATH + "/rosters"


def explode_competitions(competitions: pd.DataFrame) -> pd.DataFrame:
    """
    Explode the games column in competitions DataFrame to create separate rows for each game.

    Args:
        competitions: DataFrame containing competition data with a 'games' column containing
                     lists of games and an 'id' column for competition identifier.

    Returns:
        DataFrame with exploded games, where each game becomes a separate row,
        and 'id' column is renamed to 'competition_id'.
    """
    logger.info(f"Exploding competitions DataFrame with {len(competitions)} rows")
    result = competitions.explode("games").rename(columns={"id": "competition_id"})
    logger.info(f"Completed exploding competitions, resulting in {len(result)} rows")
    return result


def format_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Format matches DataFrame by expanding the 'games' column into separate columns.

    Args:
        matches: DataFrame containing match data with a 'games' column that contains
                nested game information as dictionaries or Series.

    Returns:
        DataFrame with the 'games' column expanded into separate columns,
        and 'id' column renamed to 'game_id'.
    """
    logger.info(f"Formatting matches DataFrame with {len(matches)} rows")
    result = pd.concat(
        [matches.drop(columns=["games"]), matches["games"].apply(pd.Series)], axis=1
    ).rename(columns={"id": "game_id"})
    logger.info(
        f"Completed formatting matches, resulting in {len(result)} rows with {len(result.columns)} columns"
    )
    return result


def format_competitions(competitions: pd.DataFrame) -> pd.DataFrame:
    """
    Format competitions DataFrame by processing games data and creating normalized structure.

    This function:
    1. Parses the 'games' column from string to actual list/dict objects
    2. Explodes games to create separate rows for each game
    3. Normalizes nested game data into flat columns
    4. Creates separate datasets for competitions, games, and game list

    Args:
        competitions: DataFrame containing competition data with 'games' column
                     stored as string representations of lists/dicts.

    Returns:
        Tuple containing:
        - competitions: Cleaned competitions DataFrame without games column
        - games: Normalized games DataFrame with competition and game details
        - games_list: List of unique game IDs
    """
    logger.info(f"Formatting competitions DataFrame with {len(competitions)} rows")
    competitions["games"] = competitions["games"].apply(ast.literal_eval)
    logger.debug("Parsed games column from string to objects")

    games_exploded = competitions.explode("games", ignore_index=True)
    logger.debug(f"Exploded games, resulting in {len(games_exploded)} rows")

    games = pd.concat(
        [
            games_exploded[["id", "name"]].rename(
                columns={"id": "competition_id", "name": "competition_name"}
            ),
            pd.json_normalize(games_exploded["games"]),
        ],
        axis=1,
    )

    games = games.rename(columns={"id": "game_id"})

    competitions = competitions.drop(columns=["games"])

    games_list = games["game_id"].unique().tolist()

    logger.info(
        f"Completed formatting competitions. Generated {len(games)} games and {len(games_list)} unique game IDs"
    )

    return competitions, games, games_list


def consolidate_event_partition_into_kedro(
    partitioned_events: Dict[str, Callable[[], Any]],
    partition: str,
    coordinates: str = "statsbomb",
) -> pd.DataFrame:
    """
    Consolidate event data from multiple partitioned sources using kloppy PFF loader.

    This function loads and consolidates event data from partitioned sources by:
    1. Iterating through all partition keys
    2. Loading event data using kloppy's PFF format loader
    3. Combining event data, metadata, and roster data for each partition

    Args:
        partitioned_events: Dictionary mapping partition keys to callable loaders
        event_data_prefix: Path prefix for event data files
        meta_data_prefix: Path prefix for metadata files
        roster_data_prefix: Path prefix for roster data files
        coordinates: Coordinate system to use (default: "statsbomb")

    Returns:
        List of loaded event datasets, one for each partition
    """
    logger.info(f"Consolidating events from {len(partitioned_events)} partitions")
    logger.info(f"Using coordinate system: {coordinates}")
    results = {}

    for i, (partition_key, _) in enumerate(partitioned_events.items(), 1):
        logger.debug(
            f"Processing partition {i}/{len(partitioned_events)}: {partition_key}"
        )

        key = partition_key.removesuffix(".json")

        try:
            event_data = pff.load_event(
                event_data=f"{event_data_prefix}/{partition}/{partition_key}",
                meta_data=f"{meta_data_prefix}/{partition_key}",
                roster_data=f"{roster_data_prefix}/{partition_key}",
                coordinates=coordinates,
            )

            results[key] = event_data

            logger.debug(f"Successfully loaded partition: {partition_key}")

        except Exception as e:
            logger.error(f"Failed to load partition {partition_key}: {str(e)}")
            raise

    logger.info(f"Successfully consolidated {len(results)} event datasets")
    return results


def consolidate_events(
    intermediate_first_partitioned_events: Dict[str, Callable[[], Any]],
    intermediate_second_partitioned_events: Dict[str, Callable[[], Any]],
) -> List[Any]:
    """
    Consolidate event data from two intermediate partitioned sources.

    This function loads pickle files from two partitioned datasets and combines
    them into a single consolidated list of events.

    Args:
        intermediate_first_partitioned_events: First partitioned event data containing
                                              callables that load pickle files
        intermediate_second_partitioned_events: Second partitioned event data containing
                                               callables that load pickle files

    Returns:
        List containing all consolidated event data from both partitioned sources
    """
    logger.info(
        f"Consolidating events from {len(intermediate_first_partitioned_events)} "
        f"first partitioned and {len(intermediate_second_partitioned_events)} "
        f"second partitioned datasets"
    )

    first_event_partition = {
        partition_key: load_func()
        for partition_key, load_func in intermediate_first_partitioned_events.items()
    }

    logger.info("Finished loading first partitioned events")

    ###
    second_event_partition = {
        partition_key: load_func()
        for partition_key, load_func in intermediate_second_partitioned_events.items()
    }

    logger.info("Finished loading second partitioned events")

    logger.warning(
        f"Consolidated {len(first_event_partition) + len(second_event_partition)} pickle files into one folder"
    )
    logger.warning("This may take a while, please be patient...")

    return {**first_event_partition, **second_event_partition}


def get_shot_events(event_data: Dict[str, Callable[[], Any]]) -> List[Any]:
    """
    Get shot events from the specified file.

    Args:
        file_path: Path to the file containing shot events.

    Returns:
        List of shot events.
    """
    logger.info(f"Consolidating all shots from {len(event_data)} matches")

    return {
        f"{record.event_name}_{record.event_id}_{record.timestamp}_{record.coordinates.x}_{record.coordinates.y}": record
        for _, load_func in event_data.items()
        for record in load_func()
        if record.event_type.name == "SHOT"
    }


def calculate_distance_to_goal(shot_x: float, shot_y: float) -> float:
    """
    Calculate the distance from a shot to the goal.

    Args:
        shots_x: X coordinate of the shot.
        shots_y: Y coordinate of the shot.

    Returns:
        Distance from the shot to the goal.
    """
    goal_x, goal_y = 120, 40
    distance = np.sqrt((goal_x - shot_x) ** 2 + (goal_y - shot_y) ** 2)
    return distance


def calculate_angle(shot_x, shot_y):
    """
    Calculate the angle of the goal as seen from the shot position.
    This function computes the angle between the two goal posts as viewed from
    a given shot position on the field using the law of cosines. The goal is
    positioned at x=120 with posts at y=36 and y=44 (8 units apart).
    Args:
        shot_x (float): X-coordinate of the shot position
        shot_y (float): Y-coordinate of the shot position
    Returns:
        float: The angle in radians between the two goal posts as seen from
               the shot position. Returns 0 if the denominator is zero.
    Note:
        - Goal position is fixed at x=120
        - Left goal post at y=36, right goal post at y=44
        - Uses law of cosines: c² = a² + b² - 2ab*cos(C)
        - Cosine values are clipped to [-1, 1] to avoid numerical errors
    """
    left_post_y, right_post_y = 36, 44
    goal_x = 120
    a = np.sqrt((goal_x - shot_x) ** 2 + (left_post_y - shot_y) ** 2)
    b = np.sqrt((goal_x - shot_x) ** 2 + (right_post_y - shot_y) ** 2)
    c = 8

    denominator = 2 * a * b
    if denominator == 0:
        return 0

    cosine_angle = (a**2 + b**2 - c**2) / denominator
    cosine_angle = np.clip(cosine_angle, -1, 1)
    angle = np.arccos(cosine_angle)
    return angle


def extract_features_from_shots(shots: Dict[str, Callable[[], Any]]) -> pd.DataFrame:
    """
    Extract features from shot events.

    Args:
        shots: Dictionary containing shot events.

    Returns:
        DataFrame containing extracted features from the shots.
    """

    return pd.DataFrame(
        [
            {
                "event_id": load_func().event_id,
                "event_name": load_func().event_name,
                "player_name": load_func().player.name,
                "team_id": load_func().team,
                "distance_to_goal": calculate_distance_to_goal(
                    shot_x=load_func().coordinates.x, shot_y=load_func().coordinates.y
                ),
                "angle_to_goal": calculate_angle(
                    shot_x=load_func().coordinates.x, shot_y=load_func().coordinates.y
                ),
                "period": load_func().period.id,
                "label": 1 if load_func().result.is_success else 0,
            }
            for _, load_func in shots.items()
        ]
    )


def train_xg_boost_model(df: pd.DataFrame) -> XGBClassifier:
    """
    Train a logistic regression model on the provided features.

    Args:
        features: DataFrame containing feature columns and target label.
        target: Name of the target column to predict (default: "label").

    Returns:
        Trained LogisticRegression model.
    """
    logger.info("Training logistic regression model")

    features = ["distance_to_goal", "angle_to_goal", "period"]

    X = df[features]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)

    logger.info(f"Generating results DataFrame with {len(X_test)} rows")

    results_df = pd.DataFrame(X_test, columns=features)
    results_df["xG_pred"] = y_pred_proba
    results_df["goal_actual"] = y_test

    logger.info(f"Model trained with AUC score: {auc:.4f}")

    return (model, results_df)
