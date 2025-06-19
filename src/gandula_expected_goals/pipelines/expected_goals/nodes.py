from typing import Any, Callable, Dict, List
import ast
import logging
import pandas as pd
import pickle
from pathlib import Path
from kloppy import pff

logger = logging.getLogger(__name__)

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
