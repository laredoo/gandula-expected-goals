from kedro.pipeline import node, Pipeline
from .nodes import explode_competitions, format_matches, format_competitions, read_match_events


def create_pipeline(**kwargs) -> Pipeline:
    """Create a pipeline to return a greeting message."""
    return Pipeline(
        [
            node(
                func=explode_competitions,
                inputs="raw_competitions",
                outputs="raw_matches",
                name="explode_competitions_node",
            ),
            node(
                func=format_matches,
                inputs="raw_matches",
                outputs="intermediate_matches",
                name="format_matches_node",
            ),
            node(
                func=format_competitions,
                inputs="raw_competitions",
                outputs=["intermediate_competitions", "primary_games", "games_list"],
                name="format_competitions_node",
            ),
        ],
    )