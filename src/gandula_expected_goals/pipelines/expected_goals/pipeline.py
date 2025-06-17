from kedro.pipeline import node, Pipeline
from .nodes import (
    explode_competitions,
    format_matches,
    format_competitions,
    consolidate_events,
)


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
            node(
                func=consolidate_events,
                inputs=[
                    "first_partitioned_raw_events",
                    "params:first_raw_events_partition",
                ],
                outputs="intermediate_first_partitioned_events",
                name="consolidate_first_events_node",
            ),
            node(
                func=consolidate_events,
                inputs=[
                    "second_partitioned_raw_events",
                    "params:second_raw_events_partition",
                ],
                outputs="intermediate_second_partitioned_events",
                name="consolidate_second_events_node",
            ),
        ],
    )
