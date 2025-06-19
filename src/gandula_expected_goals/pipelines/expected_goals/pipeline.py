from kedro.pipeline import node, Pipeline
from .nodes import (
    explode_competitions,
    format_matches,
    format_competitions,
    consolidate_event_partition_into_kedro,
    consolidate_events,
    get_shot_events,
    extract_features_from_shots,
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
                func=consolidate_event_partition_into_kedro,
                inputs=[
                    "first_partitioned_raw_events",
                    "params:first_raw_events_partition",
                ],
                outputs="intermediate_first_partitioned_events",
                name="consolidate_first_events_node",
            ),
            node(
                func=consolidate_event_partition_into_kedro,
                inputs=[
                    "second_partitioned_raw_events",
                    "params:second_raw_events_partition",
                ],
                outputs="intermediate_second_partitioned_events",
                name="consolidate_second_events_node",
            ),
            node(
                func=consolidate_events,
                inputs=[
                    "intermediate_first_partitioned_events",
                    "intermediate_second_partitioned_events",
                ],
                outputs="primary_events",
                name="consolidate_events_node",
            ),
            node(
                func=get_shot_events,
                inputs="primary_events",
                outputs="shot_events",
                name="extract_shots_node",
            ),
            node(
                func=extract_features_from_shots,
                inputs="shot_events",
                outputs="features",
                name="extract_features_from_shots_node",
            ),
        ],
    )
