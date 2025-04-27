from kedro.pipeline import node, Pipeline
from .nodes import return_greeting, join_statements


def create_pipeline(**kwargs) -> Pipeline:
    """Create a pipeline to return a greeting message."""
    return Pipeline(
        [
            node(
                func=return_greeting,
                inputs="players",
                outputs=None,
                name="return_greeting_node",
            ),
            node(
                func=join_statements,
                inputs="competitions",
                outputs=None,
                name="join_statements_node",
            ),
        ],
    )