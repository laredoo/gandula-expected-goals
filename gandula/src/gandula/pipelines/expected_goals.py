from kedro.pipeline import pipeline
from nodes.expected_goals import return_greeting_node, join_statements_node

# Assemble nodes into a pipeline
greeting_pipeline = pipeline([return_greeting_node, join_statements_node])
