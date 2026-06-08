"""Dataset-agnostic LiteReason pipeline.

The same stages the example experiments (Flawed Fictions, GSM-Hard) use, made
generic over a ``{prompt, answer}`` JSONL dataset and a user-provided
``reward_func.py`` (the OpenRLHF reward convention). Run each stage with
``python -m litereason.pipeline.<stage>``.
"""
