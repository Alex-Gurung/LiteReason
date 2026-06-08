"""Load and apply a user-provided reward function.

A task is defined by one ``reward_func.py`` that exposes the OpenRLHF reward
convention::

    def reward_func(queries, prompts, labels, **kwargs) -> dict
        # queries are the model responses; returns {"rewards": <seq of float>}

IMPORTANT contract for bring-your-own reward functions: the user's
``reward_func`` receives the model's generated **responses** as its FIRST
positional argument (named ``queries`` by OpenRLHF convention), then
``prompts``, then ``labels``. ``queries`` is NOT the input prompts -- it is
what the model produced.

The same function drives RL (via ``--reward_func``) and scores evaluation, so a
new task needs no extra glue.
"""

import importlib.util
import os
from typing import Callable, List, Sequence


def load_reward_func(path: str) -> Callable:
    """Import ``reward_func`` from a Python file path."""
    abspath = os.path.abspath(path)
    if not os.path.isfile(abspath):
        raise FileNotFoundError(f"reward function file not found: {abspath!r}")
    spec = importlib.util.spec_from_file_location("litereason_user_reward", abspath)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load a module from {abspath!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "reward_func", None)
    if not callable(fn):
        raise AttributeError(f"{abspath!r} does not define a callable `reward_func`")
    return fn


def score_responses(
    reward_func: Callable,
    responses: Sequence[str],
    prompts: Sequence[str],
    labels: Sequence,
) -> List[float]:
    """Call ``reward_func`` and return a flat list of float rewards.

    ``reward_func`` is called positionally as ``reward_func(responses, prompts,
    labels)``. By OpenRLHF convention the first positional argument is named
    ``queries`` -- it is the model's generated responses, not the input prompts.

    Accepts the OpenRLHF return shape (``{"rewards": tensor|list}``) or a bare
    sequence of floats.
    """
    out = reward_func(list(responses), list(prompts), list(labels))
    rewards = out["rewards"] if isinstance(out, dict) else out
    tolist = getattr(rewards, "tolist", None)  # torch tensor / numpy array
    if callable(tolist):
        rewards = tolist()
    return [float(x) for x in rewards]
