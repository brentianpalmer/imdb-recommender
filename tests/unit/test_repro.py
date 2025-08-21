import random

import numpy as np

from imdb_recommender.utils.repro import set_global_seed


def test_set_global_seed_deterministic(monkeypatch):
    monkeypatch.delenv("IMDBREC_SEED", raising=False)
    set_global_seed(1234)
    py_vals = [random.random(), random.random()]
    np_vals = np.random.rand(2)
    set_global_seed(1234)
    assert py_vals == [random.random(), random.random()]
    assert np.array_equal(np_vals, np.random.rand(2))


def test_env_override(monkeypatch):
    monkeypatch.setenv("IMDBREC_SEED", "4321")
    set_global_seed(1234)  # arg should be ignored
    rand_val = random.random()
    np_val = np.random.rand()
    set_global_seed(4321)
    assert rand_val == random.random()
    assert np_val == np.random.rand()
