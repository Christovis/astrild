import os
import pytest
import tempfile as tf
from pathlib import Path

import pandas as pd
import numpy as np

from wys_ars.simcoll import SimulationCollection as SimColl

dir_src = Path(__file__).parent.absolute()
sim_config_times = dir_src / "../../test_data/particle_snapshot_info.h5"
sim_config_places = dir_src / "../../test_data/rockstar_simulation_collection.yaml"
dir_sim = "/tmp/"

@pytest.fixture(scope="session", name="simcoll")
def test__init_simulation():
    return SimColl.from_file(sim_config_places, sim_config_times)

def test__simulation_collection_numbers(simcoll):
    assert (simcoll.sim_nrs == np.arange(1, 5)).all()

def test__simulation_collection_element(simcoll):
    assert simcoll.sim["sim1"].name == "box1"
    assert simcoll.sim["sim1"].boxsize == 1000.
    assert simcoll.sim["sim1"].domain_level == 1024
    assert len(simcoll.sim["sim1"].dirs["rockstar"]) == 11

