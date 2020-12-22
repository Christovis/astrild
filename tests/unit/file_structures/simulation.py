import os
import pytest
import tempfile as tf
from pathlib import Path

import pandas as pd

from wys_ars.simulation import Simulation

dir_src = Path(__file__).parent.absolute()
sim_config_part_times = pd.read_hdf(dir_src / "../../test_data/particle_snapshot_info.h5", key="df")
dir_sim = "/tmp/box1/"

@pytest.fixture(scope="session", name="simulation")
def test__init_simulation():
    return Simulation(
        dir_sim=dir_sim,
        dir_out=dir_sim,
        file_dsc={"root": 'halos', "extension": 'ascii'},
        dir_root='rockstar',
    )

def test__simulation_name(simulation):
    assert simulation.name == "box1"

def test__simulation_dirs(simulation):
    assert simulation.dirs["sim"] == "/tmp/box1/"
    assert simulation.dirs["out"] == "/tmp/box1/"
    assert len(simulation.dirs["rockstar"]) == 11

def test__simulation_files(simulation):
    assert len(simulation.files["halos"]["1"]) == 7

