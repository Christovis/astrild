import os
import pytest
import tempfile as tf



dir_sim = "/tmp/"

#@pytest.fixture(scope="session", autouse=True)
def pytest_sessionstart(session):
    for box_nr in range(1, 5):
        dir_box = dir_sim + "box%d/" % box_nr
        os.mkdir(dir_box)
        for snap_nr in range(1, 12):
            dir_rock = dir_box + "rockstar_%03d/" % snap_nr
            os.mkdir(dir_rock)
            for file_nr in range(1, 8):
                filename = dir_rock + "halos_0.%d.ascii" % file_nr
                open(filename, 'w+b')


#@pytest.fixture(scope="session")
def pytest_sessionfinish(session, exitstatus):
    for box_nr in range(1, 5):
        dir_box = dir_sim + "box%d/" % box_nr
        for snap_nr in range(1, 12):
            dir_rock = dir_box + "/rockstar_%03d/" % snap_nr
            for file_nr in range(1, 8):
                filename = dir_rock + "halos_0.%d.ascii" % file_nr
                os.remove(filename)
            os.rmdir(dir_rock)
        os.rmdir(dir_box)
