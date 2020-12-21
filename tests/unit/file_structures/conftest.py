import os
import pytest
import tempfile as tf



dir_sim = "/tmp/"

#@pytest.fixture(scope="session", autouse=True)
def pytest_sessionstart(session):
    for snap_nr in range(1, 12):
        dir_name = dir_sim + "rockstar_%03d/" % snap_nr
        os.mkdir(dir_name)
        for file_nr in range(1, 8):
            filename = dir_name + "halos_0.%d.ascii" % file_nr
            open(filename, 'w+b')


#@pytest.fixture(scope="session")
def pytest_sessionfinish(session, exitstatus):
    for snap_nr in range(1, 12):
        dir_name = "/tmp/rockstar_%03d/" % snap_nr
        for file_nr in range(1, 8):
            filename = dir_name + "halos_0.%d.ascii" % file_nr
            os.remove(filename)
        os.rmdir(dir_name)
