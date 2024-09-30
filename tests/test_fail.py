import pytest


def fail():
    raise SystemExit(1)


def test_fail():
    with pytest.raises(SystemExit):
        fail()
