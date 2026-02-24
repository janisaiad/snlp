import pytest


def test_env():
    import snlp
    assert snlp is not None


if __name__ == "__main__":
    pytest.main()