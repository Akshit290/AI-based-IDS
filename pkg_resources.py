# Minimal shim for pkg_resources used by some packages (local workaround)
# Provides get_distribution(name) and parse_version(version) used by Dash

class _Dist:
    def __init__(self, version="0"):
        self.version = version


def get_distribution(name):
    # Return an object with a .version attribute
    return _Dist(version="0")


def parse_version(v):
    return v
