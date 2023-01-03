"""
Define custom exception methods
"""


class InvalidGraph(Exception):
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)


def invalid_graph(graph):
    msg = "Method not implelented for weighted graph and required backend"
    raise InvalidGraph(msg)
