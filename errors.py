class NeuvolError(Exception):
    """
    Base exception class of neuvol library
    """


class NeuvolArchitectureError(NeuvolError):
    """
    Error in architecture related with shape incompatibilities (e.g. negative size output of CNN)
    """
