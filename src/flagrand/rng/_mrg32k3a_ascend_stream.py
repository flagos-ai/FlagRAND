"""Read the current registered Ascend stream through a private driver view."""

from types import SimpleNamespace


class _RegisteredStream:
    def __init__(self, runtime_driver):
        self.runtime_driver = runtime_driver
        self.active_driver = runtime_driver.active
        method = self.active_driver.get_current_stream
        self.driver_globals = method.__func__.__globals__
        self.backend_function = self.driver_globals["get_backend_func"]
        self.backend_globals = self.backend_function.__globals__
        self.driver_method = self.active_driver.get_current_stream

    def __getattr__(self, name):
        return getattr(self.runtime_driver.active, name)

    def get_current_device(self):
        return self.runtime_driver.active.get_current_device()

    def get_current_stream(self, device):
        active = self.runtime_driver.active
        method = active.get_current_stream
        # Bound-method equality compares the function and receiver. Attribute
        # access can create a new bound-method object on every call.
        if (
            active is not self.active_driver
            or method != self.driver_method
            or self.driver_globals["get_backend_func"] is not self.backend_function
        ):
            return method(device)
        namespace = self.backend_globals
        policy = namespace["backend_policy"]
        registry = namespace["backend_strategy_registry"]._instance
        if policy is None or registry is None:
            return method(device)
        methods = registry.strategies.get(policy)
        if methods is None or "get_current_stream" not in methods:
            return method(device)
        return methods["get_current_stream"](device)


def create_driver_view(runtime_driver):
    return SimpleNamespace(active=_RegisteredStream(runtime_driver))
