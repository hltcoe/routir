import time
from typing import Any, Dict, List, Union

from ..models.abstract import Engine
from .abstract import Processor


_callable_service_types = {"search", "score", "decompose_query", "fuse"}
_uncallable_service_types = {"content"}

# TODO: why am I creating this mess...
_output_keys = {"search": "scores", "score": "scores", "decompose_query": "queries", "fuse": "scores"}


class _ProcessorRegistry:
    def __init__(self):
        # TODO: make collection name part of the namespace
        self.all_services: Dict[str, Dict[str, Processor]] = {}

    @property
    def valid_service_types(self):
        return _callable_service_types | _uncallable_service_types

    def __getitem__(self, name: str):
        return self.all_services[name]

    def register(self, name: str, service_type: str, processor: Processor):
        assert service_type in self.valid_service_types, f"Invalid service type `{service_type}`."
        assert name not in self.all_services or service_type not in self.all_services[name], (
            f"Service type `{service_type}` of name `{name}` already exists."
        )

        if name not in self.all_services:
            self.all_services[name] = {}
        self.all_services[name][service_type] = processor

    def has_service(self, name: str, service_type: str):
        return name in self.all_services and service_type in self.all_services[name]

    def get(self, name: str, service_type: str):
        if self.has_service(name, service_type):
            return self.all_services[name][service_type]

    def get_all_services(self):
        services = {t: [] for t in self.valid_service_types}
        for name, processors in self.all_services.items():
            for s in processors:
                services[s].append(name)
        return services


class DummyProcessor(Processor):
    def __init__(self, engine: Engine, method: str):
        super().__init__(cache_size=0)
        assert hasattr(engine, method)
        self.service = getattr(engine, method)
        self.result_key = _output_keys[method]

    async def submit(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {self.result_key: (await self.service(**item)), "processed": True, "timestamp": time.time()}


# singleton
ProcessorRegistry = _ProcessorRegistry()


def auto_register(methods: Union[str, List[str]], **default_init_kwargs):
    if isinstance(methods, str):
        methods = [methods]

    def engine_dec(engine_cls: type[Engine]):
        assert issubclass(engine_cls, Engine)
        assert all(getattr(engine_cls, f"can_{method}") for method in methods)
        # register each
        for method in methods:
            ProcessorRegistry.register(
                engine_cls.__name__, method, processor=DummyProcessor(engine_cls(**default_init_kwargs), method)
            )

        return engine_cls

    return engine_dec
