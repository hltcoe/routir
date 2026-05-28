from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class PingRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class PingResponse(_message.Message):
    __slots__ = ("status",)
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: str
    def __init__(self, status: _Optional[str] = ...) -> None: ...

class AvailRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class StringList(_message.Message):
    __slots__ = ("items",)
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, items: _Optional[_Iterable[str]] = ...) -> None: ...

class AvailResponse(_message.Message):
    __slots__ = ("services", "pipeline_aliases")
    class ServicesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: StringList
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[StringList, _Mapping]] = ...) -> None: ...
    class PipelineAliasesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    SERVICES_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_ALIASES_FIELD_NUMBER: _ClassVar[int]
    services: _containers.MessageMap[str, StringList]
    pipeline_aliases: _containers.ScalarMap[str, str]
    def __init__(self, services: _Optional[_Mapping[str, StringList]] = ..., pipeline_aliases: _Optional[_Mapping[str, str]] = ...) -> None: ...

class SearchRequest(_message.Message):
    __slots__ = ("service", "query", "limit", "subset", "instruction", "extras")
    SERVICE_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    SUBSET_FIELD_NUMBER: _ClassVar[int]
    INSTRUCTION_FIELD_NUMBER: _ClassVar[int]
    EXTRAS_FIELD_NUMBER: _ClassVar[int]
    service: str
    query: str
    limit: int
    subset: str
    instruction: str
    extras: _struct_pb2.Struct
    def __init__(self, service: _Optional[str] = ..., query: _Optional[str] = ..., limit: _Optional[int] = ..., subset: _Optional[str] = ..., instruction: _Optional[str] = ..., extras: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class SearchResponse(_message.Message):
    __slots__ = ("query", "scores", "service", "cached", "timestamp")
    class ScoresEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    QUERY_FIELD_NUMBER: _ClassVar[int]
    SCORES_FIELD_NUMBER: _ClassVar[int]
    SERVICE_FIELD_NUMBER: _ClassVar[int]
    CACHED_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    query: str
    scores: _containers.ScalarMap[str, float]
    service: str
    cached: bool
    timestamp: float
    def __init__(self, query: _Optional[str] = ..., scores: _Optional[_Mapping[str, float]] = ..., service: _Optional[str] = ..., cached: bool = ..., timestamp: _Optional[float] = ...) -> None: ...

class ScoreRequest(_message.Message):
    __slots__ = ("service", "query", "passages", "prompt", "extras")
    SERVICE_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    PASSAGES_FIELD_NUMBER: _ClassVar[int]
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    EXTRAS_FIELD_NUMBER: _ClassVar[int]
    service: str
    query: str
    passages: _containers.RepeatedScalarFieldContainer[str]
    prompt: str
    extras: _struct_pb2.Struct
    def __init__(self, service: _Optional[str] = ..., query: _Optional[str] = ..., passages: _Optional[_Iterable[str]] = ..., prompt: _Optional[str] = ..., extras: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class ScoreMeta(_message.Message):
    __slots__ = ("n_passages",)
    N_PASSAGES_FIELD_NUMBER: _ClassVar[int]
    n_passages: int
    def __init__(self, n_passages: _Optional[int] = ...) -> None: ...

class ScoreResponse(_message.Message):
    __slots__ = ("query", "scores", "service", "cached", "timestamp", "meta")
    QUERY_FIELD_NUMBER: _ClassVar[int]
    SCORES_FIELD_NUMBER: _ClassVar[int]
    SERVICE_FIELD_NUMBER: _ClassVar[int]
    CACHED_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    META_FIELD_NUMBER: _ClassVar[int]
    query: str
    scores: _containers.RepeatedScalarFieldContainer[float]
    service: str
    cached: bool
    timestamp: float
    meta: ScoreMeta
    def __init__(self, query: _Optional[str] = ..., scores: _Optional[_Iterable[float]] = ..., service: _Optional[str] = ..., cached: bool = ..., timestamp: _Optional[float] = ..., meta: _Optional[_Union[ScoreMeta, _Mapping]] = ...) -> None: ...

class ContentRequest(_message.Message):
    __slots__ = ("collection", "id")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    collection: str
    id: str
    def __init__(self, collection: _Optional[str] = ..., id: _Optional[str] = ...) -> None: ...

class ContentResponse(_message.Message):
    __slots__ = ("collection", "id", "text", "cached", "timestamp")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    CACHED_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    collection: str
    id: str
    text: str
    cached: bool
    timestamp: float
    def __init__(self, collection: _Optional[str] = ..., id: _Optional[str] = ..., text: _Optional[str] = ..., cached: bool = ..., timestamp: _Optional[float] = ...) -> None: ...

class PipelineRequest(_message.Message):
    __slots__ = ("pipeline", "query", "collection", "runtime_kwargs")
    class RuntimeKwargsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _struct_pb2.Struct
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...
    PIPELINE_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_KWARGS_FIELD_NUMBER: _ClassVar[int]
    pipeline: str
    query: str
    collection: str
    runtime_kwargs: _containers.MessageMap[str, _struct_pb2.Struct]
    def __init__(self, pipeline: _Optional[str] = ..., query: _Optional[str] = ..., collection: _Optional[str] = ..., runtime_kwargs: _Optional[_Mapping[str, _struct_pb2.Struct]] = ...) -> None: ...

class PipelineResponse(_message.Message):
    __slots__ = ("query", "scores", "expanded_queries", "cached", "timestamp")
    class ScoresEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    QUERY_FIELD_NUMBER: _ClassVar[int]
    SCORES_FIELD_NUMBER: _ClassVar[int]
    EXPANDED_QUERIES_FIELD_NUMBER: _ClassVar[int]
    CACHED_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    query: str
    scores: _containers.ScalarMap[str, float]
    expanded_queries: _containers.RepeatedScalarFieldContainer[str]
    cached: bool
    timestamp: float
    def __init__(self, query: _Optional[str] = ..., scores: _Optional[_Mapping[str, float]] = ..., expanded_queries: _Optional[_Iterable[str]] = ..., cached: bool = ..., timestamp: _Optional[float] = ...) -> None: ...
