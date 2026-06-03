"""Unit tests for ``ProcessorRegistry.submit`` and ``ServiceNotFound``."""

import pytest

from routir.processors.abstract import Processor
from routir.processors.registry import ProcessorRegistry, ServiceNotFound


class _EchoProcessor(Processor):
    """Echoes the request dict so we can assert what reached ``submit``."""

    def __init__(self):
        super().__init__(cache_size=-1)

    async def _submit(self, item):
        return {"echoed": dict(item)}


_STUB_NAME = "stub-registry-test"


@pytest.fixture
def stub_service():
    """Register an echo processor under a unique name and tear it down after."""
    proc = _EchoProcessor()
    ProcessorRegistry.register(_STUB_NAME, "search", proc)
    try:
        yield proc
    finally:
        ProcessorRegistry.all_services.get(_STUB_NAME, {}).pop("search", None)
        ProcessorRegistry.slot_meta.get(_STUB_NAME, {}).pop("search", None)
        if _STUB_NAME in ProcessorRegistry.all_services and not ProcessorRegistry.all_services[_STUB_NAME]:
            del ProcessorRegistry.all_services[_STUB_NAME]
        if _STUB_NAME in ProcessorRegistry.slot_meta and not ProcessorRegistry.slot_meta[_STUB_NAME]:
            del ProcessorRegistry.slot_meta[_STUB_NAME]


async def test_submit_returns_processor_result(stub_service):
    result = await ProcessorRegistry.submit(_STUB_NAME, "search", {"query": "q"})
    # ``Processor.submit`` adds ``cached`` since caching is disabled.
    assert result["echoed"] == {"query": "q"}
    assert result["cached"] is False


async def test_submit_missing_service_raises_service_not_found():
    with pytest.raises(ServiceNotFound):
        await ProcessorRegistry.submit("does-not-exist", "search", {})


async def test_service_not_found_str_is_plain_message():
    try:
        await ProcessorRegistry.submit("does-not-exist", "score", {})
    except ServiceNotFound as e:
        msg = str(e)
        # No KeyError-style repr quotes wrapping the message.
        assert not (msg.startswith("'") and msg.endswith("'"))
        assert "does-not-exist" in msg
        assert "score" in msg
    else:
        pytest.fail("expected ServiceNotFound")


def test_service_not_found_is_key_error_subclass():
    # Backward compat: existing ``except KeyError`` clauses must still catch.
    assert issubclass(ServiceNotFound, KeyError)


async def test_submit_dispatches_to_same_processor_as_get(stub_service):
    fetched = ProcessorRegistry.get(_STUB_NAME, "search")
    assert fetched is stub_service
    # The submit helper is a sugar layer over the same processor.
    via_get = await fetched.submit({"query": "via-get"})
    via_submit = await ProcessorRegistry.submit(_STUB_NAME, "search", {"query": "via-submit"})
    assert via_get["echoed"] == {"query": "via-get"}
    assert via_submit["echoed"] == {"query": "via-submit"}
