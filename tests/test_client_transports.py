"""Round-trip equivalence + fallback + auth tests for the client transports.

The point is to catch regressions where REST and gRPC stop returning shape-
equivalent payloads (the contract :class:`AsyncClient` relies on for transport
swap), and where the fallback / auth probe logic drifts from the documented
behavior.
"""

import pytest

from routir.client import AsyncClient, RoutirClientError


async def test_ping_avail_match(server_both):
    rest_url, grpc_target = server_both

    async with AsyncClient(endpoint=rest_url, transport="rest") as c_rest:
        rest_ping = await c_rest.ping()
        rest_avail = await c_rest.avail()
        assert c_rest.transport == "rest"

    async with AsyncClient(
        endpoint=rest_url,
        grpc_endpoint=grpc_target,
        transport="auto",
    ) as c_auto:
        grpc_ping = await c_auto.ping()
        grpc_avail = await c_auto.avail()
        assert c_auto.transport == "grpc"

    # Ping is a fixed-shape dict; must match.
    assert rest_ping == grpc_ping
    # Avail role-keys must agree; the value lists are compared as sets so a
    # difference in declaration order doesn't trip the assertion.
    assert set(rest_avail.keys()) == set(grpc_avail.keys())
    for role, values in rest_avail.items():
        if isinstance(values, list):
            assert set(values) == set(grpc_avail[role])
        else:
            assert values == grpc_avail[role]


async def test_fallback_to_rest_when_grpc_unreachable(server_rest_only):
    rest_url = server_rest_only
    # Pick a port that's almost certainly not listening; the probe must fail
    # at the channel level (UNAVAILABLE) and trigger the REST fallback.
    bogus_grpc = "127.0.0.1:1"

    async with AsyncClient(
        endpoint=rest_url,
        grpc_endpoint=bogus_grpc,
        transport="auto",
    ) as c:
        result = await c.ping()
        assert result == {"status": "pong"}
        assert c.transport == "rest"


async def test_grpc_required_fails_without_endpoint():
    c = AsyncClient(endpoint="http://example.invalid", transport="grpc")
    with pytest.raises(RoutirClientError):
        await c._start()
    await c.close()


async def test_unauthenticated_does_not_fall_back(server_authed):
    rest_url, grpc_target, _api_key = server_authed

    async with AsyncClient(
        endpoint=rest_url,
        grpc_endpoint=grpc_target,
        api_key="wrong-key",
        transport="auto",
    ) as c:
        # Ping is exempt from auth on both transports.
        ping = await c.ping()
        assert ping == {"status": "pong"}
        # The first authenticated call must surface UNAUTHENTICATED, not
        # silently downgrade to REST.
        with pytest.raises(RoutirClientError) as excinfo:
            await c.avail()
        msg = str(excinfo.value).lower()
        assert "unauthenticated" in msg
        # We stayed on gRPC.
        assert c.transport == "grpc"


async def test_search_round_trip(server_both):
    rest_url, grpc_target = server_both

    payload_kwargs = {"limit": 5}

    async with AsyncClient(endpoint=rest_url, transport="rest") as c_rest:
        rest_result = await c_rest.search("trivial", "hello", **payload_kwargs)

    async with AsyncClient(
        endpoint=rest_url,
        grpc_endpoint=grpc_target,
        transport="auto",
    ) as c_grpc:
        assert c_grpc.transport == "grpc"
        grpc_result = await c_grpc.search("trivial", "hello", **payload_kwargs)

    # Server-side timestamps drift between the two calls; everything else
    # must match exactly so the wire formats stay equivalent.
    assert rest_result["query"] == grpc_result["query"]
    assert rest_result["scores"] == grpc_result["scores"]
    assert rest_result["service"] == grpc_result["service"]
    assert rest_result["scores"] == {"doc1": 1.0, "doc2": 0.5}
