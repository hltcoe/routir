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
    # PR4: both transports surface the same keys, but the per-key values are
    # heterogeneous (lists for callable roles, dicts for content/maps).
    assert set(rest_avail.keys()) == set(grpc_avail.keys())
    for role, values in rest_avail.items():
        if isinstance(values, list):
            assert set(values) == set(grpc_avail[role])
        else:
            assert values == grpc_avail[role]
    # Structured keys must be present on both sides.
    for key in ("collection", "score_view_kinds", "collection_view_kinds", "pipeline_aliases"):
        assert key in rest_avail, f"REST avail missing key '{key}'"
        assert key in grpc_avail, f"gRPC avail missing key '{key}'"


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


async def test_constructor_requires_at_least_one_endpoint():
    with pytest.raises(ValueError):
        AsyncClient()


async def test_scheme_inference_routes_grpc_url_to_grpc_slot():
    c = AsyncClient("grpc://host:50051")
    assert c.endpoint is None
    assert c.grpc_endpoint == "grpc://host:50051"


async def test_scheme_inference_routes_http_url_to_rest_slot():
    c = AsyncClient("http://host:5000")
    assert c.endpoint == "http://host:5000"
    assert c.grpc_endpoint is None


async def test_no_scheme_endpoint_warns_and_assumes_rest(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="search-service"):
        c = AsyncClient("host:5000")
    assert c.endpoint == "host:5000"
    assert c.grpc_endpoint is None
    assert any("no URL scheme" in r.message for r in caplog.records)


async def test_grpc_only_endpoint_no_rest_fallback(server_both):
    # Use only the grpc endpoint via grpc:// scheme. No REST URL → if gRPC
    # works, we use it; if it fails, there is no fallback target.
    _rest_url, grpc_target = server_both
    grpc_url = f"grpc://{grpc_target}"

    async with AsyncClient(grpc_url) as c:
        assert c.endpoint is None
        assert c.transport == "grpc"
        ping = await c.ping()
        assert ping == {"status": "pong"}


async def test_grpc_only_unreachable_raises(caplog):
    # No REST endpoint and a dead gRPC port: must raise, not hang or fall back.
    c = AsyncClient("grpc://127.0.0.1:1")
    with pytest.raises(RoutirClientError):
        await c._start()
    await c.close()


async def test_autodiscover_grpc_via_avail(server_both, caplog):
    import logging
    rest_url, grpc_target = server_both
    # Only the REST URL: the client should hit /avail, see the advertised
    # grpc_port, derive the gRPC target, probe it, and end up on gRPC.
    with caplog.at_level(logging.WARNING, logger="search-service"):
        async with AsyncClient(endpoint=rest_url) as c:
            assert c.transport == "grpc"
            assert c.grpc_endpoint is not None
            # Auto-discovered host:port must agree with the server fixture.
            _expected_host, expected_port = grpc_target.rsplit(":", 1)
            assert c.grpc_endpoint.endswith(f":{expected_port}")
    assert any("Auto-discovered gRPC" in r.message for r in caplog.records)


async def test_avail_includes_grpc_port_on_both_transports(server_both):
    rest_url, grpc_target = server_both

    async with AsyncClient(endpoint=rest_url, transport="rest") as c_rest:
        rest_avail = await c_rest.avail()
    async with AsyncClient(endpoint=rest_url, grpc_endpoint=grpc_target) as c_grpc:
        assert c_grpc.transport == "grpc"
        grpc_avail = await c_grpc.avail()

    assert "grpc_port" in rest_avail
    assert "grpc_port" in grpc_avail
    assert rest_avail["grpc_port"] == grpc_avail["grpc_port"]
    # Trivial fixture registers only a text search service.  ``collection``
    # is an empty dict (no collections), ``score_view_kinds`` is empty too.
    assert rest_avail["collection"] == {}
    assert rest_avail["collection_view_kinds"] == {}
    assert rest_avail["score_view_kinds"] == {}
    assert grpc_avail["collection"] == {}
    assert grpc_avail["collection_view_kinds"] == {}
    assert grpc_avail["score_view_kinds"] == {}


async def test_avail_omits_grpc_port_when_rest_only(server_rest_only):
    async with AsyncClient(endpoint=server_rest_only, transport="rest") as c:
        avail = await c.avail()
    assert "grpc_port" not in avail
    # Structured keys still exist on REST-only servers (they're independent
    # of whether gRPC is up).
    assert "collection" in avail and isinstance(avail["collection"], dict)
    assert "score_view_kinds" in avail
    assert "collection_view_kinds" in avail
