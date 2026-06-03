import grpc
from google.protobuf.json_format import MessageToDict

from .pipeline import PipelineAliasRegistry, SearchPipeline
from .pipeline.cache import get_bytes_content_cache_max_bytes
from .processors.registry import ProcessorRegistry, ServiceNotFound
from .proto._generated import routir_pb2 as pb
from .proto._generated import routir_pb2_grpc as pb_grpc
from .utils import logger


def _decode_score_passages(passages, context):
    """Decode ``repeated Passage`` to ``List[str]`` for the text score path.

    PR2 only wires the text arm end-to-end.  Bytes-engine support lands in
    PR4/5 -- for now, a bytes passage is rejected with INVALID_ARGUMENT so
    clients see a clear error rather than a silent misinterpretation.
    """
    out = []
    for i, p in enumerate(passages):
        which = p.WhichOneof("value")
        if which == "text":
            out.append(p.text)
        elif which == "bytes":
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(
                f"passage[{i}] uses bytes; bytes scoring is not yet wired (lands in a later PR)."
            )
            return None
        else:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"passage[{i}] has no value set (oneof empty).")
            return None
    return out


class RoutirServicer(pb_grpc.RoutirServicer):
    """gRPC servicer mirroring the REST surface in :mod:`routir.serve`."""

    def __init__(self, grpc_port: int = None):
        # Surfaced in Avail so REST-only clients can auto-discover gRPC.
        self._grpc_port = grpc_port

    async def Ping(self, request, context):
        return pb.PingResponse(status="pong")

    async def Avail(self, request, context):
        try:
            services_dict = ProcessorRegistry.get_all_services()
            aliases = PipelineAliasRegistry.source
            kwargs = {}
            if self._grpc_port is not None:
                kwargs["grpc_port"] = self._grpc_port

            # PR4: ``content`` is no longer exposed via the ``services`` map
            # (heterogeneous per-collection metadata wouldn't fit a
            # ``StringList``).  Build the dedicated ``content_views`` map
            # from each collection's slot metadata and drop the key from
            # ``services`` so the callable-role list stays clean.
            content_names = services_dict.pop("content", [])
            content_views = {}
            for name in content_names:
                meta = ProcessorRegistry.get_meta(name, "content")
                cv_kwargs = {"views": dict(meta.get("views") or {})}
                default_view = meta.get("default_view")
                if default_view is not None:
                    cv_kwargs["default"] = default_view
                content_views[name] = pb.ContentViewKinds(**cv_kwargs)

            score_view_kinds = {
                name: ProcessorRegistry.get_meta(name, "score").get("view_kind", "text")
                for name in services_dict.get("score", [])
            }

            return pb.AvailResponse(
                services={role: pb.StringList(items=list(names)) for role, names in services_dict.items()},
                pipeline_aliases=dict(aliases),
                content_views=content_views,
                score_view_kinds=score_view_kinds,
                **kwargs,
            )
        except Exception as e:
            logger.exception("gRPC Avail failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"{type(e).__name__}: {e}")
            return pb.AvailResponse()

    async def Search(self, request, context):
        try:
            if not request.service:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("service is required")
                return pb.SearchResponse()

            data = {"query": request.query}
            if request.HasField("limit"):
                data["limit"] = request.limit
            if request.HasField("subset"):
                data["subset"] = request.subset
            if request.HasField("instruction"):
                data["instruction"] = request.instruction
            data.update(MessageToDict(request.extras))

            try:
                result = await ProcessorRegistry.submit(request.service, "search", data)
            except ServiceNotFound as e:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(str(e))
                return pb.SearchResponse()
            except ValueError as e:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"{type(e).__name__}: {e}")
                return pb.SearchResponse()

            return pb.SearchResponse(
                query=result.get("query", request.query),
                scores=result.get("scores") or {},
                service=result.get("service", request.service),
                cached=bool(result.get("cached", False)),
                timestamp=float(result.get("timestamp", 0.0)),
            )
        except Exception as e:
            logger.exception("gRPC Search failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"{type(e).__name__}: {e}")
            return pb.SearchResponse()

    async def Score(self, request, context):
        try:
            if not request.service:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("service is required")
                return pb.ScoreResponse()

            decoded = _decode_score_passages(request.passages, context)
            if decoded is None:
                # _decode_score_passages already set status/details on context.
                return pb.ScoreResponse()
            data = {"query": request.query, "passages": decoded}
            if request.HasField("prompt"):
                data["prompt"] = request.prompt
            data.update(MessageToDict(request.extras))

            try:
                result = await ProcessorRegistry.submit(request.service, "score", data)
            except ServiceNotFound as e:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(str(e))
                return pb.ScoreResponse()
            except ValueError as e:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"{type(e).__name__}: {e}")
                return pb.ScoreResponse()

            meta_kwargs = {}
            n_passages = result.get("meta", {}).get("n_passages") if isinstance(result.get("meta"), dict) else None
            if n_passages is not None:
                meta_kwargs["meta"] = pb.ScoreMeta(n_passages=int(n_passages))

            return pb.ScoreResponse(
                query=result.get("query", request.query),
                scores=[float(s) for s in (result.get("scores") or [])],
                service=result.get("service", request.service),
                cached=bool(result.get("cached", False)),
                timestamp=float(result.get("timestamp", 0.0)),
                **meta_kwargs,
            )
        except Exception as e:
            logger.exception("gRPC Score failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"{type(e).__name__}: {e}")
            return pb.ScoreResponse()

    async def Content(self, request, context):
        try:
            data = {"id": request.id, "collection": request.collection}
            if request.HasField("view"):
                data["view"] = request.view

            try:
                result = await ProcessorRegistry.submit(request.collection, "content", data)
            except ServiceNotFound as e:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(str(e))
                return pb.ContentResponse()
            except ValueError as e:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"{type(e).__name__}: {e}")
                return pb.ContentResponse()

            if "error" in result:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(str(result["error"]))
                return pb.ContentResponse()

            resp_kwargs = {
                "collection": request.collection,
                "id": request.id,
                "cached": bool(result.get("cached", False)),
                "timestamp": float(result.get("timestamp", 0.0)),
            }
            if "view" in result and result["view"]:
                resp_kwargs["view"] = result["view"]
            # ``oneof`` constraint: only one of text / data can be set; prefer
            # data when both are present.
            if "data" in result:
                resp_kwargs["data"] = pb.BytesParts(parts=list(result["data"]))
            else:
                resp_kwargs["text"] = result.get("text", "")
            return pb.ContentResponse(**resp_kwargs)
        except Exception as e:
            logger.exception("gRPC Content failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"{type(e).__name__}: {e}")
            return pb.ContentResponse()

    async def Pipeline(self, request, context):
        try:
            if not request.pipeline:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("pipeline is required")
                return pb.PipelineResponse()
            if not request.query:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("query is required")
                return pb.PipelineResponse()

            runtime_kwargs = {alias: MessageToDict(struct) for alias, struct in request.runtime_kwargs.items()}
            collection = request.collection if request.HasField("collection") else None

            try:
                result = await SearchPipeline.cached_run(
                    request.pipeline,
                    request.query,
                    collection=collection,
                    runtime_kwargs=runtime_kwargs,
                    bytes_content_cache_max_bytes=get_bytes_content_cache_max_bytes(),
                )
            except (ValueError, RuntimeError) as e:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"{type(e).__name__}: {e}")
                return pb.PipelineResponse()

            return pb.PipelineResponse(
                query=result.get("query", request.query),
                scores=result.get("scores") or {},
                expanded_queries=result.get("expanded_queries") or [],
                cached=bool(result.get("cached", False)),
                timestamp=float(result.get("timestamp", 0.0)),
            )
        except Exception as e:
            logger.exception("gRPC Pipeline failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"{type(e).__name__}: {e}")
            return pb.PipelineResponse()
