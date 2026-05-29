import grpc
from google.protobuf.json_format import MessageToDict

from .pipeline import PipelineAliasRegistry, SearchPipeline
from .processors.registry import ProcessorRegistry, ServiceNotFound
from .proto._generated import routir_pb2 as pb
from .proto._generated import routir_pb2_grpc as pb_grpc
from .utils import logger


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
            return pb.AvailResponse(
                services={role: pb.StringList(items=list(names)) for role, names in services_dict.items()},
                pipeline_aliases=dict(aliases),
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

            data = {"query": request.query, "passages": list(request.passages)}
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
            try:
                result = await ProcessorRegistry.submit(
                    request.collection, "content", {"id": request.id, "collection": request.collection}
                )
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

            return pb.ContentResponse(
                collection=request.collection,
                id=request.id,
                text=result["text"],
                cached=bool(result.get("cached", False)),
                timestamp=float(result.get("timestamp", 0.0)),
            )
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
                pipeline = SearchPipeline.from_string(
                    request.pipeline,
                    collection,
                    runtime_kwargs=runtime_kwargs,
                )
            except (ValueError, RuntimeError) as e:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"{type(e).__name__}: {e}")
                return pb.PipelineResponse()

            result = await pipeline.run(request.query)

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
