"""Pipeline DSL parser for RoutIR search pipelines.

The pipeline DSL composes retrieval and reranking services into multi-stage,
parallel, or query-expansion pipelines using a concise string syntax.

**Syntax reference**::

    # Single service — retrieve top 20
    "bm25%20"

    # Sequential — retrieve 100, rerank to top 20
    "bm25%100 >> cross-encoder%20"

    # Parallel retrieval + fusion — dense and sparse run concurrently, fused by RRF
    "{dense, sparse}RRF%100"

    # Query expansion — expander generates sub-queries, each runs the parallel branches
    "expander{dense, sparse}RRF%100"

    # Alias — name a stage so runtime_kwargs can target it by alias
    "dense[d]%100 >> cross-encoder%20"

**Grammar operators**:

* ``service%N`` — call *service*, keep top *N* results.
* ``service[alias]`` — give this call the name *alias* for ``runtime_kwargs`` targeting.
* ``A >> B`` — sequential: run A, pass output to B (B auto-assigned role ``"rerank"``).
* ``{A, B}Merger`` — parallel: run A and B concurrently, fuse with Merger.
* ``Expander{A, B}Merger`` — query expansion: Expander produces sub-queries, each
  dispatched to A and B, all results fused by Merger.

The module-level :data:`parser` singleton parses DSL strings into
:class:`PipelineComponent` AST nodes consumed by :class:`~routir.pipeline.SearchPipeline`.
"""

from dataclasses import dataclass
from typing import List, Optional, Union

from lark import Lark, Transformer


PIPELINE_GRAMMAR = r"""
    ?start: seq

    seq: stage (">>" stage)*              // sequential chain: A >> B >> C

    stage: (parallel_seq | system_call)

    system_call: NAME alias? ("%" NUMBER)?  // service[alias]%limit

    alias: ("[" NAME "]")                   // [my-alias]

    seq_list: seq ("," seq)*                // comma-separated parallel branches

    parallel_seq: system_call? "{" seq_list "}" system_call
                  // [expander]{branch1, branch2}merger

    NAME: /[A-Za-z][A-Za-z0-9_\-]*/
    NUMBER: /[0-9]+/

    %import common.WS
    %ignore WS
"""

# TODO: adding `:collection` into grammar to specify where to get the content


# Data classes to represent the AST nodes
@dataclass
class SystemCall:
    """An AST node representing a single service call in the pipeline.

    Attributes:
        name (str): Service name as registered in ``ProcessorRegistry``.
        alias (str): Label used to target this stage with ``runtime_kwargs``.
            Defaults to ``name`` when not specified with the ``[alias]`` syntax.
        limit (int or None): Maximum results to return (the ``%N`` suffix).
            ``None`` means no explicit limit; the service uses its own default.
        role (str): Role assigned during pipeline construction.  One of:

            * ``"search"`` — first-stage retrieval; receives the raw query.
            * ``"rerank"`` — later stage; receives previous results and document
              text fetched from the collection.
            * ``"expander"`` — generates sub-queries for parallel execution.
            * ``"merger"`` — fuses multiple ranked lists into one final ranking.
    """

    name: str
    alias: Optional[str] = None
    limit: Optional[int] = None
    role: Optional[str] = "search"

    def __post_init__(self):
        if self.alias is None:
            self.alias = self.name

    @property
    def all_calls(self):
        return set([self])

    def __hash__(self):
        return (self.name, self.alias, self.limit).__hash__()

    def as_role(self, role: str):
        return SystemCall(self.name, self.alias, self.limit, role)


@dataclass
class CallSequence:
    """An AST node for a sequential chain of pipeline stages (``A >> B >> C``).

    During construction, all stages after the first are automatically assigned
    role ``"rerank"``: they receive the previous stage's scored results together
    with document text fetched from the collection and are expected to rescore them.

    Attributes:
        stages (list): Ordered list of :class:`SystemCall` or
            :class:`ParallelCallSequences` nodes executed left-to-right.
            ``stages[0]`` has role ``"search"``; all later stages have
            role ``"rerank"``.
    """

    stages: List[Union[SystemCall, "ParallelCallSequences"]]

    def __post_init__(self):
        if len(self.stages) > 1:
            self.stages = [self.stages[0]] + [s.as_role("rerank") for s in self.stages[1:]]

    @property
    def all_calls(self):
        return set.union(*[s.all_calls for s in self.stages])

    def as_role(self, role: str):
        assert role in ["search", "rerank"]
        return CallSequence([self.stages[0].as_role(role), *self.stages[1:]])


@dataclass
class ParallelCallSequences:
    """An AST node for parallel retrieval with a merger (``{A, B}Merger``).

    Runs multiple retrieval branches concurrently and fuses their results.
    Optionally preceded by a query-expansion service (``E{A, B}Merger``).

    Attributes:
        sequences (list[CallSequence]): Independent retrieval pipelines run in
            parallel; each is a :class:`CallSequence`.
        merger (SystemCall): Service that fuses the parallel results.  Its role
            is automatically set to ``"merger"``; it must implement
            :meth:`~routir.models.abstract.Engine.fuse_batch`.
        expander (SystemCall or None): Optional query-expansion service. When
            present, its role is ``"expander"`` and it must implement
            :meth:`~routir.models.abstract.Engine.decompose_query_batch`. The
            expanded sub-queries are each dispatched to every sequence, and all
            results are merged by ``merger``.  When ``None``, the original query
            is sent directly to all sequences.
    """

    sequences: List[CallSequence]
    merger: SystemCall
    expander: Optional[SystemCall] = None

    def __post_init__(self):
        if self.expander is not None:
            self.expander.role = "expander"
        # else:
        #     self.sequences = [ s.as_role('rerank') for s in self.sequences ]
        self.merger.role = "merger"

    @property
    def all_calls(self):
        return set.union(
            set() if self.expander is None else self.expander.all_calls,
            self.merger.all_calls,
            *[s.all_calls for s in self.sequences],
        )

    def as_role(self, role: str):
        if self.expander is not None:
            # TODO: better handle this but we shouldn't support reranking existing
            # ranked list with expanded queries
            return self
        else:
            assert role in ["search", "rerank"]
            return ParallelCallSequences(sequences=[s.as_role(role) for s in self.sequences], merger=self.merger)


PipelineComponent = Union[SystemCall, CallSequence, ParallelCallSequences]
"""Type alias for any top-level pipeline AST node."""


class PipelineTransformer(Transformer):
    """Lark ``Transformer`` that converts a parse tree into RoutIR AST nodes.

    Each method corresponds to a grammar rule and returns the appropriate
    :class:`SystemCall`, :class:`CallSequence`, or
    :class:`ParallelCallSequences` instance.  Used internally by the
    :data:`parser` singleton; not normally called directly.
    """
    def seq(self, stages):
        if len(stages) == 1:
            return stages[0].as_role("search")
        return CallSequence(stages=stages)

    def system_call(self, tokens: List[str]):
        name, alias, limit = tokens[0], None, None
        if len(tokens) == 3:
            name, alias, limit = tokens
        elif len(tokens) == 2:
            if tokens[1].isdigit():
                name, limit = tokens
            else:
                name, alias = tokens

        return SystemCall(name=str(name), alias=alias, limit=int(limit) if limit is not None else None)

    def alias(self, tokens):
        return str(tokens[0])

    def parallel_seq(self, tokens):
        if len(tokens) == 3:
            return ParallelCallSequences(expander=tokens[0], sequences=tokens[1], merger=tokens[2])
        return ParallelCallSequences(sequences=tokens[0], merger=tokens[1])

    def stage(self, tokens):
        return tokens[0]

    def seq_list(self, tokens):
        return tokens


parser = Lark(PIPELINE_GRAMMAR, parser="lalr", transformer=PipelineTransformer())
