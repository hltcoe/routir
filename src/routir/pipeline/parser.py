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

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from lark import Lark, Transformer


PIPELINE_GRAMMAR = r"""
    ?start: seq

    seq: stage (">>" stage)*              // sequential chain: A >> B >> C

    stage: (parallel_seq | system_call)

    system_call: NAME alias? view? ("%" NUMBER)?  // service[alias]@view%limit

    alias: ("[" NAME "]")                   // [my-alias]

    view: ("@" NAME)                        // @view-name

    seq_list: seq ("," seq)*                // comma-separated parallel branches

    parallel_seq: system_call? "{" seq_list "}" system_call
                  // [expander]{branch1, branch2}merger

    NAME: /[A-Za-z][A-Za-z0-9_\-]*/
    NUMBER: /[0-9]+/

    %import common.WS
    %ignore WS
"""


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

        view (str or None): View name selector (the ``@view`` suffix) used to
            pick which view of the collection's content service to fetch when
            this stage reranks.  ``None`` means fall back to the collection's
            ``default_view`` at run time.  Only meaningful for ``"rerank"``
            stages; ignored for ``search`` / ``merger`` / ``expander`` roles.
    """

    name: str
    alias: Optional[str] = None
    limit: Optional[int] = None
    role: Optional[str] = "search"
    view: Optional[str] = None

    def __post_init__(self):
        if self.alias is None:
            self.alias = self.name

    @property
    def all_calls(self):
        return set([self])

    def __hash__(self):
        return (self.name, self.alias, self.limit, self.view).__hash__()

    def as_role(self, role: str):
        return SystemCall(
            name=self.name, alias=self.alias, limit=self.limit,
            role=role, view=self.view,
        )


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


class _AliasToken(str):
    """String subclass tagging a parsed alias token to disambiguate transformer args."""


class _ViewToken(str):
    """String subclass tagging a parsed view token to disambiguate transformer args."""


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
        name = str(tokens[0])
        alias = None
        view = None
        limit = None
        for tok in tokens[1:]:
            if isinstance(tok, _AliasToken):
                alias = str(tok)
            elif isinstance(tok, _ViewToken):
                view = str(tok)
            else:
                # NUMBER token from grammar -> limit
                limit = int(tok)
        return SystemCall(name=name, alias=alias, view=view, limit=limit)

    def alias(self, tokens):
        return _AliasToken(str(tokens[0]))

    def view(self, tokens):
        return _ViewToken(str(tokens[0]))

    def parallel_seq(self, tokens):
        if len(tokens) == 3:
            return ParallelCallSequences(expander=tokens[0], sequences=tokens[1], merger=tokens[2])
        return ParallelCallSequences(sequences=tokens[0], merger=tokens[1])

    def stage(self, tokens):
        return tokens[0]

    def seq_list(self, tokens):
        return tokens


parser = Lark(PIPELINE_GRAMMAR, parser="lalr", transformer=PipelineTransformer())


def _apply_outer_limit(node: "PipelineComponent", limit: int) -> "PipelineComponent":
    """Replace the outermost (final-output) limit of a subtree.

    For a single :class:`SystemCall`, the limit replaces ``self.limit``.
    For a :class:`CallSequence`, the limit replaces the limit of the last stage
    (the stage that produces the final output).  For a
    :class:`ParallelCallSequences`, the limit replaces the merger's limit.
    """
    if isinstance(node, SystemCall):
        return SystemCall(name=node.name, alias=node.alias, limit=limit,
                          role=node.role, view=node.view)
    if isinstance(node, CallSequence):
        new_stages = list(node.stages)
        new_stages[-1] = _apply_outer_limit(new_stages[-1], limit)
        seq = CallSequence.__new__(CallSequence)
        seq.stages = new_stages
        return seq
    if isinstance(node, ParallelCallSequences):
        new_merger = SystemCall(name=node.merger.name, alias=node.merger.alias,
                                limit=limit, role="merger", view=node.merger.view)
        pcs = ParallelCallSequences.__new__(ParallelCallSequences)
        pcs.sequences = node.sequences
        pcs.merger = new_merger
        pcs.expander = node.expander
        return pcs
    raise TypeError(f"Unexpected pipeline AST node type: {type(node).__name__}")


def _propagate_view_to_leaves(node: "PipelineComponent", view: str) -> "PipelineComponent":
    """Set ``view`` on every SystemCall leaf in *node* that doesn't already
    have one of its own.

    Used by :func:`expand_aliases` to propagate a call-site ``@view`` into
    the alias body.  Leaves with their own ``@view`` are preserved
    (authorial intent in the alias body wins over the call-site override).

    Mergers and expanders inside a :class:`ParallelCallSequences` are
    skipped — they don't fetch document content, so a view annotation is
    meaningless there.
    """
    if isinstance(node, SystemCall):
        if node.view is not None:
            return node
        return SystemCall(name=node.name, alias=node.alias, limit=node.limit,
                          role=node.role, view=view)

    if isinstance(node, CallSequence):
        new_stages = [_propagate_view_to_leaves(s, view) for s in node.stages]
        seq = CallSequence.__new__(CallSequence)
        seq.stages = new_stages
        return seq

    if isinstance(node, ParallelCallSequences):
        new_sequences = [_propagate_view_to_leaves(s, view) for s in node.sequences]
        pcs = ParallelCallSequences.__new__(ParallelCallSequences)
        pcs.sequences = new_sequences
        pcs.merger = node.merger        # merger has no view of its own; skip
        pcs.expander = node.expander    # ditto
        return pcs

    raise TypeError(f"Unexpected pipeline AST node type: {type(node).__name__}")


def expand_aliases(node: "PipelineComponent", aliases: Dict[str, "PipelineComponent"]) -> "PipelineComponent":
    """Substitute alias :class:`SystemCall` nodes with their pre-parsed bodies.

    Walks the AST and, for each :class:`SystemCall` whose ``name`` is a key in
    ``aliases``, replaces it with a deep copy of the corresponding alias body.
    The original call's ``role`` is propagated into the substituted subtree via
    :meth:`as_role`, and the original call's ``limit`` (when set) is applied to
    the outermost (final-output) node via :func:`_apply_outer_limit`.

    Merger and expander positions inside a :class:`ParallelCallSequences` are
    leaf calls and are not substituted; aliases only apply at pipeline-stage
    or parallel-branch positions.

    Args:
        node: AST root to expand.
        aliases: Mapping of alias name to fully-expanded alias body AST.  When
            empty, ``node`` is returned unchanged.

    Returns:
        A new AST with all alias references substituted.  The input is not
        mutated.
    """
    if not aliases:
        return node

    if isinstance(node, SystemCall):
        if node.name in aliases:
            body = copy.deepcopy(aliases[node.name])
            body = body.as_role(node.role)
            if node.limit is not None:
                body = _apply_outer_limit(body, node.limit)
            if node.view is not None:
                body = _propagate_view_to_leaves(body, node.view)
            return body
        return node

    if isinstance(node, CallSequence):
        new_stages = [expand_aliases(s, aliases) for s in node.stages]
        seq = CallSequence.__new__(CallSequence)
        seq.stages = new_stages
        return seq

    if isinstance(node, ParallelCallSequences):
        new_sequences = [expand_aliases(s, aliases) for s in node.sequences]
        pcs = ParallelCallSequences.__new__(ParallelCallSequences)
        pcs.sequences = new_sequences
        pcs.merger = node.merger
        pcs.expander = node.expander
        return pcs

    raise TypeError(f"Unexpected pipeline AST node type: {type(node).__name__}")
