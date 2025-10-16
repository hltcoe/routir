from dataclasses import dataclass
from typing import List, Optional, Union

from lark import Lark, Transformer


PIPELINE_GRAMMAR = r"""
    ?start: seq

    seq: stage (">>" stage)*

    stage: (parallel_seq | system_call)

    system_call: NAME alias? ("%" NUMBER)?

    alias: ("[" NAME "]")

    seq_list: seq ("," seq)*

    parallel_seq: system_call? "{" seq_list "}" system_call 

    NAME: /[A-Za-z][A-Za-z0-9_\-]*/
    NUMBER: /[0-9]+/

    %import common.WS
    %ignore WS
"""

# TODO: adding `:collection` into grammar to specify where to get the content


# Data classes to represent the AST nodes
@dataclass
class SystemCall:
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


# Transformer to convert parse tree to AST
class PipelineTransformer(Transformer):
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
