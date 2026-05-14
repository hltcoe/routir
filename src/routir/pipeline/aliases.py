"""Global registry for pre-defined pipeline aliases.

A *pipeline alias* is a name (e.g. ``ragtime2``) that stands in for a longer
pipeline DSL fragment (e.g. ``"{zho%100, rus%100, spa%100, eng%100}ScoreFusion"``).
At request time, the alias name in a user-supplied pipeline string is substituted
with the alias body, with the call-site ``%limit`` applied to the substituted
subtree's outer boundary.

Aliases are declared in the JSON config under ``pipeline_aliases`` and registered
once at server startup by :func:`~routir.config.load.load_config`.
"""

from typing import Dict, Set

from ..utils import logger
from .parser import PipelineComponent, expand_aliases, parser


class _PipelineAliasRegistry:
    """Singleton storing alias name -> fully-expanded AST.

    Also retains the original DSL string for each alias so the server can
    advertise alias definitions on the ``/avail`` endpoint.
    """

    def __init__(self):
        self._expanded: Dict[str, PipelineComponent] = {}
        self._source: Dict[str, str] = {}

    @property
    def expanded(self) -> Dict[str, PipelineComponent]:
        return self._expanded

    @property
    def source(self) -> Dict[str, str]:
        return dict(self._source)

    def has_alias(self, name: str) -> bool:
        return name in self._expanded

    def clear(self):
        self._expanded.clear()
        self._source.clear()

    def register_all(self, alias_strings: Dict[str, str], reserved_names: Set[str]):
        """Parse, validate, and register a batch of aliases.

        Aliases are resolved in topological order so that alias-of-alias
        references are flattened into a single AST.  Cycles raise ``ValueError``.

        Args:
            alias_strings: Mapping of alias name to its DSL body.
            reserved_names: Names that aliases must not shadow.  Typically the
                set of services callable from a pipeline DSL string (search,
                score, fuse, decompose_query roles); collections are excluded
                because they only appear in the ``collection`` request field,
                never inside the pipeline string.

        Raises:
            ValueError: If an alias name collides with ``reserved_names`` or if
                a dependency cycle is detected among aliases.
        """
        if not alias_strings:
            return

        collisions = set(alias_strings) & reserved_names
        if collisions:
            raise ValueError(
                f"pipeline_aliases names collide with registered services: {sorted(collisions)}"
            )

        parsed = {name: parser.parse(body) for name, body in alias_strings.items()}

        visiting: Set[str] = set()
        resolved: Dict[str, PipelineComponent] = {}

        def _resolve(name: str) -> PipelineComponent:
            if name in resolved:
                return resolved[name]
            if name in visiting:
                raise ValueError(f"Cycle detected in pipeline_aliases involving '{name}'")
            visiting.add(name)
            body = parsed[name]
            for call in body.all_calls:
                if call.name in parsed:
                    _resolve(call.name)
            resolved[name] = expand_aliases(body, resolved)
            visiting.remove(name)
            return resolved[name]

        for name in parsed:
            _resolve(name)

        self._expanded.update(resolved)
        self._source.update(alias_strings)
        for name, body in alias_strings.items():
            logger.info(f"Registered pipeline alias '{name}' = {body}")


PipelineAliasRegistry = _PipelineAliasRegistry()
