"""Render AccFG's substructure hierarchy without atom indices or stdout capture."""

from __future__ import annotations

from typing import Any, Mapping


def render_functional_group_tree(groups: Mapping[str, Any], graph: Any) -> str:
    """Keep nested matches, with counts restricted to the current parent branch.

    AccFG's graph joins group *names*, even for unrelated matches. Expand it to
    actual matches before reducing edges, so one branch cannot hide another.
    All state is local: concurrent service calls must not redirect global stdout
    (as wrapping AccFG's print_fg_tree would do).
    """
    import networkx as nx

    if not groups:
        return "No functional groups matched by the configured AccFG definitions."
    matches_by_name = {
        name: {frozenset(match) for match in data["mapped_atoms"]}
        for name, data in graph.nodes(data=True)
    }
    hierarchy = nx.DiGraph()
    hierarchy.add_nodes_from(
        (name, match) for name, matches in matches_by_name.items() for match in matches
    )
    hierarchy.add_edges_from(
        ((parent, parent_match), (child, child_match))
        for parent, child in graph.edges
        for parent_match in matches_by_name[parent]
        for child_match in matches_by_name[child]
        if child_match <= parent_match
    )
    hierarchy = nx.transitive_reduction(hierarchy)
    lines: list[str] = []

    def visit(name: str, matches: set[frozenset[int]], prefix: str, last: bool) -> None:
        lines.append(f"{prefix}{'└──' if last else '├──'}{name}: {len(matches)}")
        child_matches_by_name: dict[str, set[frozenset[int]]] = {}
        for match in matches:
            for child, child_match in hierarchy.successors((name, match)):
                child_matches_by_name.setdefault(child, set()).add(child_match)
        children = sorted(child_matches_by_name.items(), key=lambda item: item[0].casefold())
        for index, (child, child_matches) in enumerate(children):
            visit(child, child_matches, prefix + ("   " if last else "│  "), index == len(children) - 1)

    roots = sorted(groups, key=str.casefold)
    for index, name in enumerate(roots):
        visit(name, {frozenset(match) for match in groups[name]}, "", index == len(roots) - 1)
    return "\n".join(lines)
