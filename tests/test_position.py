import networkx as nx
import pytest

from src.Graph.graph_constructor import construct_graph


@pytest.fixture(scope="module")
def graph() -> nx.DiGraph:
    return construct_graph()


def _find_single_node_by_description(graph: nx.DiGraph, description: str) -> int:
    matches = [node for node, data in graph.nodes(data=True) if data.get("description") == description]
    assert len(matches) == 1, f"Expected one node for description '{description}', found {len(matches)}"
    return matches[0]


def _find_edges_by_description(graph: nx.DiGraph, description: str):
    return [(u, v, data) for u, v, data in graph.edges(data=True) if data.get("description") == description]


def test_swap_players(graph: nx.DiGraph):
    # Legacy intent: swapping players should still represent the same move context.
    # Current equivalent: this is encoded on transitions as `swaps_players`.
    to_honey_edges = _find_edges_by_description(graph, "to honey")
    assert to_honey_edges, "Expected at least one 'to honey' transition"
    assert any(edge_data.get("swaps_players") for _, _, edge_data in to_honey_edges)


def test_procrustes_analysis(graph: nx.DiGraph):
    # Legacy intent: imanari and back step are not the same position.
    # Current equivalent: they are distinct graph nodes.
    imanari_node = _find_single_node_by_description(graph, "completed imanari roll")
    backstep_node = _find_single_node_by_description(graph, "back step pass")
    assert imanari_node != backstep_node


def test_imanari_honey(graph: nx.DiGraph):
    # Legacy intent: `to honey` starts from the completed imanari roll position.
    imanari_node = _find_single_node_by_description(graph, "completed imanari roll")
    to_honey_edges = _find_edges_by_description(graph, "to honey")
    assert to_honey_edges, "Expected at least one 'to honey' transition"
    assert any(start == imanari_node for start, _, _ in to_honey_edges)


def test_backstep_topfree(graph: nx.DiGraph):
    # Legacy intent: `top tries to free leg` starts from back step pass, while imanari is different.
    imanari_node = _find_single_node_by_description(graph, "completed imanari roll")
    backstep_node = _find_single_node_by_description(graph, "back step pass")
    top_free_edges = _find_edges_by_description(graph, "top tries\\nto free leg")

    assert imanari_node != backstep_node
    assert top_free_edges, "Expected at least one 'top tries\\nto free leg' transition"
    assert any(start == backstep_node for start, _, _ in top_free_edges)
