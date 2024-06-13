import torch
from queue import Queue
from typing import Dict, List, Set

ROOT_NODE = 0

def _validate_level_dict(level_dict: Dict[int, List[int]])-> (int, int):
    """
    For a level, validate that all nodes are indexed from 
    [all_nodes_till_precious_level, all_nodes_till_precious_level+nodes_in_curr_level)
    """
    tree_depth = len(level_dict)
    nodes_counter = 0
    for lvl in range(0, tree_depth):
        nodes_in_level = len(level_dict[lvl])
        # Validate nodes from [nodes_counter,nodes_counter + nodes_in_level)
        expected_nodes = {i for i in range(nodes_counter, nodes_counter + nodes_in_level)}
        for node in level_dict[lvl]:
            assert node in expected_nodes, f"Node {node} not indexed correctly in a leveled order for level {lvl}"
            expected_nodes.remove(node)
        nodes_counter = nodes_counter + nodes_in_level
    return nodes_counter, tree_depth


def _validate_all_nodes_discovered(visited: Set[int], token_tree: Dict[int, List[int]])-> None:
    """
    Checks if the set of nodes discovered while using level order traversal from root node 0 is
    the complete set of nodes defined in the input token tree.
    """
    for k in token_tree.keys():
        assert k in visited, f"Invalid token tree with node {k}."


def validate_token_tree(token_tree: Dict[int, List[int]])-> (int, int):
    """
    Assume index 0 to be the root node (no incoming edges) and start level order traversal from it.
    Validate tree structure (not graph) while doing level order traversal
    Validate all nodes are discovered at the end of the level order traversal.
    Also validate with level order traversal that all nodes are in required order.
    """
    assert ROOT_NODE in token_tree, "Token tree does not have the root node indexed as 0"
    visited = set()
    level_dict = {}
    q = Queue(maxsize = 0)
    visited.add(ROOT_NODE)
    q.put((ROOT_NODE,0))
    level_dict[0] = [ROOT_NODE]
    while not q.empty():
        curr, lvl = q.get()
        if curr in token_tree:
            for child in token_tree[curr]:
                assert child not in visited, "Cycle/Graph found instead of a tree."
                visited.add(child)
                q.put((child, lvl+1))
                if lvl+1 not in level_dict:
                    level_dict[lvl+1] = []
                level_dict[lvl+1].append(child)
    _validate_all_nodes_discovered(visited, token_tree)
    return _validate_level_dict(level_dict)


def generate_attention_mask(token_tree: Dict[int, List[int]])->torch.Tensor:
    """
    Generate attention mask based on the token tree.
    """
    total_nodes, depth = validate_token_tree(token_tree)
    attn_mask = torch.zeros(total_nodes, total_nodes, dtype=torch.int32)
    buffer = []
    def populate_mask():
        top = buffer[-1]
        for node in buffer:
            attn_mask[top][node] = 1
    # DFS on a tree.
    def DFS(node: int):
        buffer.append(node)
        if node in token_tree:
            for child in token_tree[node]:
                DFS(child)
        populate_mask()
        buffer.pop()
    DFS(ROOT_NODE)
    return attn_mask
