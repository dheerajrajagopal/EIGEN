"""Collection of tests for data generation
"""
from src.data.creation.influence_graph import InfluenceGraph
from src.data.creation.influence_graph import WhatIfExample
from src.data.creation.influence_graph import Rels

test_graph = {
    "para_id": "1",
    "prompt": "",
    "paragraph": "PARA",
    "para_outcome_accelerate": "MORE",
    "para_outcome_decelerate": "LESS",
    "Y_is_outcome": "",
    "X": "X",
    "Y": "Y",
    "W": [
        "W1",
        "W2"
    ],
    "U": [
        "U1",
        "U2"
    ],
    "Z": [
        "Z"
    ],
    "V": [
        "V"
    ],
    "Y_affects_outcome": "more",
    "graph_id": "1"
}
"""
true_all_paths = {
    WhatIfExample(src='V', reln='HELPS', dest='LESS'),
    WhatIfExample(src='V', reln='HURTS', dest='MORE'), 
    WhatIfExample(src='V', reln='HELPS', dest='W1'), 
    WhatIfExample(src='V', reln='HELPS', dest='W2'), 
    WhatIfExample(src='V', reln='HURTS', dest='X'), 
    WhatIfExample(src='V', reln='HURTS', dest='Y'), 
    WhatIfExample(src='X', reln='HURTS', dest='LESS'), 
    WhatIfExample(src='X', reln='HELPS', dest='MORE'), 
    WhatIfExample(src='X', reln='HURT_BY', dest='V'), 
    WhatIfExample(src='X', reln='HURTS', dest='W1'), 
    WhatIfExample(src='X', reln='HURTS', dest='W2'), 
    WhatIfExample(src='X', reln='HELPS', dest='Y'), 
    WhatIfExample(src='X', reln='HELPED_BY', dest='Z'), 
    WhatIfExample(src='Y', reln='HURTS', dest='LESS'), 
    WhatIfExample(src='Y', reln='HELPS', dest='MORE'), 
    WhatIfExample(src='Y', reln='HURT_BY', dest='U1'), 
    WhatIfExample(src='Y', reln='HURT_BY', dest='U2'), 
    WhatIfExample(src='Y', reln='HURT_BY', dest='V'), 
    WhatIfExample(src='Y', reln='HELPED_BY', dest='X'), 
    WhatIfExample(src='Y', reln='HELPED_BY', dest='Z'), 
    WhatIfExample(src='Z', reln='HURTS', dest='LESS'), 
    WhatIfExample(src='Z', reln='HELPS', dest='MORE'), 
    WhatIfExample(src='Z', reln='HURTS', dest='W1'), 
    WhatIfExample(src='Z', reln='HURTS', dest='W2'), 
    WhatIfExample(src='Z', reln='HELPS', dest='X'), 
    WhatIfExample(src='Z', reln='HELPS', dest='Y'), 
    WhatIfExample(src='U1', reln='HELPS', dest='LESS'), 
    WhatIfExample(src='U1', reln='HURTS', dest='MORE'), 
    WhatIfExample(src='U1', reln='HURTS', dest='Y'), 
    WhatIfExample(src='U2', reln='HELPS', dest='LESS'), 
    WhatIfExample(src='U2', reln='HURTS', dest='MORE'), 
    WhatIfExample(src='U2', reln='HURTS', dest='Y'), 
    WhatIfExample(src='W1', reln='HELPS', dest='LESS'), 
    WhatIfExample(src='W1', reln='HURTS', dest='MORE'), 
    WhatIfExample(src='W1', reln='HELPED_BY', dest='V'), 
    WhatIfExample(src='W1', reln='HURT_BY', dest='X'), 
    WhatIfExample(src='W1', reln='HURT_BY', dest='Z'), 
    WhatIfExample(src='W2', reln='HELPS', dest='LESS'), 
    WhatIfExample(src='W2', reln='HURTS', dest='MORE'), 
    WhatIfExample(src='W2', reln='HELPED_BY', dest='V'), 
    WhatIfExample(src='W2', reln='HURT_BY', dest='X'), 
    WhatIfExample(src='W2', reln='HURT_BY', dest='Z'), 
    WhatIfExample(src='LESS', reln='HELPED_BY', dest='U1'), 
    WhatIfExample(src='LESS', reln='HELPED_BY', dest='U2'), 
    WhatIfExample(src='LESS', reln='HELPED_BY', dest='V'), 
    WhatIfExample(src='LESS', reln='HELPED_BY', dest='W1'), 
    WhatIfExample(src='LESS', reln='HELPED_BY', dest='W2'), 
    WhatIfExample(src='LESS', reln='HURT_BY', dest='X'), 
    WhatIfExample(src='LESS', reln='HURT_BY', dest='Y'), 
    WhatIfExample(src='LESS', reln='HURT_BY', dest='Z'), 
    WhatIfExample(src='MORE', reln='HURT_BY', dest='U1'), 
    WhatIfExample(src='MORE', reln='HURT_BY', dest='U2'), 
    WhatIfExample(src='MORE', reln='HURT_BY', dest='V'), 
    WhatIfExample(src='MORE', reln='HURT_BY', dest='W1'), 
    WhatIfExample(src='MORE', reln='HURT_BY', dest='W2'), 
    WhatIfExample(src='MORE', reln='HELPED_BY', dest='X'), 
    WhatIfExample(src='MORE', reln='HELPED_BY', dest='Y'), 
    WhatIfExample(src='MORE', reln='HELPED_BY', dest='Z'), 
    WhatIfExample(src='V HURTS X', reln='HURTS', dest='W1'), 
    WhatIfExample(src='V HURTS X', reln='HURTS', dest='W2'), 
    WhatIfExample(src='V HURTS X', reln='HELPS', dest='Y'), 
    WhatIfExample(src='X HELPS Y', reln='HURTS', dest='LESS'), 
    WhatIfExample(src='X HELPS Y', reln='HELPS', dest='MORE'), 
    WhatIfExample(src='Z HELPS X', reln='HURTS', dest='W1'), 
    WhatIfExample(src='Z HELPS X', reln='HURTS', dest='W2'), 
    WhatIfExample(src='Z HELPS X', reln='HELPS', dest='Y'), 
    WhatIfExample(src='U1 HURTS Y', reln='HURTS', dest='LESS'), 
    WhatIfExample(src='U1 HURTS Y', reln='HELPS', dest='MORE'), 
    WhatIfExample(src='U2 HURTS Y', reln='HURTS', dest='LESS'), 
    WhatIfExample(src='U2 HURTS Y', reln='HELPS', dest='MORE'), 
    WhatIfExample(src='X HURTS W1', reln='HELPS', dest='LESS'), 
    WhatIfExample(src='X HURTS W1', reln='HURTS', dest='MORE'), 
    WhatIfExample(src='X HURTS W2', reln='HELPS', dest='LESS'), 
    WhatIfExample(src='X HURTS W2', reln='HURTS', dest='MORE'), 
    WhatIfExample(src='V HURTS X HELPS Y', reln='HURTS', dest='LESS'), 
    WhatIfExample(src='V HURTS X HELPS Y', reln='HELPS', dest='MORE'), 
    WhatIfExample(src='V HURTS X HURTS W1', reln='HELPS', dest='LESS'), 
    WhatIfExample(src='V HURTS X HURTS W1', reln='HURTS', dest='MORE'), 
    WhatIfExample(src='V HURTS X HURTS W2', reln='HELPS', dest='LESS'), 
    WhatIfExample(src='V HURTS X HURTS W2', reln='HURTS', dest='MORE'), 
    WhatIfExample(src='Z HELPS X HELPS Y', reln='HURTS', dest='LESS'), 
    WhatIfExample(src='Z HELPS X HELPS Y', reln='HELPS', dest='MORE'), 
    WhatIfExample(src='Z HELPS X HURTS W1', reln='HELPS', dest='LESS'), 
    WhatIfExample(src='Z HELPS X HURTS W1', reln='HURTS', dest='MORE'), 
    WhatIfExample(src='Z HELPS X HURTS W2', reln='HELPS', dest='LESS'), 
    WhatIfExample(src='Z HELPS X HURTS W2', reln='HURTS', dest='MORE')
}
"""

def test_bfs():
    ig = InfluenceGraph(test_graph)
    x_paths = ig.get_paths("X")
    z_paths = ig.get_paths("Z")
    print(z_paths)
    predicted_z_whatif_examples = ig.make_whatif_start_finish(z_paths, add_reversed=True)
    print(predicted_z_whatif_examples)
    return
    actual_z_whatif_examples = {WhatIfExample(src='Z', reln='HURTS', dest='W1'),
                                WhatIfExample(src='Z', reln='HELPS', dest='Y'),
                                WhatIfExample(
                                    src='Z', reln='HURTS', dest='LESS'),
                                WhatIfExample(
                                    src='Z', reln='HURTS', dest='W2'),
                                WhatIfExample(src='Z', reln='HELPS', dest='X'),
                                WhatIfExample(src='Z', reln='HELPS', dest='MORE')}
    assert predicted_z_whatif_examples == actual_z_whatif_examples

    actual_z_whatif_examples_with_reverse = set()
    for e in actual_z_whatif_examples:
        actual_z_whatif_examples_with_reverse.add(e)
        rev_reln = Rels.HELPED_BY if e.reln == Rels.HELPS else Rels.HURT_BY
        actual_z_whatif_examples_with_reverse.add(
            WhatIfExample(src=e.dest, dest=e.src, reln=rev_reln))
    assert actual_z_whatif_examples_with_reverse == ig.make_whatif_start_finish(
        z_paths, add_reversed=True)

    predicted_z_whatif_examples_all = ig.make_whatif_entire_path(z_paths)
    actual_z_whatif_examples_all = {WhatIfExample(src='Z', reln='HELPS', dest='X'),
                                    WhatIfExample(
                                        src='Z HELPS X', reln='HURTS', dest='W1'),
                                    WhatIfExample(
                                        src='Z HELPS X', reln='HURTS', dest='W2'),
                                    WhatIfExample(
                                        src='Z HELPS X', reln='HELPS', dest='Y'),
                                    WhatIfExample(
                                        src='Z HELPS X HELPS Y', reln='HELPS', dest='MORE'),
                                    WhatIfExample(
                                        src='Z HELPS X HELPS Y', reln='HURTS', dest='LESS'),
                                    WhatIfExample(
                                        src='Z HELPS X HURTS W1', reln='HURTS', dest='MORE'),
                                    WhatIfExample(
                                        src='Z HELPS X HURTS W1', reln='HELPS', dest='LESS'),
                                    WhatIfExample(
                                        src='Z HELPS X HURTS W2', reln='HELPS', dest='LESS'),
                                    WhatIfExample(src='Z HELPS X HURTS W2', reln='HURTS', dest='MORE')}
    assert predicted_z_whatif_examples_all == actual_z_whatif_examples_all

    print(ig.get_examples())


if __name__ == "__main__":
    test_bfs()
