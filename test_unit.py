# Bug Fixes
# 1. In DoubleStartDFA.__str__, a missing comma caused the "Start State 2" and "Final States" lines to concatenate. Fixed signature below.

# test_automata.py
import sys
from io import StringIO
from solution import (
    get_double_start_dfa_from_input,
    convert_ds_to_mf_minimized
)
import itertools


# test_automata.py
import pytest
from solution import DFA, DoubleStartDFA, MultiFinalDFA, remove_unreachable_states, convert_double_start_to_multi_final_dfa, hopcroft_minimization

# 1. Test relabel for a simple DFA
def test_dfa_relabel():
    states = {'A', 'B'}
    alphabet = {'0'}
    transition = {('A','0'): 'B', ('B','0'): 'A'}
    dfa = DFA(states, alphabet, transition, 'A', {'A'})
    dfa.relabel()
    # After relabel, states should be {0,1}, start=0, final={0}
    assert dfa.states == {0, 1}
    assert dfa.transition == {(0, '0'): 1, (1, '0'): 0}

# 2. Test removal of unreachable states
def test_remove_unreachable_states():
    states = {0, 1, 2}
    alphabet = {'a'}
    transition = {(0,'a'): 1, (1,'a'): 0, (2,'a'): 2}
    dfa = DFA(states, alphabet, transition, 0, {1, 2})
    trimmed = remove_unreachable_states(dfa)
    # State 2 is unreachable from start 0; final={1,2} intersect reachable gives {1}
    assert set(trimmed.states) == {0, 1}
    assert trimmed.final == {1}
    assert (2, 'a') not in trimmed.transition

# 3. Test __str__ formatting for DoubleStartDFA
def test_double_start_str():
    states = {0, 1}
    alphabet = {'x'}
    transition = {(0,'x'): 1, (1,'x'): 0}
    dfa = DoubleStartDFA(states, alphabet, transition, 0, 1, {1})
    text = str(dfa)
    assert 'Start State 1: 0' in text
    assert 'Start State 2: 1' in text
    assert 'Final States: [1]' in text

# 4. Test conversion from DoubleStartDFA to MultiFinalDFA
def test_convert_double_start_to_double_final():
    states = {0, 1}
    alphabet = {'a'}
    transition = {(0,'a'): 0, (1,'a'): 1}
    dfa = DoubleStartDFA(states, alphabet, transition, 0, 1, {1})
    md = convert_double_start_to_multi_final_dfa(dfa)
    # New start should be pair (0,1)
    assert md.start == (0, 1)
    assert md.partition[(0,1)] == 'Half-Accepting'

# 5. Test Hopcroft minimization on a simple 2-state DFA
def test_hopcroft_minimization():
    states = {0, 1}
    alphabet = {'a'}
    transition = {(0,'a'): 0, (1,'a'): 1}
    dfa = DFA(states, alphabet, transition, 0, {0})
    parts = hopcroft_minimization(dfa)
    # Expect two blocks: {0} (final) and {1} (non-final)
    assert any(block == {0} for block in parts)
    assert any(block == {1} for block in parts)

# 6. Test minimization partitions for the provided example input
def test_minimization_example_partitions():
    # Build the DoubleStartDFA from the provided example
    states = set(range(8))
    alphabet = {'a','b'}
    transition = {
        (0,'a'):1, (0,'b'):2,
        (1,'a'):3, (1,'b'):4,
        (2,'a'):5, (2,'b'):6,
        (3,'a'):1, (3,'b'):4,
        (4,'a'):4, (4,'b'):5,
        (5,'a'):4, (5,'b'):5,
        (6,'a'):7, (6,'b'):7,
        (7,'a'):6, (7,'b'):6,
    }
    dfa = DoubleStartDFA(states, alphabet, transition, 0, 2, {4,5})
    trimmed = remove_unreachable_states(dfa)
    parts = hopcroft_minimization(trimmed)
    expected_blocks = [{4,5}, {6,7}, {0}, {2}, {1,3}]
    assert len(parts) == 5
    assert all(any(block == exp for block in parts) for exp in expected_blocks)

# 7. Test language equivalence via convert_ds_to_mf_minimized

def all_strings(alphabet, max_len):
    for length in range(max_len + 1):
        for prod in itertools.product(sorted(alphabet), repeat=length):
            yield ''.join(prod)


# 7. Parameterized test reading from input files for language equivalence
@pytest.mark.parametrize("input_file, max_len", [
    ("sample.in", 6),
    # Add more ("filename.in", max_len) pairs here
])
def test_language_equivalence(input_file, max_len, monkeypatch):
    # Redirect stdin to read from the given file
    f = open(input_file, 'r')
    monkeypatch.setattr(sys, 'stdin', f)

    # Build and transform
    orig = get_double_start_dfa_from_input()
    minimized = convert_ds_to_mf_minimized(orig)

    # For each string up to length 6, ensure acceptance matches
    for s in all_strings(orig.alphabet, max_len):
        part_names = ['Rejecting', 'Half-Accepting', 'Accepting']
        orig_acc = part_names[sum(orig.extended_transition(s, start_no=i) in orig.final for i in [1,2])]

        new_end = minimized.extended_transition(s)
        new_acc = minimized.partition[new_end]

        assert orig_acc == new_acc, (
            f"Mismatch on input '{s}': ``orig_acc={orig_acc}`` vs ``new_acc={new_acc}``"
        )
