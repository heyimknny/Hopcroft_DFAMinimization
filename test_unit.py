# Bug Fixes
# 1. In DoubleStartDFA.__str__, a missing comma caused the "Start State 2" and "Final States" lines to concatenate. Fixed signature below.


# test_automata.py
import pytest
from solution import DFA, DoubleStartDFA, MultiFinalDFA, remove_unreachable_states, convert_double_start_to_double_final_dfa, hopcroft_minimization

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
    md = convert_double_start_to_double_final_dfa(dfa)
    # New start should be pair (0,1)
    assert md.start == (0, 1)
    # Partition for (0,1): one of the pair is final, so partition value == 1
    assert md.partition[(0,1)] == 1

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
