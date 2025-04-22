from collections import defaultdict, deque
from typing import Dict, Set, Tuple, List

class DoubleStartDFA:
    def __init__(self, states: Set, alphabet: Set,
                 transition: Dict,
                 start1: int, start2: int, final: Set):
        self.states = states
        self.alphabet = alphabet
        self.transition = transition  # mapping: (state, symbol) -> state
        self.start1 = start1
        self.start2 = start2
        self.final = final

class DoubleFinalDFA:
    def __init__(self, states: Set, alphabet: Set,
                 transition: Dict,
                 start: int, half_final: Set, final: Set):
        self.states = states
        self.alphabet = alphabet
        self.transition = transition  # mapping: (state, symbol) -> state
        self.start = start
        self.half_final = half_final
        self.final = final

def convert_double_start_to_double_final_dfa(dfa: DoubleStartDFA) -> DoubleFinalDFA:
    # Every pair of states is a state in the new DFA
    new_states = {(p,q) for p in dfa.states for q in dfa.states}

    new_transition = {}
    for p, q in new_states:
        for c in dfa.alphabet:
            new_transition[((p,q), c)] = (dfa.transition[(p, c)], dfa.transition[(q, c)])

    new_start = (dfa.start1, dfa.start2)

    half_final = set()
    final = set()

    for p, q in new_states:
        amount_in_final = 0
        if p in dfa.final:
            amount_in_final += 1
        if q in dfa.final:
            amount_in_final += 1
        if amount_in_final == 1:
            half_final.add((p,q))
        elif amount_in_final == 2:
            final.add((p,q))

    return DoubleFinalDFA(new_states, dfa.alphabet, new_transition, new_start, half_final, final)
