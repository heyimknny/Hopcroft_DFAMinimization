from collections import defaultdict, deque
from typing import Dict, Set, Tuple, List, Optional
from collections import defaultdict
import sys, os

class Automata:
    def __init__(self, states: Set[int], alphabet: Set[str],
                 transition: Dict[Tuple[int, str], int]):
        self.states = states
        self.alphabet = alphabet
        self.transition = transition  # mapping: (state, symbol) -> state

    def relabel(self):
        self._state_to_id = {q: i for i, q in enumerate(self.states)}
        self.states = set(range(len(self.states)))
        self.transition = {(self._state_to_id[p], c): self._state_to_id[q]
                           for (p, c), q in self.transition.items()}
        return self

    def extended_transition(self, input_string: str, start_state: Optional[int] = None) -> Optional[int]:
        if start_state is None:
            raise ValueError("A start_state must be provided for Automata extended_transition.")
        state = start_state
        for symbol in input_string:
            if symbol not in self.alphabet:
                raise ValueError(f"Symbol '{symbol}' not in alphabet.")
            state = self.transition.get((state, symbol))
            if state is None:
                return None
        return state

    def __str__(self):
        lines = ["  Transitions:"]
        # List transitions in a sorted order for reproducibility.
        for state in sorted(self.states):
            for symbol in sorted(self.alphabet):
                # Look up the transition if it exists; else mark as missing.
                next_state = self.transition.get((state, symbol), None)
                lines.append(f"    δ({state}, '{symbol}') = {next_state}")
            lines.append('')
        return "\n".join(lines)

class DFA(Automata):
    def __init__(self, states: Set[int], alphabet: Set[str],
                 transition: Dict[Tuple[int, str], int],
                 start: int, final: Set[int]):
        super().__init__(states, alphabet, transition)
        self.start = start
        self.final = final

    def relabel(self):
        super().relabel()
        self.start = self._state_to_id[self.start]
        self.final = {self._state_to_id[q] for q in self.final}
        return self

    def extended_transition(self, input_string: str) -> Optional[int]:
        return super().extended_transition(input_string, self.start)

    def __str__(self):
        # Header information: start state and final states.
        lines = [
            "DFA:",
            f"  Start State: {self.start}",
            f"  Final States: {sorted(list(self.final))}"
        ]
        return '\n'.join(lines) + super().__str__()

class DoubleStartDFA(Automata):
    def __init__(self, states: Set[int], alphabet: Set[str],
                 transition: Dict[Tuple[int, str], int],
                 start1: int, start2: int, final: Set[int]):
        super().__init__(states, alphabet, transition)
        self.start1 = start1
        self.start2 = start2
        self.final = final

    def relabel(self):
        super().relabel()
        self.start1 = self._state_to_id[self.start1]
        self.start2 = self._state_to_id[self.start2]
        self.final = {self._state_to_id[q] for q in self.final}
        return self

    def extended_transition(self, input_string: str, start_no: int = 1) -> Optional[int]:
        if start_no == 1:
            return super().extended_transition(input_string, self.start1)
        elif start_no == 2:
            return super().extended_transition(input_string, self.start2)
        else:
            raise ValueError("start_no must be 1 or 2 for DoubleStartDFA.")

    def __str__(self):
        lines = [
            "Double Start DFA:",
            f"  Start State 1: {self.start1}",
            f"  Start State 2: {self.start2}",
            f"  Final States: {sorted(list(self.final))}"
        ]
        return "\n".join(lines) + super().__str__()

class MultiFinalDFA(Automata):
    def __init__(self, states: Set[int], alphabet: Set[str],
                 transition: Dict[Tuple[int, str], int],
                 start: int, partition: Dict[int, str]):
        super().__init__(states, alphabet, transition)
        self.start = start
        self.partition = partition # mapping: state -> which partition it is in

    def relabel(self):
        super().relabel()
        self.start = self._state_to_id[self.start]
        self.partition = {self._state_to_id[q]: part for q, part in self.partition.items()}
        return self

    def extended_transition(self, input_string: str) -> Optional[int]:
        return super().extended_transition(input_string, self.start)

    def get_partition_list(self) -> List[Set[int]]:
        return list(self.get_reversed_paritions().values())

    def get_reversed_paritions(self) -> dict:
        P_dict = defaultdict(set)
        for q, part in self.partition.items():
            P_dict[part].add(q)
        return P_dict

    def get_formatted_partitions(self) -> str:
        lines = []
        for part, states in self.get_reversed_paritions().items():
            lines.append(f'\t{part}: {states}')
        return '\n'.join(lines)

    def __str__(self):
        lines = [
            "MultiFinal DFA:",
            f"  Start State: {self.start}",
            f"  State Partition: \n{self.get_formatted_partitions()}", ''
        ]
        return '\n'.join(lines) + super().__str__()

def convert_double_start_to_multi_final_dfa(dfa: DoubleStartDFA) -> MultiFinalDFA:
    # Every pair of states is a state in the new DFA
    new_states = {(p,q) for p in dfa.states for q in dfa.states}

    new_transition = {}
    for p, q in new_states:
        for c in dfa.alphabet:
            new_transition[((p,q), c)] = (dfa.transition[(p, c)], dfa.transition[(q, c)])

    new_start = (dfa.start1, dfa.start2)

    name_of_part = ['Rejecting', 'Half-Accepting', 'Accepting']

    partition = {
        pair: name_of_part[sum(q in dfa.final for q in pair)] for pair in new_states
    }

    return MultiFinalDFA(new_states, dfa.alphabet, new_transition, new_start, partition)

def get_start_states(dfa: Automata):
    match dfa:
        case DFA() | MultiFinalDFA():
            return [dfa.start]
        case DoubleStartDFA():
            return [dfa.start1, dfa.start2]

def remove_unreachable_states(dfa: Automata) -> Automata:
    # 1. Find reachable states via DFS/BFS from start_state
    reachable = set()
    stack = get_start_states(dfa)
    while stack:
        q = stack.pop()
        if q in reachable:
            continue
        reachable.add(q)
        for a in dfa.alphabet:
            # get the next state (if defined)
            q_next = dfa.transition.get((q, a))
            if q_next is not None and q_next not in reachable:
                stack.append(q_next)

    # 2. Build trimmed components
    new_states = reachable
    new_final = set()
    new_partition = {}
    match dfa:
        case DFA() | DoubleStartDFA():
            new_final = dfa.final & reachable
        case MultiFinalDFA():
            new_partition = {q: part for q, part in dfa.partition.items() if q in reachable}

    # keep only transitions whose source is reachable
    new_transitions = {}
    for (q, a), q_next in dfa.transition.items():
        if q in reachable and q_next in reachable:
            new_transitions[(q, a)] = q_next

    match dfa:
        case DFA():
            return DFA(new_states, dfa.alphabet, new_transitions, dfa.start, new_final)
        case DoubleStartDFA():
            return DoubleStartDFA(new_states, dfa.alphabet, new_transitions, dfa.start1, dfa.start2, new_final)
        case MultiFinalDFA():
            return MultiFinalDFA(new_states, dfa.alphabet, new_transitions, dfa.start, new_partition)

def hopcroft_minimization(dfa: Automata) -> List[Set[int]]:
    """
    Performs DFA minimization using Hopcroft's algorithm.
    Returns a list of sets, where each set is an equivalence class of states.
    """
    # Initialize partition
    P = []
    match dfa:
        case MultiFinalDFA():
            P = dfa.get_partition_list()
        case DFA() | DoubleStartDFA():
            P = [dfa.final, dfa.states - dfa.final]
        case _:
            raise ValueError(f'Unrecognized dfa type')
    # Remove empty set if exists.
    P = [block for block in P if block]
    
    # Worklist for refining partitions.
    W = deque(P.copy())
    
    # Build reverse transitions: for each symbol and destination state, record source states.
    reverse_transitions = {a: defaultdict(set) for a in dfa.alphabet}
    for (s, a), t in dfa.transition.items():
        reverse_transitions[a][t].add(s)
    
    while W:
        A = W.popleft()
        for a in dfa.alphabet:
            # X: set of states with transitions on a that lead into A.
            X = set()
            for state in A:
                X.update(reverse_transitions[a].get(state, set()))
            new_P = []
            for Y in P:
                intersection = Y & X
                difference = Y - X
                if intersection and difference:
                    new_P.extend([intersection, difference])
                    if Y in W:
                        # Replace Y with intersection and difference in the worklist.
                        W.remove(Y)
                        W.append(intersection)
                        W.append(difference)
                    else:
                        # Append the smaller part to maintain efficiency.
                        if len(intersection) <= len(difference):
                            W.append(intersection)
                        else:
                            W.append(difference)
                else:
                    new_P.append(Y)
            P = new_P
    
    return P

def build_minimized_automata(dfa: Automata, partitions: List[Set[int]]) -> Automata:
    """
    Given the original DFA and its partitions (equivalence classes), builds a new minimized DFA.
    """
    # Map each original state to the partition (block) it belongs to.
    state_to_block = {}
    for i, block in enumerate(partitions):
        for state in block:
            state_to_block[state] = i

    new_states = set(range(len(partitions)))
    new_start: set
    new_start1: set
    new_start2: set
    match dfa:
        case DFA() | MultiFinalDFA():
            new_start = state_to_block[dfa.start]
        case DoubleStartDFA():
            new_start1 = state_to_block[dfa.start1]
            new_start2 = state_to_block[dfa.start2]
        case _:
            raise ValueError(f'Unrecognized dfa type')

    new_transition = {}

    # For each block, use one representative state (any from the block) to define transitions.
    representatives = {i: next(iter(block)) for i, block in enumerate(partitions)}
    
    for block_id, rep in representatives.items():
        for ch in dfa.alphabet:
            target = dfa.transition[(rep, ch)]
            new_transition[(block_id, ch)] = state_to_block[target]
    
    new_final: set
    new_partition: Dict
    match dfa:
        case DFA() | DoubleStartDFA():
            new_final = {block_id for block_id, rep in representatives.items() if rep in dfa.final}
        case MultiFinalDFA():
            new_partition = {block_id: dfa.partition[rep] for block_id, rep in representatives.items()}
        case _:
            raise ValueError(f'Unrecognized dfa type')

    result: Automata
    match dfa:
        case DFA():
            result = DFA(new_states, dfa.alphabet, new_transition, new_start, new_final) 
        case MultiFinalDFA():
            result = MultiFinalDFA(new_states, dfa.alphabet, new_transition, new_start, new_partition)
        case DoubleStartDFA():
            result = DoubleStartDFA(new_states, dfa.alphabet, new_transition, new_start1, new_start2, new_final)
        case _:
            raise ValueError(f'Unrecognized dfa type')

    return result

##############################
# Input Handling
##############################

def get_double_start_dfa_from_input() -> DoubleStartDFA:
    # First line is the number of states
    n = int(input())
    states = set(range(n))

    # Alphabet is always {a,b}
    alphabet = {'a', 'b'}

    # Transition is given line-by-line
    transition = {}
    for _ in range(n*len(alphabet)):
        state, ch, next_state = input().strip().split(' ')
        state = int(state)
        next_state = int(next_state)
        transition[(state, ch)] = next_state
    
    # Next line are start states
    starts = [int(q) for q in input().split(' ')]
    start1, start2 = starts

    # Finally, the next lines have the final states
    final_count = int(input())
    final = {int(input()) for _ in range(final_count)}

    return DoubleStartDFA(states, alphabet, transition, start1, start2, final)

# verify_outputs.py
# Usage: python3 verify_outputs.py [io_dir] [test_prefix] [true_prefix]
# verify_outputs.py
# Usage: python3 verify_outputs.py [io_dir] [test_prefix] [true_prefix]

from collections import deque

def load_mf_dfa(path: str) -> MultiFinalDFA:
    """
    Load a MultiFinalDFA from a serialized .out file in the format:
      n
      p c q    (one line per transition)
      ...
      start    (single int)
      r        (number of rejecting states)
      ...      (rejecting states)
      h        (number of half-accepting states)
      ...      (half-accepting states)
      a        (number of accepting states)
      ...      (accepting states)
    Partitions: 0=rejecting, 1=half-accepting, 2=accepting.
    """
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    it = iter(lines)

    # Number of states
    n = int(next(it))

    # Transitions: expect n * |alphabet| lines of 'p c q'
    transition = {}
    alphabet = {'a', 'b'}
    for _ in range(len(alphabet) * n):
        p, c, q = next(it)
        p, q = int(p), int(q)
        transition[(p, c)] = q
    
    start = int(next(it))
    
    partition = {}
    # rejecting states
    r = int(next(it))
    partition.update({int(next(it)): 'Rejecting' for _ in range(r)})
    # half-accepting states
    h = int(next(it))
    partition.update({int(next(it)): 'Half-Accepting' for _ in range(r)})
    # accepting states
    a = int(next(it))
    partition.update({int(next(it)): 'Accepting' for _ in range(r)})

    return MultiFinalDFA(set(range(n)), alphabet, transition, start, partition)


def equivalent_dfa(d1: MultiFinalDFA, d2: MultiFinalDFA) -> bool:
    """
    Check language equivalence of two DFAs via BFS on the product automaton.
    Accepting = partition > 0.
    """
    if d1.alphabet != d2.alphabet:
        print(f"Alphabet mismatch: {d1.alphabet} vs {d2.alphabet}")
        return False

    seen = set()
    queue = deque([(d1.start, d2.start)])
    while queue:
        s1, s2 = queue.popleft()
        if (s1, s2) in seen:
            continue
        seen.add((s1, s2))

        f1 = d1.partition[s1]
        f2 = d2.partition[s2]
        if f1 != f2:
            print(f"Distinguishing at ({s1},{s2}): {f1} vs {f2}")
            return False

        for a in sorted(d1.alphabet):
            t1 = d1.transition.get((s1, a))
            t2 = d2.transition.get((s2, a))
            if t1 is None or t2 is None:
                if t1 != t2:
                    print(f"Missing transition for '{a}' at ({s1},{s2}): {t1} vs {t2}")
                    return False
                continue
            queue.append((t1, t2))
    return True


def get_md_output_format(dfa: MultiFinalDFA) -> str:
    """
    Serialize this MultiFinalDFA into the standard output format:
    - Line 1: |states|
    - Next |states| * |alphabet| lines: "p c q"
    - Next line: single start state
    - Next line: number of Rejecting states
    - Next lines: each rejecting state on its own line
    - Next line: number of Half-Accepting states
    - Next lines: each half-accepting state on its own line
    - Next line: number of Accepting states
    - Next lines: each accepting state on its own line
    """
    # Number of states
    n = len(dfa.states)
    # Transitions
    lines = [str(n)]
    for p in sorted(dfa.states):
        for c in sorted(dfa.alphabet):
            q = dfa.transition.get((p, c))
            lines.append(f"{p} {c} {q}")
    # Start
    lines.append(str(dfa.start))

    parts = dfa.get_reversed_paritions()
    for part_name in ['Rejecting', 'Half-Accepting', 'Accepting']:
        part = parts[part_name]
        lines.append(str(len(part)))
        for p in part:
            lines.append(str(p))
    return "\n".join(lines)


def convert_ds_to_mf_minimized(double_dfa: DoubleStartDFA) -> MultiFinalDFA:
    dfa = remove_unreachable_states(double_dfa)
    partitions = hopcroft_minimization(dfa)
    dfa = build_minimized_automata(dfa, partitions)
    dfa = convert_double_start_to_multi_final_dfa(dfa)
    dfa = remove_unreachable_states(dfa)
    partitions = hopcroft_minimization(dfa)
    dfa = build_minimized_automata(dfa, partitions)
    return dfa

def full_pipeline():
    orig = get_double_start_dfa_from_input()
    minimized = convert_ds_to_mf_minimized(orig)
    print(get_md_output_format(minimized))

import time 

def timed_full_pipeline():
    start = time.perf_counter()        # high-res timer
    full_pipeline()
    end   = time.perf_counter()
    return end - start

def main():
    full_pipeline()

def write_to_out(in_path: str):
    out_path = in_path.replace('.in', '.out')
    # back up real stdin/stdout
    real_stdin, real_stdout = sys.stdin, sys.stdout
    time_elapsed = 0
    try:
        # open the .in for reading as stdin
        sys.stdin = open(in_path, 'r')
        # open the .out for writing as stdout
        sys.stdout = open(out_path, 'w')
        time_elapsed = timed_full_pipeline()        # this will read from stdin and write to stdout
    finally:
        # close the fds we opened
        sys.stdin.close()
        sys.stdout.close()
        # restore
        sys.stdin, sys.stdout = real_stdin, real_stdout
    print(f'{time_elapsed:.6f}s')

def write_all_in_to_out():
    for fn in os.listdir('io'):
        if fn.endswith('.in'):
            if any(str(n) in fn for n in [200, 300]):  #skip the long ones for now
                continue
            print(f'{fn:<20}', end=' | ', flush=True)
            write_to_out(os.path.join('io', fn))

if __name__ == '__main__':
    write_all_in_to_out()
