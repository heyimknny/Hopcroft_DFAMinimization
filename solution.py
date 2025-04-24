from collections import defaultdict, deque
from typing import Dict, Set, Tuple, List, Optional
from collections import defaultdict

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
# Query Processing
##############################

def query_dfa(dfa: DFA, query: str) -> bool:
    """
    Simulates the DFA on the given query string.
    Returns True if the query reaches an accepting state, else False.
    """
    current_state = dfa.start
    for ch in query:
        if ch not in dfa.alphabet:
            raise ValueError(f'Unrecognized character in DFA: {ch}')
        current_state = dfa.transition[(current_state, ch)]
        if current_state in dfa.final:
            return True
    return False

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
    print(n)
    for _ in range(n*len(alphabet)):
        state, ch, next_state = input().strip().split(' ')
        state = int(state)
        next_state = int(next_state)
        transition[(state, ch)] = next_state
    
    # Next line are start states
    starts = [int(q) for q in input().split(' ')]
    start1, start2 = starts

    # Finally, the next line has the final states
    final = {int(q) for q in input().strip().split(' ')}

    return DoubleStartDFA(states, alphabet, transition, start1, start2, final)


##############################
# Main: Putting Everything Together
##############################

def main():
    # Example forbidden words and alphabet
    forbidden_words = ["bad", "hate", "spam", "over", "ridiculous", "one", "two", "three"]
    alphabet = set("abcdefghijklmnopqrstuvwxyz ")  # including space if needed
    
    # Build full DFA from forbidden words
    states, transitions, start_state, final_states = build_dfa(forbidden_words, alphabet)
    full_dfa = DFA(states, alphabet, transitions, start_state, final_states)
    # print(f'Full DFA:\n{full_dfa}')
    print(f'Full DFA states:\n{full_dfa.states}')
    
    # Minimize the DFA using Hopcroft's algorithm
    partitions = hopcroft_minimization(full_dfa)
    print(f'Partitions: {partitions}')
    minimized_dfa = build_minimized_dfa(full_dfa, partitions)
    # print(f'Min DFA:\n{minimized_dfa}')
    print(f'Min DFA states:\n{minimized_dfa.states}')

    # Example queries
    queries = [
        "I had a bad day",
        "This is absolutely great",
        "I really hate when it rains",
        "No issues here"
    ]
    
    print("Forbidden Words:", forbidden_words)
    print("\nQuery Results:")
    for q in queries:
        result = query_dfa(minimized_dfa, q.lower())  # use lower-case for consistency with alphabet
        print(f"'{q}': {result}")

if __name__ == "__main__":
    main()
