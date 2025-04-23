from collections import defaultdict, deque
from typing import Dict, Set, Tuple, List

class Automata:
    def __init__(self, states: Set[int], alphabet: Set[str],
                 transition: Dict[Tuple[int, str], int]):
        self.states = states
        self.alphabet = alphabet
        self.transition = transition  # mapping: (state, symbol) -> state

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

    def __str__(self):
        # Header information: start state and final states.
        lines = [
            "DFA:",
            f"  Start State: {self.start}",
            f"  Final States: {sorted(list(self.final))}"
        ]
        return '\n'.join(lines) + super().__str__()

class DoubleStartDFA(Automata):
    def __init__(self, states: Set, alphabet: Set,
                 transition: Dict,
                 start1: int, start2: int, final: Set):
        super().__init__(states, alphabet, transition)
        self.start1 = start1
        self.start2 = start2
        self.final = final

    def __str__(self):
        # Header information: start state and final states.
        lines = [
            "Double Start DFA:",
            f"  Start State 1: {self.start1}",
            f"  Start State 2: {self.start2}"
            f"  Final States: {sorted(list(self.final))}"
        ]
        return '\n'.join(lines) + super().__str__()

class MultiFinalDFA(Automata):
    def __init__(self, states: Set, alphabet: Set,
                 transition: Dict,
                 start: int, partition: Dict):
        super().__init__(states, alphabet, transition)
        self.start = start
        self.partition = partition # mapping: state -> which partition it is in (int)
    
    def __str__(self):
        # Header information: start state and final states.
        lines = [
            "MultiFinal DFA:",
            f"  Start State: {self.start}",
            f"  State Partition: {self.partition}", ''
        ]
        return '\n'.join(lines) + super().__str__()


def convert_double_start_to_double_final_dfa(dfa: DoubleStartDFA) -> MultiFinalDFA:
    # Every pair of states is a state in the new DFA
    new_states = {(p,q) for p in dfa.states for q in dfa.states}

    new_transition = {}
    for p, q in new_states:
        for c in dfa.alphabet:
            new_transition[((p,q), c)] = (dfa.transition[(p, c)], dfa.transition[(q, c)])

    new_start = (dfa.start1, dfa.start2)

    partition = {
        (p,q): len(set(pair) & dfa.final) for pair in new_states
    }

    return MultiFinalDFA(new_states, dfa.alphabet, new_transition, new_start, partition)

def hopcroft_minimization(dfa: DFA) -> List[Set[int]]:
    """
    Performs DFA minimization using Hopcroft's algorithm.
    Returns a list of sets, where each set is an equivalence class of states.
    """
    # Initialize partition: final vs. non-final states.
    P = [dfa.final, dfa.states - dfa.final]
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

def build_minimized_dfa(dfa: DFA, partitions: List[Set[int]]) -> DFA:
    """
    Given the original DFA and its partitions (equivalence classes), builds a new minimized DFA.
    """
    # Map each original state to the partition (block) it belongs to.
    state_to_block = {}
    for i, block in enumerate(partitions):
        for state in block:
            state_to_block[state] = i

    new_states = set(range(len(partitions)))
    new_start = state_to_block[dfa.start]
    new_final = set()
    new_transition = {}

    # For each block, use one representative state (any from the block) to define transitions.
    representatives = {i: next(iter(block)) for i, block in enumerate(partitions)}
    
    for block_id, rep in representatives.items():
        # If the representative is accepting, then the whole block is accepting.
        if rep in dfa.final:
            new_final.add(block_id)
        for ch in dfa.alphabet:
            target = dfa.transition[(rep, ch)]
            new_transition[(block_id, ch)] = state_to_block[target]

    return DFA(new_states, dfa.alphabet, new_transition, new_start, new_final)

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

if __name__ == "__main__":
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
