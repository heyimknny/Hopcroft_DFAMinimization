from collections import defaultdict, deque
from typing import Dict, Set, Tuple, List

# Define a DFA class for clarity
class DFA:
    def __init__(self, states: Set[int], alphabet: Set[str],
                 transition: Dict[Tuple[int, str], int],
                 start: int, final: Set[int]):
        self.states = states
        self.alphabet = alphabet
        self.transition = transition  # mapping: (state, symbol) -> state
        self.start = start
        self.final = final

def hopcroft_minimization(dfa: DFA) -> List[Set[int]]:
    """
    Performs DFA minimization using Hopcroft's algorithm.
    Returns a list of sets, where each set is an equivalence class of states.
    """
    # Initialize partition: accepting and non-accepting states
    P = [dfa.final, dfa.states - dfa.final]
    # Filter out empty sets if they exist
    P = [block for block in P if block]

    # Worklist - we use deque for BFS-like processing
    W = deque(P.copy())

    # A dictionary to track transitions in reverse for each symbol
    reverse_transitions = {a: defaultdict(set) for a in dfa.alphabet}
    for (s, a), t in dfa.transition.items():
        reverse_transitions[a][t].add(s)
    
    # Main loop: refine the partition
    while W:
        A = W.popleft()
        # For each input symbol
        for a in dfa.alphabet:
            # X: set of states that have transitions on symbol a to any state in A
            X = set()
            for state in A:
                X.update(reverse_transitions[a].get(state, set()))
            
            # For each block Y in partition P; we'll be refining this block with X.
            new_P = []
            for Y in P:
                intersection = Y & X
                difference = Y - X
                # If both parts are nonempty, then split Y into intersection and difference
                if intersection and difference:
                    new_P.extend([intersection, difference])
                    # If Y was in W, replace it by the two subsets
                    if Y in W:
                        W.remove(Y)
                        W.append(intersection)
                        W.append(difference)
                    else:
                        # Otherwise, add the smaller subset to W
                        if len(intersection) <= len(difference):
                            W.append(intersection)
                        else:
                            W.append(difference)
                else:
                    new_P.append(Y)
            P = new_P

    return P

# Example usage
if __name__ == "__main__":
    # Define states, alphabet, transitions, start, and final states.
    states = {0, 1, 2, 3, 4}
    alphabet = {'a', 'b'}
    # Representing transitions: (state, symbol) -> next state
    transitions = {
        (0, 'a'): 1,
        (0, 'b'): 2,
        (1, 'a'): 1,
        (1, 'b'): 3,
        (2, 'a'): 4,
        (2, 'b'): 2,
        (3, 'a'): 1,
        (3, 'b'): 3,
        (4, 'a'): 4,
        (4, 'b'): 2,
    }
    start_state = 0
    final_states = {1, 3}

    dfa = DFA(states, alphabet, transitions, start_state, final_states)
    minimized_partitions = hopcroft_minimization(dfa)

    print("Equivalence Classes of states (minimized partitions):")
    for eq_class in minimized_partitions:
        print(eq_class)
