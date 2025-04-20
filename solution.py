from collections import defaultdict, deque
from typing import Dict, Set, Tuple, List

##############################
# Part 1. Build the Complete DFA for Forbidden Substrings
##############################

class TrieNode:
    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.fail: TrieNode = None  # failure link
        self.output: bool = False   # True if any forbidden word ends here
        self.state_id: int = None   # later assigned

def build_trie(forbidden_words: List[str], alphabet: Set[str]) -> TrieNode:
    """Creates a trie of forbidden words."""
    root = TrieNode()
    for word in forbidden_words:
        node = root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.output = True  # mark end of a forbidden word
    return root

def compute_failure_links(root: TrieNode, alphabet: Set[str]) -> None:
    """Computes failure links to allow for a complete DFA (like in Aho-Corasick)."""
    queue = deque()
    # Set failure link for children of root to root
    for ch in alphabet:
        if ch in root.children:
            child = root.children[ch]
            child.fail = root
            queue.append(child)
        else:
            # Implicitly add missing transitions to root
            root.children[ch] = root

    while queue:
        current = queue.popleft()
        # Propagate the output flag along the failure links:
        if current.fail and current.fail.output:
            current.output = True
        for ch in alphabet:
            if ch in current.children:
                child = current.children[ch]
                # The failure link of the child is the state you'd get from the failure of current
                f = current.fail
                # Follow failure links until we find a node with this transition
                while ch not in f.children and f is not None:
                    f = f.fail
                child.fail = f.children[ch] if f and ch in f.children else root
                queue.append(child)
            else:
                # Define missing transition: use current.fail's transition
                current.children[ch] = current.fail.children[ch]

def assign_state_ids(root: TrieNode) -> List[TrieNode]:
    """Traverses the automaton (BFS) and assigns a unique state id to each node."""
    queue = deque([root])
    seen = set()
    state_list = []
    root.state_id = 0
    seen.add(id(root))
    state_list.append(root)
    
    while queue:
        node = queue.popleft()
        for child in node.children.values():
            if id(child) not in seen:
                seen.add(id(child))
                child.state_id = len(state_list)
                state_list.append(child)
                queue.append(child)
    return state_list

def build_dfa(forbidden_words: List[str], alphabet: Set[str]) -> Tuple[Set[int], Dict[Tuple[int, str], int], int, Set[int]]:
    """
    Builds a complete DFA that recognizes any string containing a forbidden substring.
    Returns a tuple (states, transitions, start_state, final_states).
    """
    # Build the trie and compute failure links
    root = build_trie(forbidden_words, alphabet)
    compute_failure_links(root, alphabet)
    state_list = assign_state_ids(root)
    
    states = set(range(len(state_list)))
    transitions = {}
    final_states = set()
    
    for node in state_list:
        sid = node.state_id
        # If this node (or any by failure chain) is terminal, mark as accepting.
        if node.output:
            final_states.add(sid)
        for ch in alphabet:
            # Every node now has a defined transition by construction.
            next_state = node.children[ch].state_id
            transitions[(sid, ch)] = next_state
            
    start_state = root.state_id
    return states, transitions, start_state, final_states

##############################
# Part 2. DFA Minimization via Hopcroft's Algorithm
##############################

class DFA:
    def __init__(self, states: Set[int], alphabet: Set[str],
                 transition: Dict[Tuple[int, str], int],
                 start: int, final: Set[int]):
        self.states = states
        self.alphabet = alphabet
        self.transition = transition  # mapping: (state, symbol) -> state
        self.start = start
        self.final = final

    def __str__(self):
        # Header information: start state and final states.
        lines = [
            "DFA:",
            f"  Start State: {self.start}",
            f"  Final States: {sorted(list(self.final))}",
            "  Transitions:"
        ]
        
        # List transitions in a sorted order for reproducibility.
        for state in sorted(self.states):
            for symbol in sorted(self.alphabet):
                # Look up the transition if it exists; else mark as missing.
                next_state = self.transition.get((state, symbol), None)
                lines.append(f"    δ({state}, '{symbol}') = {next_state}")
            lines.append('')
        return "\n".join(lines)
    
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
# Part 3. Query Processing
##############################

def query_dfa(dfa: DFA, query: str) -> bool:
    """
    Simulates the DFA on the given query string.
    Returns True if the query reaches an accepting state, else False.
    """
    current_state = dfa.start
    for ch in query:
        if ch not in dfa.alphabet:
            # If symbol not in alphabet, you might decide to skip it,
            # or alternatively, treat it as a dead symbol leading to a sink.
            continue
        current_state = dfa.transition[(current_state, ch)]
        if current_state in dfa.final:
            return True
    return False

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
