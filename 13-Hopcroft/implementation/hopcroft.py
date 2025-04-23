import csv
import os
from collections import deque, defaultdict

class DFA:
    def __init__(self, states, alphabet, transition_function, start_state, accept_states):
        self.states = states
        self.alphabet = alphabet
        self.tf = transition_function
        self.start = start_state
        self.accept = set(accept_states)

    def remove_unreachable_states(self):
        reachable = set()
        queue = deque([self.start])
        while queue:
            state = queue.popleft()
            if state not in reachable:
                reachable.add(state)
                for c in self.alphabet:
                    target = self.tf.get((state, c))
                    if target and target not in reachable:
                        queue.append(target)
        self.states = reachable
        self.accept &= reachable
        self.tf = {(s, c): t for (s, c), t in self.tf.items() if s in reachable and t in reachable}

    def make_total(self):
        trap = "__TRAP__"
        needs_trap = False
        for state in self.states:
            for symbol in self.alphabet:
                if (state, symbol) not in self.tf:
                    self.tf[(state, symbol)] = trap
                    needs_trap = True
        if needs_trap:
            self.states.add(trap)
            for symbol in self.alphabet:
                self.tf[(trap, symbol)] = trap

    def minimize(self):
        self.remove_unreachable_states()
        self.make_total()

        P = [self.accept, self.states - self.accept]
        W = deque(P.copy())

        while W:
            A = W.popleft()
            for c in self.alphabet:
                X = {s for s in self.states if self.tf.get((s, c)) in A}
                new_P = []
                for Y in P:
                    inter = X & Y
                    diff = Y - X
                    if inter and diff:
                        new_P.extend([inter, diff])
                        if Y in W:
                            W.remove(Y)
                            W.extend([inter, diff])
                        else:
                            W.append(inter if len(inter) <= len(diff) else diff)
                    else:
                        new_P.append(Y)
                P = new_P

        sorted_blocks = sorted([sorted(list(block)) for block in P], key=lambda b: b[0])
        block_map = {frozenset(block): i for i, block in enumerate(sorted_blocks)}
        representative = {state: block_map[frozenset(block)] for block in sorted_blocks for state in block}
        new_states = set(block_map.values())
        new_start = representative[self._find_block(self.start, P)]
        new_accept = {representative[s] for s in self.accept}
        new_tf = {}

        for block in P:
            rep = next(iter(block))
            for c in self.alphabet:
                target = self.tf.get((rep, c))
                if target:
                    target_block = self._find_block(target, P)
                    new_tf[(representative[rep], c)] = representative[target_block]

        return DFA(new_states, self.alphabet, new_tf, new_start, new_accept)

    def _find_block(self, state, partition):
        for block in partition:
            if state in block:
                return next(iter(block))
        return None

def parse_dfa_from_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        alphabet = reader.fieldnames[3:]
        states = set()
        tf = {}
        start_state = None
        accept_states = set()

        for row in rows:
            state = row["state"]
            states.add(state)
            if row["is_start"] == "1":
                start_state = state
            if row["is_accept"] == "1":
                accept_states.add(state)
            for sym in alphabet:
                tf[(state, sym)] = row[sym]

    return DFA(states, set(alphabet), tf, start_state, accept_states)

def write_dfa_to_csv(dfa, filename):
    rows = []
    state_data = defaultdict(dict)
    for (state, symbol), target in dfa.tf.items():
        state_data[state][symbol] = target

    for state in sorted(dfa.states):
        row = {
            "state": state,
            "is_start": "1" if state == dfa.start else "0",
            "is_accept": "1" if state in dfa.accept else "0"
        }
        for sym in sorted(dfa.alphabet):
            row[sym] = state_data[state].get(sym, "")
        rows.append(row)

    fieldnames = ["state", "is_start", "is_accept"] + sorted(dfa.alphabet)
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def compare_csv_files(file1, file2):
    def read_csv_as_dict(file):
        with open(file, 'r') as f:
            return list(csv.DictReader(f))
    return read_csv_as_dict(file1) == read_csv_as_dict(file2)

def run_test(index):
    base = "io"
    input_file = os.path.join(base, f"sample.in.{index}")
    expected_output_file = os.path.join(base, f"sample.out.{index}")
    generated_output_file = os.path.join(base, f"generated.out.{index}")

    print(f"Running Test {index}...")

    dfa = parse_dfa_from_csv(input_file)
    minimized = dfa.minimize()
    write_dfa_to_csv(minimized, generated_output_file)

    if compare_csv_files(expected_output_file, generated_output_file):
        print(f"✅ Test {index} passed.")
    else:
        print(f"❌ Test {index} failed. Compare {expected_output_file} and {generated_output_file}.")

if __name__ == "__main__":
    for i in range(1, 4):  
        run_test(i)
