import random

def generate_random_double_start_dfa_input(
    num_states: int,
    alphabet: list = ['a', 'b'],
    num_finals: int = None,
    allow_same_start: bool = True,
    seed: int = None
) -> str:
    """
    Generates a random DoubleStart DFA input string in the format:
    
    n
    p c q  (2n lines for each state and symbol)
    ...
    start1 start2
    f       (number of final states)
    q1
    q2
    ...
    
    Parameters:
    - num_states: number of states (0 through num_states-1)
    - alphabet: list of symbols
    - num_finals: exact number of finals (random if None)
    - allow_same_start: if True, start1 and start2 can be the same
    - seed: optional random seed for reproducibility
    
    Returns:
    - A multiline string representing the DFA input.
    """
    if seed is not None:
        random.seed(seed)
    
    # Decide number of finals if not provided
    if num_finals is None:
        num_finals = random.randint(0, num_states)
    
    # Build transition lines
    transitions = []
    for p in range(num_states):
        for c in alphabet:
            q = random.randrange(num_states)
            transitions.append(f"{p} {c} {q}")
    
    # Choose start states
    start1 = random.randrange(num_states)
    if allow_same_start:
        start2 = random.randrange(num_states)
    else:
        start2 = random.choice([s for s in range(num_states) if s != start1])
    
    # Choose final states
    finals = sorted(random.sample(range(num_states), num_finals))
    
    # Assemble all parts
    lines = []
    lines.append(str(num_states))
    lines.extend(transitions)
    lines.append(f"{start1} {start2}")
    lines.append(str(len(finals)))
    lines.extend(str(f) for f in finals)
    
    return "\n".join(lines)

def write_dfa_to_file(num_states):
    with open(f'io/random{num_states}.in', 'w') as file:
        file.write(generate_random_double_start_dfa_input(num_states))

def main():
    for num_states in [5,10,25,50,100,500,1000,5000]:
        write_dfa_to_file(num_states)

if __name__ == '__main__':
    main()
