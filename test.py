from solution import *

def test():
    dfa = get_double_start_dfa_from_input()
    print(dfa)
    partitions = hopcroft_minimization(dfa)
    print(partitions)
    dfa = build_minimized_automata(dfa, partitions)
    # print(dfa)
    dfa = convert_double_start_to_double_final_dfa(dfa)
    # print(dfa)
    partitions = hopcroft_minimization(dfa)
    dfa = build_minimized_automata(dfa, partitions)
    print(dfa)

test()
