import sys
from io import StringIO
from solution import get_double_start_dfa_from_input, remove_unreachable_states, hopcroft_minimization, build_minimized_automata, convert_double_start_to_multi_final_dfa

sample_in = '''
8
0 a 1
0 b 2
1 a 3
1 b 4
2 a 5
2 b 6
3 a 1
3 b 4
4 a 4
4 b 5
5 a 4
5 b 5
6 a 7
6 b 7
7 a 6
7 b 6
0 2
4 5
'''.strip()

def test():
    first_dfa = get_double_start_dfa_from_input()
    print(first_dfa)
    dfa = remove_unreachable_states(first_dfa)
    partitions = hopcroft_minimization(dfa)
    dfa = build_minimized_automata(dfa, partitions)
    print(f'Minimized:\n{dfa}')
    dfa = convert_double_start_to_multi_final_dfa(dfa)
    print(f'MultiFinal state num:\n{len(dfa.states)}')
    print(f'Multifinal partition:\n{dfa.get_formatted_partitions()}')
    dfa = remove_unreachable_states(dfa)
    print(f'Trimmed:\n{len(dfa.states)}')
    print(f'Trimmed:\n{dfa}')
    partitions = hopcroft_minimization(dfa)
    dfa = build_minimized_automata(dfa, partitions)
    print(dfa)

test()
