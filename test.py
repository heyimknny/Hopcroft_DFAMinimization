from solution import *

def test():
    dfa = get_double_start_dfa_from_input()
    dfa = convert_double_start_to_double_final_dfa(dfa)
    print(dfa)

test()
