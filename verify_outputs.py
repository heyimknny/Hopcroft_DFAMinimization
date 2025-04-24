'''
verify_outputs.py
Usage: python3 verify_outputs.py [file1.out] [file2.out]
'''

import sys, os
from solution import load_mf_dfa, equivalent_dfa
def main():
    if len(sys.argv) != 3:
        print("Usage: python3 verify_outputs.py [file1.out] [file2.out]")
        sys.exit(1)

    file1, file2 = sys.argv[1], sys.argv[2]
    if not os.path.exists(file1) or not os.path.exists(file2):
        print(f"Error: One or both files not found: {file1}, {file2}")
        sys.exit(1)

    d1 = load_mf_dfa(file1)
    d2 = load_mf_dfa(file2)

    ok = True
    if len(d1.states) != len(d2.states):
        print(f"State count mismatch: {len(d1.states)} vs {len(d2.states)}")
        ok = False

    if not equivalent_dfa(d1, d2):
        ok = False

    if ok:
        print("[PASS] DFAs are equivalent and same size")
        sys.exit(0)
    else:
        print("[FAIL] DFAs differ")
        sys.exit(2)

if __name__ == '__main__':
    main()
