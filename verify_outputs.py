'''
verify_outputs.py
Usage: python3 verify_outputs.py [io_dir] [test_prefix] [true_prefix]
'''

import sys, os
from solution import load_mf_dfa, equivalent_dfa

def main2():
    io_dir = sys.argv[1] if len(sys.argv) > 1 else 'io'
    test_pref = sys.argv[2] if len(sys.argv) > 2 else 'test_'
    true_pref = sys.argv[3] if len(sys.argv) > 3 else 'true_'

    ok = True
    for fn in sorted(os.listdir(io_dir)):
        if not fn.startswith(test_pref) or not fn.endswith('.out'):
            continue
        base = fn[len(test_pref):-4]
        test_path = os.path.join(io_dir, fn)
        true_fn = f"{true_pref}{base}.out"
        true_path = os.path.join(io_dir, true_fn)
        if not os.path.exists(true_path):
            print(f"[ERROR] Missing true output: {true_fn}")
            ok = False
            continue

        d_test = load_mf_dfa(test_path)
        d_true = load_mf_dfa(true_path)

        # state count
        if len(d_test.states) != len(d_true.states):
            print(f"[FAIL] {base}: state count {len(d_test.states)} != {len(d_true.states)}")
            ok = False
            continue

        # language equivalence
        if not equivalent_dfa(d_test, d_true):
            print(f"[FAIL] {base}: language mismatch")
            ok = False
            continue

        print(f"[PASS] {base}")

    sys.exit(0 if ok else 1)


