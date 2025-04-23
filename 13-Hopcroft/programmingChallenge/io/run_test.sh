#!/usr/bin/env bash
# run_tests.sh — run CandyLandEspionage on test.in.0…test.in.20 and compare to test.out.0…test.out.20

PASS=0
FAIL=0

for i in $(seq 0 20); do
  IN=test.in.$i
  OUT=out.$i
  EXPECT=test.out.$i

  if [ ! -f "$IN" ] || [ ! -f "$EXPECT" ]; then
    echo "Skipping test $i: missing $IN or $EXPECT"
    continue
  fi

  # Run the program
  java CandyLandEspionage < "$IN" > "$OUT"
  STATUS=$?

  if [ $STATUS -ne 0 ]; then
    echo "Test $i: ERROR (exit code $STATUS)"
    ((FAIL++))
    continue
  fi

  # Compare ignoring trailing whitespace
  if diff -u <(sed 's/[[:space:]]*$//' "$EXPECT") \
            <(sed 's/[[:space:]]*$//' "$OUT") > /dev/null; then
    echo "Test $i: PASS"
    ((PASS++))
  else
    echo "Test $i: FAIL"
    diff -u "$EXPECT" "$OUT" | sed 's/^/    /'
    ((FAIL++))
  fi
done

echo
echo "Results: $PASS passed, $FAIL failed."
