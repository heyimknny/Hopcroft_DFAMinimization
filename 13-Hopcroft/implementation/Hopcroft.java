import java.io.*;
import java.nio.file.Files;
import java.util.*;

public class Hopcroft {

    public static void main(String[] args) {
        try {
            for (int i = 1; i <= 3; i++) {
                runTest(i);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    static class DFA {
        Set<String> states;
        Set<String> alphabet;
        Map<TransitionKey, String> tf;
        String start;
        Set<String> accept;

        DFA(Set<String> states, Set<String> alphabet,
            Map<TransitionKey, String> tf, String start, Set<String> accept) {
            this.states = new HashSet<>(states);
            this.alphabet = new HashSet<>(alphabet);
            this.tf = new HashMap<>(tf);
            this.start = start;
            this.accept = new HashSet<>(accept);
        }

        void removeUnreachable() {
            Set<String> reachable = new HashSet<>();
            Deque<String> queue = new ArrayDeque<>();
            queue.add(start);
            while (!queue.isEmpty()) {
                String s = queue.poll();
                if (reachable.add(s)) {
                    for (String c : alphabet) {
                        String t = tf.get(new TransitionKey(s, c));
                        if (t != null && !reachable.contains(t)) {
                            queue.add(t);
                        }
                    }
                }
            }
            states.retainAll(reachable);
            accept.retainAll(reachable);
            tf.entrySet().removeIf(e -> {
                String s = e.getKey().state;
                String t = e.getValue();
                return !reachable.contains(s) || !reachable.contains(t);
            });
        }

        void makeTotal() {
            String trap = "__TRAP__";
            boolean needsTrap = false;
            for (String s : new HashSet<>(states)) {
                for (String c : alphabet) {
                    TransitionKey key = new TransitionKey(s, c);
                    if (!tf.containsKey(key)) {
                        tf.put(key, trap);
                        needsTrap = true;
                    }
                }
            }
            if (needsTrap) {
                states.add(trap);
                for (String c : alphabet) {
                    tf.put(new TransitionKey(trap, c), trap);
                }
            }
        }

        DFA minimize() {
            removeUnreachable();
            makeTotal();

            // Initial partition: accepting vs non-accepting
            List<Set<String>> P = new ArrayList<>();
            Set<String> nonAccept = new HashSet<>(states);
            nonAccept.removeAll(accept);
            if (!accept.isEmpty()) P.add(new HashSet<>(accept));
            if (!nonAccept.isEmpty()) P.add(nonAccept);

            Deque<Set<String>> W = new ArrayDeque<>(P);

            while (!W.isEmpty()) {
                Set<String> A = W.poll();
                for (String c : alphabet) {
                    Set<String> X = new HashSet<>();
                    for (String s : states) {
                        String t = tf.get(new TransitionKey(s, c));
                        if (t != null && A.contains(t)) {
                            X.add(s);
                        }
                    }
                    List<Set<String>> newP = new ArrayList<>();
                    for (Set<String> Y : P) {
                        Set<String> inter = new HashSet<>(Y);
                        inter.retainAll(X);
                        Set<String> diff = new HashSet<>(Y);
                        diff.removeAll(X);
                        if (!inter.isEmpty() && !diff.isEmpty()) {
                            newP.add(inter);
                            newP.add(diff);
                            if (W.remove(Y)) {
                                W.add(inter);
                                W.add(diff);
                            } else {
                                W.add(inter.size() <= diff.size() ? inter : diff);
                            }
                        } else {
                            newP.add(Y);
                        }
                    }
                    P = newP;
                }
            }

            // Map each final block to a numeric label 0..n-1, ordered by original state names
            List<Set<String>> sortedBlocks = new ArrayList<>(P);
            sortedBlocks.sort(Comparator.comparing(b -> Collections.min(b)));
            Map<Set<String>, String> blockIndex = new HashMap<>();
            for (int i = 0; i < sortedBlocks.size(); i++) {
                blockIndex.put(sortedBlocks.get(i), Integer.toString(i));
            }

            // Build representative map
            Map<String, String> rep = new HashMap<>();
            for (Set<String> block : sortedBlocks) {
                String idx = blockIndex.get(block);
                for (String s : block) {
                    rep.put(s, idx);
                }
            }

            // Construct new DFA
            Set<String> newStates = new HashSet<>(blockIndex.values());
            String newStart = rep.get(start);
            Set<String> newAccept = new HashSet<>();
            for (String s : accept) newAccept.add(rep.get(s));
            Map<TransitionKey, String> newTf = new HashMap<>();
            for (Set<String> block : sortedBlocks) {
                String any = Collections.min(block);
                String from = rep.get(any);
                for (String c : alphabet) {
                    String toState = tf.get(new TransitionKey(any, c));
                    if (toState != null) newTf.put(new TransitionKey(from, c), rep.get(toState));
                }
            }

            return new DFA(newStates, alphabet, newTf, newStart, newAccept);
        }
    }

    static class TransitionKey {
        String state;
        String symbol;

        TransitionKey(String state, String symbol) {
            this.state = state;
            this.symbol = symbol;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof TransitionKey)) return false;
            TransitionKey that = (TransitionKey) o;
            return state.equals(that.state) && symbol.equals(that.symbol);
        }

        @Override
        public int hashCode() {
            return Objects.hash(state, symbol);
        }
    }

    static DFA parseDFAFromCSV(String filename) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String[] fields = br.readLine().split(",");
            List<String> alphabet = Arrays.asList(fields).subList(3, fields.length);

            Set<String> states = new HashSet<>();
            Map<TransitionKey, String> tf = new HashMap<>();
            String start = null;
            Set<String> accept = new HashSet<>();

            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                String s = parts[0];
                states.add(s);
                if (parts[1].equals("1")) start = s;
                if (parts[2].equals("1")) accept.add(s);
                for (int i = 3; i < parts.length; i++) {
                    tf.put(new TransitionKey(s, alphabet.get(i - 3)), parts[i]);
                }
            }
            return new DFA(states, new HashSet<>(alphabet), tf, start, accept);
        }
    }

    static void writeDFAtoCSV(DFA dfa, String filename) throws IOException {
        List<String> sortedAlpha = new ArrayList<>(dfa.alphabet);
        Collections.sort(sortedAlpha);
        List<String> sortedStates = new ArrayList<>(dfa.states);
        sortedStates.sort(Comparator.comparingInt(Integer::parseInt));

        try (BufferedWriter bw = new BufferedWriter(new FileWriter(filename))) {
            // Header
            bw.write("state,is_start,is_accept");
            for (String c : sortedAlpha) bw.write("," + c);
            bw.newLine();
            // Rows
            for (String s : sortedStates) {
                bw.write(s + "," + (s.equals(dfa.start) ? "1" : "0") + "," + (dfa.accept.contains(s) ? "1" : "0"));
                for (String c : sortedAlpha) {
                    String to = dfa.tf.get(new TransitionKey(s, c));
                    bw.write("," + (to != null ? to : ""));
                }
                bw.newLine();
            }
        }
    }

    static boolean compareCSVFiles(String f1, String f2) throws IOException {
        List<String> a = Files.readAllLines(new File(f1).toPath());
        List<String> b = Files.readAllLines(new File(f2).toPath());
        return a.equals(b);
    }

    static void runTest(int index) throws IOException {
        String base = "io" + File.separator;
        String in = base + "sample.in." + index;
        String out = base + "sample.out." + index;
        String gen = base + "generated.out." + index;

        System.out.println("Running Test " + index + "...");
        DFA dfa = parseDFAFromCSV(in);
        DFA min = dfa.minimize();
        writeDFAtoCSV(min, gen);
        if (compareCSVFiles(out, gen)) {
            System.out.println("✅ Test " + index + " passed.");
        } else {
            System.out.println("❌ Test " + index + " failed. Compare " + out + " and " + gen + ".");
        }
    }
}
