import java.io.*;
import java.util.*;

public class CandyLandEspionage {
    public static void main(String[] args) throws IOException {
        FastReader in = new FastReader();

        // 1) Read header
        int n     = in.nextInt();    // original # states
        int sigma = in.nextInt();    // alphabet size
        int T     = in.nextInt();    // # transitions
        int F     = in.nextInt();    // # accept states

        // 2) Read accept states
        boolean[] isAccept = new boolean[n];
        for (int i = 0; i < F; i++) {
            isAccept[in.nextInt()] = true;
        }

        // 3) Read transitions (supporting '?' as a wildcard for any single letter)
        int[][] trans = new int[n][sigma];
        for (int i = 0; i < T; i++) {
            int u = in.nextInt();
            char c = in.next().charAt(0);
            int v = in.nextInt();
            if (c == '?') {
                // wildcard: for every symbol in [0..sigma-1], go to v
                for (int x = 0; x < sigma; x++) {
                    trans[u][x] = v;
                }
            } else {
                // normal letter
                trans[u][c - 'a'] = v;
            }
        }

        // 4) Read initial state
        int start = in.nextInt();

        // 5) Remove unreachable states via BFS
        boolean[] reachable = new boolean[n];
        Deque<Integer> dq = new ArrayDeque<>();
        reachable[start] = true;
        dq.add(start);
        while (!dq.isEmpty()) {
            int u = dq.poll();
            for (int c = 0; c < sigma; c++) {
                int v = trans[u][c];
                if (!reachable[v]) {
                    reachable[v] = true;
                    dq.add(v);
                }
            }
        }

        // 6) Reindex reachable states 0..R-1
        int R = 0;
        int[] oldToNewState = new int[n];
        Arrays.fill(oldToNewState, -1);
        for (int i = 0; i < n; i++) {
            if (reachable[i]) {
                oldToNewState[i] = R++;
            }
        }
        int newStart = oldToNewState[start];

        // 7) Build trimmed DFA
        boolean[] accR = new boolean[R];
        int[][] trR   = new int[R][sigma];
        for (int i = 0; i < n; i++) {
            if (!reachable[i]) continue;
            int ni = oldToNewState[i];
            accR[ni] = isAccept[i];
            for (int c = 0; c < sigma; c++) {
                trR[ni][c] = oldToNewState[trans[i][c]];
            }
        }

        // 8) Minimize with Hopcroft
        HopcroftResult hr = hopcroft(trR, accR, sigma);
        int pc                = hr.partitionCount;
        int[] partOfState     = hr.partitionForState;
        boolean[] isAccBlock  = hr.isAcceptPartition;
        int[][] blockTrans    = hr.transitions;

        // 9) Canonical renumbering via BFS on quotient
        int[] newBlockId   = new int[pc];
        boolean[] visBlock = new boolean[pc];
        Arrays.fill(newBlockId, -1);
        Deque<Integer> q2 = new ArrayDeque<>();
        int startBlk = partOfState[newStart];
        visBlock[startBlk] = true;
        newBlockId[startBlk] = 0;
        q2.add(startBlk);
        int nextId = 1;
        while (!q2.isEmpty()) {
            int b = q2.poll();
            for (int c = 0; c < sigma; c++) {
                int d = blockTrans[b][c];
                if (!visBlock[d]) {
                    visBlock[d] = true;
                    newBlockId[d] = nextId++;
                    q2.add(d);
                }
            }
        }

        // 10) Collect accepting blocks
        List<Integer> accList = new ArrayList<>();
        for (int b = 0; b < pc; b++) {
            if (isAccBlock[b]) accList.add(newBlockId[b]);
        }
        Collections.sort(accList);

        // 11) Build reverse map new→old block
        int[] oldBlockOfNew = new int[pc];
        for (int b = 0; b < pc; b++) {
            oldBlockOfNew[newBlockId[b]] = b;
        }

        // 12) Output
        StringBuilder out = new StringBuilder();
        out.append(pc).append(" ").append(sigma).append(" ")
           .append(pc * sigma).append(" ").append(accList.size()).append("\n");

        for (int i = 0; i < accList.size(); i++) {
            if (i > 0) out.append(' ');
            out.append(accList.get(i));
        }
        out.append("\n");

        // transitions
        for (int uNew = 0; uNew < pc; uNew++) {
            int ob = oldBlockOfNew[uNew];
            for (int c = 0; c < sigma; c++) {
                int dest = newBlockId[ blockTrans[ob][c] ];
                out.append(uNew).append(' ').append((char)('a'+c))
                   .append(' ').append(dest).append("\n");
            }
        }

        // initial
        out.append("0\n");

        // mapping
        out.append("Mapping:");
        for (int i = 0; i < n; i++) {
            out.append(' ');
            out.append(reachable[i] ? newBlockId[ partOfState[ oldToNewState[i] ] ] : -1);
        }
        out.append("\n");

        System.out.print(out);
    }

    static class HopcroftResult {
        int partitionCount;
        int[] partitionForState;
        boolean[] isAcceptPartition;
        int[][] transitions;
        HopcroftResult(int pc, int[] pfs, boolean[] iap, int[][] tr) {
            partitionCount = pc;
            partitionForState = pfs;
            isAcceptPartition = iap;
            transitions = tr;
        }
    }

    static HopcroftResult hopcroft(int[][] trans, boolean[] accept, int sigma) {
        int R = trans.length;
        List<List<List<Integer>>> rev = new ArrayList<>();
        for (int c = 0; c < sigma; c++) {
            rev.add(new ArrayList<>());
            for (int i = 0; i < R; i++) rev.get(c).add(new ArrayList<>());
        }
        for (int u = 0; u < R; u++) for (int c = 0; c < sigma; c++)
            rev.get(c).get(trans[u][c]).add(u);

        int[] stateBlock = new int[R];
        List<List<Integer>> blocks = new ArrayList<>();
        List<Integer> nonAcc = new ArrayList<>(), acc = new ArrayList<>();
        for (int s = 0; s < R; s++) {
            (accept[s] ? acc : nonAcc).add(s);
        }
        if (!nonAcc.isEmpty()) { blocks.add(nonAcc); for (int s: nonAcc) stateBlock[s] = blocks.size()-1; }
        if (!acc.isEmpty())    { blocks.add(acc);    for (int s: acc)    stateBlock[s] = blocks.size()-1; }

        Deque<int[]> work = new ArrayDeque<>();
        for (int b = 0; b < blocks.size(); b++)
            for (int c = 0; c < sigma; c++)
                work.add(new int[]{b,c});

        while (!work.isEmpty()) {
            int[] bc = work.poll();
            int A=bc[0], c=bc[1];
            List<Integer> X = new ArrayList<>();
            for (int q: blocks.get(A)) X.addAll(rev.get(c).get(q));
            if (X.isEmpty()) continue;
            Map<Integer,List<Integer>> bucket = new HashMap<>();
            for (int q: X) bucket.computeIfAbsent(stateBlock[q], k->new ArrayList<>()).add(q);
            for (var e: bucket.entrySet()) {
                int B = e.getKey();
                List<Integer> inX = e.getValue();
                List<Integer> blk = blocks.get(B);
                if (inX.size() == blk.size()) continue;
                Set<Integer> inSet = new HashSet<>(inX);
                List<Integer> rem = new ArrayList<>();
                for (int q: blk) if (!inSet.contains(q)) rem.add(q);
                blocks.set(B, rem);
                int Bp = blocks.size();
                blocks.add(inX);
                for (int q: inX) stateBlock[q] = Bp;
                for (int cc=0; cc<sigma; cc++) work.add(new int[]{Bp,cc});
            }
        }

        int pc = blocks.size();
        int[] partOf = new int[R]; boolean[] blockAcc = new boolean[pc];
        for (int b = 0; b < pc; b++) {
            for (int s: blocks.get(b)) partOf[s] = b;
            blockAcc[b] = accept[ blocks.get(b).get(0) ];
        }
        int[][] bTrans = new int[pc][sigma];
        for (int b = 0; b < pc; b++) {
            int r = blocks.get(b).get(0);
            for (int c = 0; c < sigma; c++) bTrans[b][c] = partOf[ trans[r][c] ];
        }
        return new HopcroftResult(pc, partOf, blockAcc, bTrans);
    }

    static class FastReader {
        BufferedReader br; StringTokenizer st;
        FastReader() { br = new BufferedReader(new InputStreamReader(System.in)); }
        String next() throws IOException {
            while (st == null || !st.hasMoreTokens()) st = new StringTokenizer(br.readLine());
            return st.nextToken();
        }
        int nextInt() throws IOException { return Integer.parseInt(next()); }
    }
}