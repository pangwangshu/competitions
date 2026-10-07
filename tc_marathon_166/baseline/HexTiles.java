import java.io.*;
import java.util.*;

/**
 * HexTiles - TopCoder Marathon Match 166. Day-one baseline.
 *
 * For each target exit pair, run Dijkstra over (tile, entryEdge) states to
 * find the minimum-rotation-cost route between the two exits, treating tiles
 * already rotated ("claimed") by earlier pairs as fixed. Pairs are routed in
 * order of increasing hex distance between their exit tiles. There is no
 * bonus routing, no joint optimization across pairs and no local search; the
 * Python solver in ../src grew out of this design.
 *
 * Geometry, exit numbering, tile connections and scoring follow the official
 * HexTilesTester.java (direction table, base chords {0,5},{1,3},{2,4} rotated
 * by the tile's orientation, and the initExits() border walk).
 */
public class HexTiles
{
  static final int S = 6;
  static final int[] dr = {-1, 0, +1, +1, 0, -1};
  static final int[] dc = {+1, +1, 0, -1, -1, 0};
  static final int[][] connections = {{0, 5}, {1, 3}, {2, 4}};

  static int N, M, B, W;
  static int[][] grid;
  static boolean[][] claimed;

  static int numExits;
  static int[] exitToR, exitToC, exitToEdge;
  static HashMap<Long, Integer> loc2exit;

  static List<int[]> moveList = new ArrayList<int[]>(); // each = {r, c, dir}
  static int movesUsed = 0;
  static int safeBudget;

  public static void main(String[] args) throws Exception
  {
    BufferedReader in = new BufferedReader(new InputStreamReader(System.in));

    N = Integer.parseInt(in.readLine());
    M = Integer.parseInt(in.readLine());
    B = Integer.parseInt(in.readLine());
    int edges = Integer.parseInt(in.readLine());

    int[][] pairs = new int[edges][2];
    for (int i = 0; i < edges; i++)
    {
      StringTokenizer st = new StringTokenizer(in.readLine());
      pairs[i][0] = Integer.parseInt(st.nextToken());
      pairs[i][1] = Integer.parseInt(st.nextToken());
    }

    W = 2 * N - 1;
    grid = new int[W][W];
    for (int r = 0; r < W; r++)
      for (int c = 0; c < W; c++)
        grid[r][c] = Integer.parseInt(in.readLine());

    // Bonus tile locations are read (to keep parsing in sync with the
    // input format) but not used for routing decisions in this v1 -- a
    // natural v2 improvement is to bias Dijkstra edge costs to prefer
    // paths that cross bonus tiles, since they multiply a matched path's
    // score by (bonusesCrossed + 1).
    for (int i = 0; i < B; i++)
      in.readLine();

    claimed = new boolean[W][W];
    numExits = S * (2 * N - 1);
    initExits();

    int moveBudget = S * 4 * N * N; // 24*N*N, mirrors the tester's hard cap
    safeBudget = moveBudget - Math.max(6, N); // small safety margin

    final int[][] pairsRef = pairs;
    Integer[] order = new Integer[edges];
    for (int i = 0; i < edges; i++) order[i] = i;
    Arrays.sort(order, new Comparator<Integer>() {
      public int compare(Integer a, Integer b) {
        int da = rawDist(pairsRef[a][0], pairsRef[a][1]);
        int db = rawDist(pairsRef[b][0], pairsRef[b][1]);
        return Integer.compare(da, db);
      }
    });

    for (int idx : order)
    {
      int p = pairs[idx][0], q = pairs[idx][1];
      tryConnect(p, q);
    }

    StringBuilder sb = new StringBuilder();
    sb.append(moveList.size()).append('\n');
    for (int[] mv : moveList)
      sb.append(mv[0]).append(' ').append(mv[1]).append(' ').append(mv[2]).append('\n');
    System.out.print(sb);
    System.out.flush();
  }

  // ---- geometry helpers, mirroring HexTilesTester.java exactly ----

  static int getStart(int r) { return r < N ? N - 1 - r : 0; }
  static int getEnd(int r) { return r < N ? W - 1 : W + N - r - 2; }
  static boolean inGrid(int r, int c)
  {
    return r >= 0 && r < W && c >= getStart(r) && c <= getEnd(r);
  }

  // Given a tile's orientation, what edge does `entry` connect to?
  static int partnerAt(int entry, int orientation)
  {
    for (int[] p : connections)
    {
      int a = (p[0] + orientation) % S;
      int b = (p[1] + orientation) % S;
      if (a == entry) return b;
      if (b == entry) return a;
    }
    return -1; // unreachable given a valid entry 0..5
  }

  // Replicates HexTilesTester.initExits(): walks the grid border assigning
  // sequential exit IDs to every outward-facing tile edge.
  static void initExits()
  {
    exitToR = new int[numExits];
    exitToC = new int[numExits];
    exitToEdge = new int[numExits];
    loc2exit = new HashMap<Long, Integer>();

    int cur = 0;
    int r = 0;
    int c = getStart(0);
    int d = 1;
    while (cur < numExits)
    {
      int d2 = (d + 1) % S;
      for (int q = 0; q < S; q++, d2 = (d2 + 1) % S)
      {
        int r2 = r + dr[d2];
        int c2 = c + dc[d2];
        if (!inGrid(r2, c2))
        {
          loc2exit.put(locKey(r, c, d2), cur);
          exitToR[cur] = r;
          exitToC[cur] = c;
          exitToEdge[cur] = d2;
          cur++;
        }
      }
      int r2 = r + dr[d];
      int c2 = c + dc[d];
      if (!inGrid(r2, c2))
      {
        d = (d + 1) % S;
        r2 = r + dr[d];
        c2 = c + dc[d];
      }
      r = r2;
      c = c2;
    }
  }

  static long locKey(int r, int c, int edge)
  {
    return ((long) r * W + c) * S + edge;
  }

  static int rotationDist(int from, int to)
  {
    int d = ((to - from) % S + S) % S;
    return Math.min(d, S - d);
  }

  // Minimum rotation cost, from cur orientation, to make `entry` pair with
  // `exitEdge` at this tile (there is always at least one such orientation
  // for any entry/exitEdge pair that isn't the "opposite" edge, and
  // occasionally two -- take the cheaper).
  static int minRotation(int entry, int exitEdge, int curOrient)
  {
    int best = Integer.MAX_VALUE;
    for (int o = 0; o < S; o++)
      if (partnerAt(entry, o) == exitEdge)
        best = Math.min(best, rotationDist(curOrient, o));
    return best;
  }

  static int rawDist(int p, int q)
  {
    int r1 = exitToR[p], c1 = exitToC[p];
    int r2 = exitToR[q], c2 = exitToC[q];
    int drr = r1 - r2, dcc = c1 - c2;
    return (Math.abs(drr) + Math.abs(dcc) + Math.abs(drr + dcc)) / 2;
  }

  static void applyRotation(int r, int c, int from, int to)
  {
    int dPlus = ((to - from) % S + S) % S;
    int dMinus = ((from - to) % S + S) % S;
    int dir = (dPlus <= dMinus) ? +1 : -1;
    int steps = Math.min(dPlus, dMinus);
    for (int i = 0; i < steps; i++)
    {
      moveList.add(new int[]{r, c, dir});
      movesUsed++;
    }
    grid[r][c] = to;
  }

  // ---- per-pair Dijkstra over (tile, entryEdge) states ----

  static void tryConnect(int p, int q)
  {
    int startR = exitToR[p], startC = exitToC[p], startEdge = exitToEdge[p];
    int startNode = (startR * W + startC) * S + startEdge;
    int totalRegular = W * W * S;
    int goalNode = totalRegular + q;
    int totalNodes = totalRegular + numExits;

    int[] dist = new int[totalNodes];
    int[] prevNode = new int[totalNodes];
    int[] prevChoice = new int[totalNodes];
    Arrays.fill(dist, Integer.MAX_VALUE);
    Arrays.fill(prevNode, -1);
    boolean[] visited = new boolean[totalNodes];

    dist[startNode] = 0;
    PriorityQueue<int[]> pq = new PriorityQueue<int[]>(new Comparator<int[]>() {
      public int compare(int[] a, int[] b) { return Integer.compare(a[0], b[0]); }
    });
    pq.add(new int[]{0, startNode});

    while (!pq.isEmpty())
    {
      int[] top = pq.poll();
      int d0 = top[0], node = top[1];
      if (visited[node]) continue;
      if (d0 > dist[node]) continue;
      visited[node] = true;
      if (node == goalNode) break;
      if (node >= totalRegular) continue; // terminal nodes have no outgoing edges

      int entryEdge = node % S;
      int rc = node / S;
      int r = rc / W, c = rc % W;

      if (claimed[r][c])
      {
        int exitEdge = partnerAt(entryEdge, grid[r][c]);
        relax(dist, prevNode, prevChoice, pq, totalRegular, node, d0, r, c, entryEdge, exitEdge, 0);
      }
      else
      {
        int curOrient = grid[r][c];
        int opp = (entryEdge + 3) % S;
        for (int exitEdge = 0; exitEdge < S; exitEdge++)
        {
          if (exitEdge == entryEdge || exitEdge == opp) continue; // opposite edges are never connectable
          int cost = minRotation(entryEdge, exitEdge, curOrient);
          relax(dist, prevNode, prevChoice, pq, totalRegular, node, d0, r, c, entryEdge, exitEdge, cost);
        }
      }
    }

    if (dist[goalNode] == Integer.MAX_VALUE) return; // no route found, skip this pair
    if (movesUsed + dist[goalNode] > safeBudget) return; // would blow the move budget, skip

    // reconstruct the chain of (predecessorNode, chosenExitEdge) decisions
    List<int[]> chain = new ArrayList<int[]>();
    int cur = goalNode;
    while (cur != startNode)
    {
      int pn = prevNode[cur];
      chain.add(new int[]{pn, prevChoice[cur]});
      cur = pn;
    }
    Collections.reverse(chain);

    for (int[] step : chain)
    {
      int node = step[0], chosenExit = step[1];
      if (node >= totalRegular) continue;
      int entryEdge = node % S;
      int rc = node / S;
      int r = rc / W, c = rc % W;
      if (!claimed[r][c])
      {
        int curOrient = grid[r][c];
        int bestO = -1, bestCost = Integer.MAX_VALUE;
        for (int o = 0; o < S; o++)
        {
          if (partnerAt(entryEdge, o) == chosenExit)
          {
            int cst = rotationDist(curOrient, o);
            if (cst < bestCost) { bestCost = cst; bestO = o; }
          }
        }
        applyRotation(r, c, curOrient, bestO);
        claimed[r][c] = true;
      }
      // if already claimed, its (single, deterministic) transition needs
      // no rotation -- it was already accounted for with cost 0 above.
    }
  }

  static void relax(int[] dist, int[] prevNode, int[] prevChoice, PriorityQueue<int[]> pq,
                     int totalRegular, int node, int d0, int r, int c, int entryEdge, int exitEdge, int cost)
  {
    int nd = d0 + cost;
    int r2 = r + dr[exitEdge], c2 = c + dc[exitEdge];
    int nextNode;
    if (inGrid(r2, c2))
    {
      int nextEntry = (exitEdge + 3) % S;
      nextNode = (r2 * W + c2) * S + nextEntry;
    }
    else
    {
      Integer exitId = loc2exit.get(locKey(r, c, exitEdge));
      if (exitId == null) return; // shouldn't happen for a valid boundary edge
      nextNode = totalRegular + exitId;
    }
    if (nd < dist[nextNode])
    {
      dist[nextNode] = nd;
      prevNode[nextNode] = node;
      prevChoice[nextNode] = exitEdge;
      pq.add(new int[]{nd, nextNode});
    }
  }
}
