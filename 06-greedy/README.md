# Chapter 6: 탐욕 알고리즘 (Greedy Algorithms)

## 6.1 탐욕 알고리즘이란?

### 정의
탐욕 알고리즘(Greedy Algorithm)은 각 단계에서 지역적으로 최적인 선택을 하여 전역적으로 최적인 해를 구하려는 알고리즘입니다.

### 탐욕 알고리즘의 특징
1. **지역 최적 선택**: 현재 상태에서 가장 좋은 선택
2. **한 번의 선택**: 선택을 번복하지 않음
3. **빠른 실행**: 일반적으로 효율적
4. **최적해 보장 조건**: 탐욕 선택 속성과 최적 부분 구조

### 탐욕 선택 속성 (Greedy Choice Property)
전역 최적해가 지역 최적 선택들로 구성될 수 있는 속성

### 최적 부분 구조 (Optimal Substructure)
문제의 최적해가 부분 문제의 최적해를 포함하는 구조

## 6.2 활동 선택 문제 (Activity Selection)

### 문제 정의
시작 시간과 종료 시간이 주어진 활동들 중 겹치지 않는 최대 개수의 활동 선택

#### Java 구현
```java
import java.util.*;

public class ActivitySelection {
    static class Activity {
        int start, finish;
        
        Activity(int start, int finish) {
            this.start = start;
            this.finish = finish;
        }
    }
    
    // 종료 시간 기준 정렬 후 선택
    public static List<Activity> selectActivities(Activity[] activities) {
        // 종료 시간으로 정렬
        Arrays.sort(activities, (a, b) -> Integer.compare(a.finish, b.finish));
        
        List<Activity> selected = new ArrayList<>();
        selected.add(activities[0]);
        int lastFinish = activities[0].finish;
        
        for (int i = 1; i < activities.length; i++) {
            if (activities[i].start >= lastFinish) {
                selected.add(activities[i]);
                lastFinish = activities[i].finish;
            }
        }
        
        return selected;
    }
    
    // 가중치가 있는 활동 선택 (DP 필요)
    public static int weightedActivitySelection(Activity[] activities, int[] weights) {
        int n = activities.length;
        
        // 종료 시간으로 정렬
        Integer[] indices = new Integer[n];
        for (int i = 0; i < n; i++) indices[i] = i;
        Arrays.sort(indices, (i, j) -> 
            Integer.compare(activities[i].finish, activities[j].finish));
        
        // DP 배열
        int[] dp = new int[n];
        dp[0] = weights[indices[0]];
        
        for (int i = 1; i < n; i++) {
            // 현재 활동을 포함하는 경우
            int inclWeight = weights[indices[i]];
            int latest = findLatestNonConflict(activities, indices, i);
            if (latest != -1) {
                inclWeight += dp[latest];
            }
            
            // 현재 활동을 포함하지 않는 경우
            int exclWeight = dp[i - 1];
            
            dp[i] = Math.max(inclWeight, exclWeight);
        }
        
        return dp[n - 1];
    }
    
    private static int findLatestNonConflict(Activity[] activities, 
                                           Integer[] indices, int i) {
        for (int j = i - 1; j >= 0; j--) {
            if (activities[indices[j]].finish <= activities[indices[i]].start) {
                return j;
            }
        }
        return -1;
    }
    
    // 회의실 배정 문제
    public static int minMeetingRooms(Activity[] activities) {
        if (activities.length == 0) return 0;
        
        // 시작 시간과 종료 시간 분리
        int[] starts = new int[activities.length];
        int[] ends = new int[activities.length];
        
        for (int i = 0; i < activities.length; i++) {
            starts[i] = activities[i].start;
            ends[i] = activities[i].finish;
        }
        
        Arrays.sort(starts);
        Arrays.sort(ends);
        
        int rooms = 0, maxRooms = 0;
        int s = 0, e = 0;
        
        while (s < activities.length) {
            if (starts[s] < ends[e]) {
                rooms++;
                maxRooms = Math.max(maxRooms, rooms);
                s++;
            } else {
                rooms--;
                e++;
            }
        }
        
        return maxRooms;
    }
}
```

#### C 구현
```c
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int start, finish;
} Activity;

// 종료 시간 기준 비교 함수
int compareActivities(const void* a, const void* b) {
    Activity* act1 = (Activity*)a;
    Activity* act2 = (Activity*)b;
    return act1->finish - act2->finish;
}

// 활동 선택
int selectActivities(Activity activities[], int n, int selected[]) {
    // 종료 시간으로 정렬
    qsort(activities, n, sizeof(Activity), compareActivities);
    
    int count = 1;
    selected[0] = 0;
    int lastFinish = activities[0].finish;
    
    for (int i = 1; i < n; i++) {
        if (activities[i].start >= lastFinish) {
            selected[count++] = i;
            lastFinish = activities[i].finish;
        }
    }
    
    return count;
}
```

## 6.3 Huffman 코딩

### 문제 정의
문자의 빈도에 따라 가변 길이 코드를 할당하여 데이터 압축

#### Java 구현
```java
import java.util.*;

public class HuffmanCoding {
    static class Node implements Comparable<Node> {
        char ch;
        int freq;
        Node left, right;
        
        Node(char ch, int freq) {
            this.ch = ch;
            this.freq = freq;
            this.left = null;
            this.right = null;
        }
        
        @Override
        public int compareTo(Node other) {
            return Integer.compare(this.freq, other.freq);
        }
    }
    
    // Huffman 트리 구축
    public static Node buildHuffmanTree(char[] chars, int[] freqs) {
        PriorityQueue<Node> pq = new PriorityQueue<>();
        
        // 리프 노드 생성
        for (int i = 0; i < chars.length; i++) {
            pq.offer(new Node(chars[i], freqs[i]));
        }
        
        // 트리 구축
        while (pq.size() > 1) {
            Node left = pq.poll();
            Node right = pq.poll();
            
            Node parent = new Node('\0', left.freq + right.freq);
            parent.left = left;
            parent.right = right;
            
            pq.offer(parent);
        }
        
        return pq.poll();
    }
    
    // Huffman 코드 생성
    public static Map<Character, String> generateCodes(Node root) {
        Map<Character, String> codes = new HashMap<>();
        generateCodesHelper(root, "", codes);
        return codes;
    }
    
    private static void generateCodesHelper(Node node, String code, 
                                          Map<Character, String> codes) {
        if (node == null) return;
        
        // 리프 노드인 경우
        if (node.left == null && node.right == null) {
            codes.put(node.ch, code.length() > 0 ? code : "0");
            return;
        }
        
        generateCodesHelper(node.left, code + "0", codes);
        generateCodesHelper(node.right, code + "1", codes);
    }
    
    // 인코딩
    public static String encode(String text, Map<Character, String> codes) {
        StringBuilder encoded = new StringBuilder();
        
        for (char ch : text.toCharArray()) {
            encoded.append(codes.get(ch));
        }
        
        return encoded.toString();
    }
    
    // 디코딩
    public static String decode(String encoded, Node root) {
        StringBuilder decoded = new StringBuilder();
        Node current = root;
        
        for (char bit : encoded.toCharArray()) {
            if (bit == '0') {
                current = current.left;
            } else {
                current = current.right;
            }
            
            // 리프 노드 도달
            if (current.left == null && current.right == null) {
                decoded.append(current.ch);
                current = root;
            }
        }
        
        return decoded.toString();
    }
    
    // 압축률 계산
    public static void printCompressionStats(String original, 
                                           Map<Character, String> codes) {
        int originalBits = original.length() * 8;
        int compressedBits = 0;
        
        for (char ch : original.toCharArray()) {
            compressedBits += codes.get(ch).length();
        }
        
        double ratio = (double) compressedBits / originalBits;
        System.out.printf("Original: %d bits%n", originalBits);
        System.out.printf("Compressed: %d bits%n", compressedBits);
        System.out.printf("Compression ratio: %.2f%%%n", ratio * 100);
    }
}
```

## 6.4 최소 신장 트리 (MST)

### Kruskal 알고리즘
간선을 가중치 순으로 정렬 후 사이클을 만들지 않는 간선 선택

#### Java 구현
```java
public class KruskalMST {
    static class Edge implements Comparable<Edge> {
        int src, dest, weight;
        
        Edge(int src, int dest, int weight) {
            this.src = src;
            this.dest = dest;
            this.weight = weight;
        }
        
        @Override
        public int compareTo(Edge other) {
            return Integer.compare(this.weight, other.weight);
        }
    }
    
    static class UnionFind {
        int[] parent, rank;
        
        UnionFind(int n) {
            parent = new int[n];
            rank = new int[n];
            for (int i = 0; i < n; i++) {
                parent[i] = i;
            }
        }
        
        int find(int x) {
            if (parent[x] != x) {
                parent[x] = find(parent[x]); // 경로 압축
            }
            return parent[x];
        }
        
        boolean union(int x, int y) {
            int px = find(x);
            int py = find(y);
            
            if (px == py) return false;
            
            // 랭크에 따른 합병
            if (rank[px] < rank[py]) {
                parent[px] = py;
            } else if (rank[px] > rank[py]) {
                parent[py] = px;
            } else {
                parent[py] = px;
                rank[px]++;
            }
            return true;
        }
    }
    
    public static List<Edge> kruskal(int V, List<Edge> edges) {
        // 간선을 가중치 순으로 정렬
        Collections.sort(edges);
        
        UnionFind uf = new UnionFind(V);
        List<Edge> mst = new ArrayList<>();
        int totalWeight = 0;
        
        for (Edge edge : edges) {
            if (uf.union(edge.src, edge.dest)) {
                mst.add(edge);
                totalWeight += edge.weight;
                
                // V-1개의 간선을 선택하면 완료
                if (mst.size() == V - 1) break;
            }
        }
        
        System.out.println("Total MST weight: " + totalWeight);
        return mst;
    }
}
```

### Prim 알고리즘
정점을 하나씩 추가하며 MST 구성

#### Java 구현
```java
public class PrimMST {
    static class Edge {
        int dest, weight;
        
        Edge(int dest, int weight) {
            this.dest = dest;
            this.weight = weight;
        }
    }
    
    static class Node implements Comparable<Node> {
        int vertex, key;
        
        Node(int vertex, int key) {
            this.vertex = vertex;
            this.key = key;
        }
        
        @Override
        public int compareTo(Node other) {
            return Integer.compare(this.key, other.key);
        }
    }
    
    public static void prim(List<List<Edge>> graph, int V) {
        boolean[] inMST = new boolean[V];
        int[] key = new int[V];
        int[] parent = new int[V];
        PriorityQueue<Node> pq = new PriorityQueue<>();
        
        // 초기화
        Arrays.fill(key, Integer.MAX_VALUE);
        Arrays.fill(parent, -1);
        
        // 시작 정점
        key[0] = 0;
        pq.offer(new Node(0, 0));
        
        int totalWeight = 0;
        
        while (!pq.isEmpty()) {
            Node node = pq.poll();
            int u = node.vertex;
            
            if (inMST[u]) continue;
            
            inMST[u] = true;
            totalWeight += key[u];
            
            // 인접 정점 업데이트
            for (Edge edge : graph.get(u)) {
                int v = edge.dest;
                int weight = edge.weight;
                
                if (!inMST[v] && weight < key[v]) {
                    key[v] = weight;
                    parent[v] = u;
                    pq.offer(new Node(v, key[v]));
                }
            }
        }
        
        // MST 출력
        System.out.println("Total MST weight: " + totalWeight);
        for (int i = 1; i < V; i++) {
            System.out.println(parent[i] + " - " + i + " : " + key[i]);
        }
    }
}
```

## 6.5 Dijkstra 최단 경로

### 단일 출발점 최단 경로
음이 아닌 가중치에서 최단 경로 찾기

#### Java 구현
```java
public class DijkstraAlgorithm {
    static class Edge {
        int dest, weight;
        
        Edge(int dest, int weight) {
            this.dest = dest;
            this.weight = weight;
        }
    }
    
    static class Node implements Comparable<Node> {
        int vertex, distance;
        
        Node(int vertex, int distance) {
            this.vertex = vertex;
            this.distance = distance;
        }
        
        @Override
        public int compareTo(Node other) {
            return Integer.compare(this.distance, other.distance);
        }
    }
    
    public static int[] dijkstra(List<List<Edge>> graph, int src, int V) {
        int[] dist = new int[V];
        boolean[] visited = new boolean[V];
        PriorityQueue<Node> pq = new PriorityQueue<>();
        
        // 초기화
        Arrays.fill(dist, Integer.MAX_VALUE);
        dist[src] = 0;
        pq.offer(new Node(src, 0));
        
        while (!pq.isEmpty()) {
            Node node = pq.poll();
            int u = node.vertex;
            
            if (visited[u]) continue;
            visited[u] = true;
            
            // 인접 정점 relaxation
            for (Edge edge : graph.get(u)) {
                int v = edge.dest;
                int weight = edge.weight;
                
                if (dist[u] + weight < dist[v]) {
                    dist[v] = dist[u] + weight;
                    pq.offer(new Node(v, dist[v]));
                }
            }
        }
        
        return dist;
    }
    
    // 경로 추적 포함
    public static class DijkstraResult {
        int[] dist;
        int[] parent;
        
        DijkstraResult(int[] dist, int[] parent) {
            this.dist = dist;
            this.parent = parent;
        }
        
        public List<Integer> getPath(int dest) {
            List<Integer> path = new ArrayList<>();
            for (int at = dest; at != -1; at = parent[at]) {
                path.add(at);
            }
            Collections.reverse(path);
            return path;
        }
    }
    
    public static DijkstraResult dijkstraWithPath(List<List<Edge>> graph, 
                                                  int src, int V) {
        int[] dist = new int[V];
        int[] parent = new int[V];
        boolean[] visited = new boolean[V];
        PriorityQueue<Node> pq = new PriorityQueue<>();
        
        Arrays.fill(dist, Integer.MAX_VALUE);
        Arrays.fill(parent, -1);
        dist[src] = 0;
        pq.offer(new Node(src, 0));
        
        while (!pq.isEmpty()) {
            Node node = pq.poll();
            int u = node.vertex;
            
            if (visited[u]) continue;
            visited[u] = true;
            
            for (Edge edge : graph.get(u)) {
                int v = edge.dest;
                int weight = edge.weight;
                
                if (dist[u] + weight < dist[v]) {
                    dist[v] = dist[u] + weight;
                    parent[v] = u;
                    pq.offer(new Node(v, dist[v]));
                }
            }
        }
        
        return new DijkstraResult(dist, parent);
    }
}
```

## 6.6 작업 스케줄링

### 작업 스케줄링 문제들

#### Java 구현
```java
public class JobScheduling {
    static class Job {
        String id;
        int deadline;
        int profit;
        
        Job(String id, int deadline, int profit) {
            this.id = id;
            this.deadline = deadline;
            this.profit = profit;
        }
    }
    
    // 최대 이익 작업 스케줄링
    public static List<Job> scheduleJobs(Job[] jobs) {
        // 이익 기준 내림차순 정렬
        Arrays.sort(jobs, (a, b) -> Integer.compare(b.profit, a.profit));
        
        int maxDeadline = 0;
        for (Job job : jobs) {
            maxDeadline = Math.max(maxDeadline, job.deadline);
        }
        
        // 각 시간 슬롯의 사용 여부
        Job[] slots = new Job[maxDeadline + 1];
        
        for (Job job : jobs) {
            // 가능한 가장 늦은 시간에 배치
            for (int j = job.deadline; j > 0; j--) {
                if (slots[j] == null) {
                    slots[j] = job;
                    break;
                }
            }
        }
        
        List<Job> scheduled = new ArrayList<>();
        int totalProfit = 0;
        
        for (int i = 1; i <= maxDeadline; i++) {
            if (slots[i] != null) {
                scheduled.add(slots[i]);
                totalProfit += slots[i].profit;
            }
        }
        
        System.out.println("Total profit: " + totalProfit);
        return scheduled;
    }
    
    // 최소 완료 시간 스케줄링
    static class Task {
        int processingTime;
        int weight;
        
        Task(int processingTime, int weight) {
            this.processingTime = processingTime;
            this.weight = weight;
        }
    }
    
    public static double minimizeWeightedCompletionTime(Task[] tasks) {
        // 가중치/처리시간 비율로 정렬
        Arrays.sort(tasks, (a, b) -> 
            Double.compare((double)b.weight / b.processingTime, 
                          (double)a.weight / a.processingTime));
        
        double totalWeightedTime = 0;
        int currentTime = 0;
        
        for (Task task : tasks) {
            currentTime += task.processingTime;
            totalWeightedTime += currentTime * task.weight;
        }
        
        return totalWeightedTime;
    }
    
    // 인터벌 파티셔닝
    public static int intervalPartitioning(Activity[] activities) {
        // 시작 시간으로 정렬
        Arrays.sort(activities, (a, b) -> 
            Integer.compare(a.start, b.start));
        
        PriorityQueue<Integer> endTimes = new PriorityQueue<>();
        
        for (Activity activity : activities) {
            if (!endTimes.isEmpty() && endTimes.peek() <= activity.start) {
                endTimes.poll();
            }
            endTimes.offer(activity.finish);
        }
        
        return endTimes.size();
    }
}
```

## 6.7 그리디 알고리즘의 정당성 증명

### 증명 기법
1. **교환 논증 (Exchange Argument)**
2. **귀납법 (Induction)**
3. **모순에 의한 증명 (Contradiction)**

### 예제: 활동 선택 문제의 정당성
```java
public class GreedyProof {
    // 활동 선택 문제의 최적성 증명
    /*
     * 증명:
     * 1. 가장 일찍 끝나는 활동을 선택하는 것이 최적
     * 2. 교환 논증: 최적해에서 첫 번째 활동을 가장 일찍 끝나는 
     *    활동으로 교체해도 여전히 최적해
     * 3. 귀납적으로 나머지 부분 문제도 동일하게 해결
     */
    
    // 그리디 선택이 최적해에 포함됨을 보이는 예제
    public static boolean verifyGreedyChoice(Activity[] activities) {
        // 종료 시간으로 정렬
        Arrays.sort(activities, (a, b) -> 
            Integer.compare(a.finish, b.finish));
        
        // 그리디 선택
        List<Activity> greedy = selectActivities(activities);
        
        // 동적 프로그래밍으로 최적해 계산
        int dpOptimal = calculateOptimalDP(activities);
        
        return greedy.size() == dpOptimal;
    }
    
    private static int calculateOptimalDP(Activity[] activities) {
        int n = activities.length;
        int[] dp = new int[n];
        dp[0] = 1;
        
        for (int i = 1; i < n; i++) {
            // 현재 활동 포함
            int include = 1;
            for (int j = i - 1; j >= 0; j--) {
                if (activities[j].finish <= activities[i].start) {
                    include = dp[j] + 1;
                    break;
                }
            }
            
            // 현재 활동 미포함
            int exclude = dp[i - 1];
            
            dp[i] = Math.max(include, exclude);
        }
        
        return dp[n - 1];
    }
}
```

## 6.8 동전 교환 문제

### 그리디가 작동하는 경우와 작동하지 않는 경우

#### Java 구현
```java
public class CoinChange {
    // 그리디 동전 교환 (특정 동전 시스템에서만 최적)
    public static List<Integer> greedyCoinChange(int[] coins, int amount) {
        // 큰 동전부터 정렬
        Arrays.sort(coins);
        List<Integer> result = new ArrayList<>();
        
        for (int i = coins.length - 1; i >= 0; i--) {
            while (amount >= coins[i]) {
                result.add(coins[i]);
                amount -= coins[i];
            }
        }
        
        if (amount > 0) {
            return new ArrayList<>(); // 불가능
        }
        
        return result;
    }
    
    // 그리디가 최적인지 확인
    public static boolean isGreedyOptimal(int[] coins, int maxAmount) {
        for (int amount = 1; amount <= maxAmount; amount++) {
            List<Integer> greedy = greedyCoinChange(coins, amount);
            int dp = minCoinChangeDP(coins, amount);
            
            if (greedy.isEmpty() && dp == -1) continue;
            if (greedy.isEmpty() || greedy.size() != dp) {
                System.out.println("Greedy fails at amount: " + amount);
                return false;
            }
        }
        return true;
    }
    
    // DP로 최소 동전 개수 계산
    private static int minCoinChangeDP(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, amount + 1);
        dp[0] = 0;
        
        for (int i = 1; i <= amount; i++) {
            for (int coin : coins) {
                if (coin <= i) {
                    dp[i] = Math.min(dp[i], dp[i - coin] + 1);
                }
            }
        }
        
        return dp[amount] > amount ? -1 : dp[amount];
    }
}
```

## 6.9 탐욕 알고리즘 설계 전략

### 설계 단계
1. **문제 분석**: 최적 부분 구조 확인
2. **탐욕 선택**: 지역 최적 선택 정의
3. **정당성 증명**: 탐욕 선택이 전역 최적임을 증명
4. **구현**: 효율적인 자료구조 선택

### 일반적인 패턴
1. **정렬 후 선택**: 특정 기준으로 정렬 후 순차 선택
2. **우선순위 큐 사용**: 동적으로 최적 선택
3. **Union-Find**: 집합 관리 (Kruskal)

## 6.10 실습 문제

1. **부분 배낭 문제**: 물건을 쪼갤 수 있는 배낭 문제
2. **주유소 문제**: 최소 주유 횟수로 목적지 도달
3. **점프 게임**: 최소 점프로 배열 끝 도달
4. **구간 커버**: 최소 구간으로 전체 범위 커버

## 6.11 요약

- 탐욕 알고리즘은 각 단계에서 지역 최적을 선택
- 탐욕 선택 속성과 최적 부분 구조가 필요
- 항상 최적해를 보장하지는 않음
- 활동 선택, Huffman 코딩, MST, 최단 경로 등에 활용
- 정당성 증명이 중요