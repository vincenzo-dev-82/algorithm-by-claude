# Chapter 7: 그래프 알고리즘 (Graph Algorithms)

## 7.1 그래프란?

### 정의
그래프는 정점(Vertex)과 간선(Edge)으로 구성된 자료구조입니다. 네트워크, 관계, 경로 등을 표현하는 데 사용됩니다.

### 그래프 용어
- **정점(Vertex/Node)**: 그래프의 기본 단위
- **간선(Edge)**: 정점 간의 연결
- **인접(Adjacent)**: 간선으로 직접 연결된 정점
- **차수(Degree)**: 정점에 연결된 간선의 수
- **경로(Path)**: 정점들의 연속된 연결
- **사이클(Cycle)**: 시작점과 끝점이 같은 경로

### 그래프의 종류
1. **방향 그래프 vs 무방향 그래프**
2. **가중치 그래프 vs 비가중치 그래프**
3. **연결 그래프 vs 비연결 그래프**
4. **순환 그래프 vs 비순환 그래프(DAG)**

## 7.2 그래프 표현 방법

### 인접 행렬 (Adjacency Matrix)
2차원 배열로 정점 간의 연결 표현

#### Java 구현
```java
public class GraphMatrix {
    private int[][] adjMatrix;
    private int numVertices;
    
    public GraphMatrix(int numVertices) {
        this.numVertices = numVertices;
        adjMatrix = new int[numVertices][numVertices];
    }
    
    // 간선 추가 (무방향 그래프)
    public void addEdge(int i, int j) {
        adjMatrix[i][j] = 1;
        adjMatrix[j][i] = 1;
    }
    
    // 간선 추가 (가중치 그래프)
    public void addEdge(int i, int j, int weight) {
        adjMatrix[i][j] = weight;
        adjMatrix[j][i] = weight;
    }
    
    // 간선 제거
    public void removeEdge(int i, int j) {
        adjMatrix[i][j] = 0;
        adjMatrix[j][i] = 0;
    }
    
    // 인접 여부 확인
    public boolean isEdge(int i, int j) {
        return adjMatrix[i][j] != 0;
    }
}
```

### 인접 리스트 (Adjacency List)
각 정점마다 인접한 정점들의 리스트 저장

#### Java 구현
```java
import java.util.*;

public class GraphList {
    private Map<Integer, List<Integer>> adjList;
    private int numVertices;
    
    public GraphList(int numVertices) {
        this.numVertices = numVertices;
        adjList = new HashMap<>();
        for (int i = 0; i < numVertices; i++) {
            adjList.put(i, new ArrayList<>());
        }
    }
    
    // 간선 추가 (무방향 그래프)
    public void addEdge(int src, int dest) {
        adjList.get(src).add(dest);
        adjList.get(dest).add(src);
    }
    
    // 간선 제거
    public void removeEdge(int src, int dest) {
        adjList.get(src).remove(Integer.valueOf(dest));
        adjList.get(dest).remove(Integer.valueOf(src));
    }
    
    // 인접 정점 리스트 반환
    public List<Integer> getNeighbors(int vertex) {
        return adjList.get(vertex);
    }
}

// 가중치 그래프용 인접 리스트
public class WeightedGraphList {
    class Edge {
        int dest, weight;
        
        Edge(int dest, int weight) {
            this.dest = dest;
            this.weight = weight;
        }
    }
    
    private Map<Integer, List<Edge>> adjList;
    
    public WeightedGraphList(int numVertices) {
        adjList = new HashMap<>();
        for (int i = 0; i < numVertices; i++) {
            adjList.put(i, new ArrayList<>());
        }
    }
    
    public void addEdge(int src, int dest, int weight) {
        adjList.get(src).add(new Edge(dest, weight));
        adjList.get(dest).add(new Edge(src, weight));
    }
}
```

#### C 구현
```c
#include <stdio.h>
#include <stdlib.h>

// 인접 리스트 노드
typedef struct Node {
    int vertex;
    struct Node* next;
} Node;

// 그래프 구조체
typedef struct Graph {
    int numVertices;
    Node** adjLists;
} Graph;

// 노드 생성
Node* createNode(int v) {
    Node* newNode = (Node*)malloc(sizeof(Node));
    newNode->vertex = v;
    newNode->next = NULL;
    return newNode;
}

// 그래프 생성
Graph* createGraph(int vertices) {
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    graph->numVertices = vertices;
    graph->adjLists = (Node**)malloc(vertices * sizeof(Node*));
    
    for (int i = 0; i < vertices; i++) {
        graph->adjLists[i] = NULL;
    }
    
    return graph;
}

// 간선 추가
void addEdge(Graph* graph, int src, int dest) {
    // src -> dest
    Node* newNode = createNode(dest);
    newNode->next = graph->adjLists[src];
    graph->adjLists[src] = newNode;
    
    // dest -> src (무방향 그래프)
    newNode = createNode(src);
    newNode->next = graph->adjLists[dest];
    graph->adjLists[dest] = newNode;
}
```

## 7.3 그래프 순회

### 깊이 우선 탐색 (DFS - Depth First Search)
한 경로를 끝까지 탐색한 후 다른 경로 탐색

#### Java 구현
```java
public class DFS {
    // 재귀적 DFS
    public static void dfsRecursive(GraphList graph, int start) {
        boolean[] visited = new boolean[graph.numVertices];
        dfsHelper(graph, start, visited);
    }
    
    private static void dfsHelper(GraphList graph, int vertex, boolean[] visited) {
        visited[vertex] = true;
        System.out.print(vertex + " ");
        
        for (int neighbor : graph.getNeighbors(vertex)) {
            if (!visited[neighbor]) {
                dfsHelper(graph, neighbor, visited);
            }
        }
    }
    
    // 반복적 DFS (스택 사용)
    public static void dfsIterative(GraphList graph, int start) {
        boolean[] visited = new boolean[graph.numVertices];
        Stack<Integer> stack = new Stack<>();
        
        stack.push(start);
        
        while (!stack.isEmpty()) {
            int vertex = stack.pop();
            
            if (!visited[vertex]) {
                visited[vertex] = true;
                System.out.print(vertex + " ");
                
                for (int neighbor : graph.getNeighbors(vertex)) {
                    if (!visited[neighbor]) {
                        stack.push(neighbor);
                    }
                }
            }
        }
    }
    
    // 모든 컴포넌트 방문
    public static void dfsAllComponents(GraphList graph) {
        boolean[] visited = new boolean[graph.numVertices];
        
        for (int i = 0; i < graph.numVertices; i++) {
            if (!visited[i]) {
                System.out.print("Component: ");
                dfsHelper(graph, i, visited);
                System.out.println();
            }
        }
    }
    
    // DFS 응용: 사이클 감지
    public static boolean hasCycle(GraphList graph) {
        boolean[] visited = new boolean[graph.numVertices];
        boolean[] recursionStack = new boolean[graph.numVertices];
        
        for (int i = 0; i < graph.numVertices; i++) {
            if (hasCycleUtil(graph, i, visited, recursionStack)) {
                return true;
            }
        }
        return false;
    }
    
    private static boolean hasCycleUtil(GraphList graph, int vertex, 
                                       boolean[] visited, boolean[] recursionStack) {
        visited[vertex] = true;
        recursionStack[vertex] = true;
        
        for (int neighbor : graph.getNeighbors(vertex)) {
            if (!visited[neighbor]) {
                if (hasCycleUtil(graph, neighbor, visited, recursionStack)) {
                    return true;
                }
            } else if (recursionStack[neighbor]) {
                return true;
            }
        }
        
        recursionStack[vertex] = false;
        return false;
    }
}
```

### 너비 우선 탐색 (BFS - Breadth First Search)
같은 레벨의 모든 정점을 먼저 탐색

#### Java 구현
```java
public class BFS {
    // 기본 BFS
    public static void bfs(GraphList graph, int start) {
        boolean[] visited = new boolean[graph.numVertices];
        Queue<Integer> queue = new LinkedList<>();
        
        visited[start] = true;
        queue.offer(start);
        
        while (!queue.isEmpty()) {
            int vertex = queue.poll();
            System.out.print(vertex + " ");
            
            for (int neighbor : graph.getNeighbors(vertex)) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    queue.offer(neighbor);
                }
            }
        }
    }
    
    // BFS로 최단 경로 찾기 (비가중치 그래프)
    public static int[] shortestPath(GraphList graph, int start) {
        int[] distance = new int[graph.numVertices];
        int[] parent = new int[graph.numVertices];
        boolean[] visited = new boolean[graph.numVertices];
        Queue<Integer> queue = new LinkedList<>();
        
        Arrays.fill(distance, -1);
        Arrays.fill(parent, -1);
        
        visited[start] = true;
        distance[start] = 0;
        queue.offer(start);
        
        while (!queue.isEmpty()) {
            int vertex = queue.poll();
            
            for (int neighbor : graph.getNeighbors(vertex)) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    distance[neighbor] = distance[vertex] + 1;
                    parent[neighbor] = vertex;
                    queue.offer(neighbor);
                }
            }
        }
        
        return distance;
    }
    
    // 경로 복원
    public static List<Integer> getPath(int[] parent, int start, int end) {
        List<Integer> path = new ArrayList<>();
        
        if (parent[end] == -1) {
            return path; // 경로 없음
        }
        
        int current = end;
        while (current != -1) {
            path.add(current);
            current = parent[current];
        }
        
        Collections.reverse(path);
        return path;
    }
}
```

#### C 구현
```c
#include <stdbool.h>
#include <string.h>

// DFS
void dfsUtil(Graph* graph, int vertex, bool visited[]) {
    visited[vertex] = true;
    printf("%d ", vertex);
    
    Node* temp = graph->adjLists[vertex];
    while (temp) {
        int adjVertex = temp->vertex;
        if (!visited[adjVertex]) {
            dfsUtil(graph, adjVertex, visited);
        }
        temp = temp->next;
    }
}

void dfs(Graph* graph, int start) {
    bool* visited = (bool*)calloc(graph->numVertices, sizeof(bool));
    dfsUtil(graph, start, visited);
    free(visited);
}

// BFS
void bfs(Graph* graph, int start) {
    bool* visited = (bool*)calloc(graph->numVertices, sizeof(bool));
    int* queue = (int*)malloc(graph->numVertices * sizeof(int));
    int front = 0, rear = 0;
    
    visited[start] = true;
    queue[rear++] = start;
    
    while (front < rear) {
        int vertex = queue[front++];
        printf("%d ", vertex);
        
        Node* temp = graph->adjLists[vertex];
        while (temp) {
            int adjVertex = temp->vertex;
            if (!visited[adjVertex]) {
                visited[adjVertex] = true;
                queue[rear++] = adjVertex;
            }
            temp = temp->next;
        }
    }
    
    free(visited);
    free(queue);
}
```

## 7.4 최단 경로 알고리즘

### Dijkstra 알고리즘
하나의 시작점에서 모든 정점까지의 최단 경로 (음수 가중치 불가)

#### Java 구현
```java
public class Dijkstra {
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
    
    public static int[] dijkstra(WeightedGraphList graph, int start) {
        int n = graph.numVertices;
        int[] dist = new int[n];
        boolean[] visited = new boolean[n];
        PriorityQueue<Node> pq = new PriorityQueue<>();
        
        Arrays.fill(dist, Integer.MAX_VALUE);
        dist[start] = 0;
        pq.offer(new Node(start, 0));
        
        while (!pq.isEmpty()) {
            Node current = pq.poll();
            int u = current.vertex;
            
            if (visited[u]) continue;
            visited[u] = true;
            
            for (Edge edge : graph.adjList.get(u)) {
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
        
        public List<Integer> getPath(int end) {
            List<Integer> path = new ArrayList<>();
            for (int at = end; at != -1; at = parent[at]) {
                path.add(at);
            }
            Collections.reverse(path);
            return path;
        }
    }
    
    public static DijkstraResult dijkstraWithPath(WeightedGraphList graph, int start) {
        int n = graph.numVertices;
        int[] dist = new int[n];
        int[] parent = new int[n];
        boolean[] visited = new boolean[n];
        PriorityQueue<Node> pq = new PriorityQueue<>();
        
        Arrays.fill(dist, Integer.MAX_VALUE);
        Arrays.fill(parent, -1);
        dist[start] = 0;
        pq.offer(new Node(start, 0));
        
        while (!pq.isEmpty()) {
            Node current = pq.poll();
            int u = current.vertex;
            
            if (visited[u]) continue;
            visited[u] = true;
            
            for (Edge edge : graph.adjList.get(u)) {
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

### Bellman-Ford 알고리즘
음수 가중치 허용, 음수 사이클 감지 가능

#### Java 구현
```java
public class BellmanFord {
    static class Edge {
        int src, dest, weight;
        
        Edge(int src, int dest, int weight) {
            this.src = src;
            this.dest = dest;
            this.weight = weight;
        }
    }
    
    public static int[] bellmanFord(List<Edge> edges, int V, int start) {
        int[] dist = new int[V];
        Arrays.fill(dist, Integer.MAX_VALUE);
        dist[start] = 0;
        
        // V-1번 반복
        for (int i = 0; i < V - 1; i++) {
            for (Edge edge : edges) {
                if (dist[edge.src] != Integer.MAX_VALUE &&
                    dist[edge.src] + edge.weight < dist[edge.dest]) {
                    dist[edge.dest] = dist[edge.src] + edge.weight;
                }
            }
        }
        
        // 음수 사이클 검사
        for (Edge edge : edges) {
            if (dist[edge.src] != Integer.MAX_VALUE &&
                dist[edge.src] + edge.weight < dist[edge.dest]) {
                System.out.println("Graph contains negative weight cycle");
                return null;
            }
        }
        
        return dist;
    }
}
```

### Floyd-Warshall 알고리즘
모든 정점 쌍 간의 최단 경로

#### Java 구현
```java
public class FloydWarshall {
    public static int[][] floydWarshall(int[][] graph) {
        int V = graph.length;
        int[][] dist = new int[V][V];
        
        // 초기화
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                dist[i][j] = graph[i][j];
            }
        }
        
        // Floyd-Warshall
        for (int k = 0; k < V; k++) {
            for (int i = 0; i < V; i++) {
                for (int j = 0; j < V; j++) {
                    if (dist[i][k] != Integer.MAX_VALUE && 
                        dist[k][j] != Integer.MAX_VALUE &&
                        dist[i][k] + dist[k][j] < dist[i][j]) {
                        dist[i][j] = dist[i][k] + dist[k][j];
                    }
                }
            }
        }
        
        return dist;
    }
}
```

## 7.5 최소 신장 트리 (MST)

### Kruskal's Algorithm
간선을 가중치 순으로 정렬 후 사이클을 만들지 않는 간선 선택

#### Java 구현
```java
public class Kruskal {
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
    
    public static List<Edge> kruskal(List<Edge> edges, int V) {
        Collections.sort(edges);
        UnionFind uf = new UnionFind(V);
        List<Edge> mst = new ArrayList<>();
        
        for (Edge edge : edges) {
            if (uf.union(edge.src, edge.dest)) {
                mst.add(edge);
                if (mst.size() == V - 1) break;
            }
        }
        
        return mst;
    }
}
```

### Prim's Algorithm
정점 중심으로 MST를 확장

#### Java 구현
```java
public class Prim {
    static class Edge implements Comparable<Edge> {
        int vertex, weight;
        
        Edge(int vertex, int weight) {
            this.vertex = vertex;
            this.weight = weight;
        }
        
        @Override
        public int compareTo(Edge other) {
            return Integer.compare(this.weight, other.weight);
        }
    }
    
    public static int prim(WeightedGraphList graph, int start) {
        int V = graph.numVertices;
        boolean[] inMST = new boolean[V];
        PriorityQueue<Edge> pq = new PriorityQueue<>();
        int mstWeight = 0;
        
        pq.offer(new Edge(start, 0));
        
        while (!pq.isEmpty()) {
            Edge current = pq.poll();
            int u = current.vertex;
            
            if (inMST[u]) continue;
            
            inMST[u] = true;
            mstWeight += current.weight;
            
            for (WeightedGraphList.Edge edge : graph.adjList.get(u)) {
                if (!inMST[edge.dest]) {
                    pq.offer(new Edge(edge.dest, edge.weight));
                }
            }
        }
        
        return mstWeight;
    }
}
```

## 7.6 위상 정렬 (Topological Sort)

### DFS 기반 위상 정렬
```java
public class TopologicalSort {
    public static List<Integer> topologicalSort(GraphList graph) {
        int V = graph.numVertices;
        boolean[] visited = new boolean[V];
        Stack<Integer> stack = new Stack<>();
        
        for (int i = 0; i < V; i++) {
            if (!visited[i]) {
                topologicalSortUtil(graph, i, visited, stack);
            }
        }
        
        List<Integer> result = new ArrayList<>();
        while (!stack.isEmpty()) {
            result.add(stack.pop());
        }
        return result;
    }
    
    private static void topologicalSortUtil(GraphList graph, int v,
                                           boolean[] visited, Stack<Integer> stack) {
        visited[v] = true;
        
        for (int neighbor : graph.getNeighbors(v)) {
            if (!visited[neighbor]) {
                topologicalSortUtil(graph, neighbor, visited, stack);
            }
        }
        
        stack.push(v);
    }
    
    // Kahn's Algorithm (BFS 기반)
    public static List<Integer> kahnsAlgorithm(GraphList graph) {
        int V = graph.numVertices;
        int[] indegree = new int[V];
        
        // 진입 차수 계산
        for (int i = 0; i < V; i++) {
            for (int neighbor : graph.getNeighbors(i)) {
                indegree[neighbor]++;
            }
        }
        
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < V; i++) {
            if (indegree[i] == 0) {
                queue.offer(i);
            }
        }
        
        List<Integer> result = new ArrayList<>();
        while (!queue.isEmpty()) {
            int u = queue.poll();
            result.add(u);
            
            for (int neighbor : graph.getNeighbors(u)) {
                indegree[neighbor]--;
                if (indegree[neighbor] == 0) {
                    queue.offer(neighbor);
                }
            }
        }
        
        if (result.size() != V) {
            return new ArrayList<>(); // 사이클 존재
        }
        
        return result;
    }
}
```

## 7.7 강연결 요소 (Strongly Connected Components)

### Kosaraju's Algorithm
```java
public class StronglyConnectedComponents {
    public static List<List<Integer>> kosaraju(GraphList graph) {
        int V = graph.numVertices;
        Stack<Integer> stack = new Stack<>();
        boolean[] visited = new boolean[V];
        
        // 1단계: 모든 정점에 대해 DFS 수행하고 스택에 저장
        for (int i = 0; i < V; i++) {
            if (!visited[i]) {
                fillOrder(graph, i, visited, stack);
            }
        }
        
        // 2단계: 그래프 전치
        GraphList transposed = getTranspose(graph);
        
        // 3단계: 전치 그래프에서 DFS 수행
        Arrays.fill(visited, false);
        List<List<Integer>> sccs = new ArrayList<>();
        
        while (!stack.isEmpty()) {
            int v = stack.pop();
            if (!visited[v]) {
                List<Integer> component = new ArrayList<>();
                dfsUtil(transposed, v, visited, component);
                sccs.add(component);
            }
        }
        
        return sccs;
    }
    
    private static void fillOrder(GraphList graph, int v, 
                                 boolean[] visited, Stack<Integer> stack) {
        visited[v] = true;
        for (int neighbor : graph.getNeighbors(v)) {
            if (!visited[neighbor]) {
                fillOrder(graph, neighbor, visited, stack);
            }
        }
        stack.push(v);
    }
    
    private static GraphList getTranspose(GraphList graph) {
        GraphList transposed = new GraphList(graph.numVertices);
        for (int v = 0; v < graph.numVertices; v++) {
            for (int neighbor : graph.getNeighbors(v)) {
                transposed.adjList.get(neighbor).add(v);
            }
        }
        return transposed;
    }
    
    private static void dfsUtil(GraphList graph, int v, 
                               boolean[] visited, List<Integer> component) {
        visited[v] = true;
        component.add(v);
        for (int neighbor : graph.getNeighbors(v)) {
            if (!visited[neighbor]) {
                dfsUtil(graph, neighbor, visited, component);
            }
        }
    }
}
```

## 7.8 요약

- 그래프는 정점과 간선으로 이루어진 자료구조
- DFS와 BFS는 그래프 순회의 기본 알고리즘
- 최단 경로: Dijkstra (양수), Bellman-Ford (음수 가능), Floyd-Warshall (모든 쌍)
- MST: Kruskal (간선 중심), Prim (정점 중심)
- 위상 정렬은 DAG에서 의존성 순서 결정
- 강연결 요소는 서로 도달 가능한 정점들의 집합