# Chapter 8: 백트래킹 (Backtracking)

## 8.1 백트래킹이란?

### 정의
백트래킹은 모든 가능한 경우를 탐색하면서, 조건에 맞지 않으면 이전 단계로 돌아가는 알고리즘입니다. 깊이 우선 탐색(DFS)의 변형으로 볼 수 있습니다.

### 백트래킹의 특징
1. **체계적 탐색**: 모든 가능한 해를 체계적으로 탐색
2. **가지치기(Pruning)**: 불필요한 탐색 경로를 조기에 차단
3. **상태 공간 트리**: 문제를 트리 구조로 표현
4. **백트랙**: 막다른 길에서 이전 상태로 복귀

### 백트래킹 vs 브루트 포스
- 브루트 포스: 모든 경우를 무조건 탐색
- 백트래킹: 가능성이 없는 경우를 조기에 제외

## 8.2 N-Queens 문제

### 문제 정의
N×N 체스판에 N개의 퀸을 서로 공격할 수 없도록 배치

#### Java 구현
```java
public class NQueens {
    // 기본 N-Queens 해결
    public static List<List<String>> solveNQueens(int n) {
        List<List<String>> solutions = new ArrayList<>();
        char[][] board = new char[n][n];
        
        // 보드 초기화
        for (int i = 0; i < n; i++) {
            Arrays.fill(board[i], '.');
        }
        
        backtrack(solutions, board, 0, n);
        return solutions;
    }
    
    private static void backtrack(List<List<String>> solutions, 
                                 char[][] board, int row, int n) {
        // 기저 사례: 모든 퀸을 배치
        if (row == n) {
            solutions.add(construct(board));
            return;
        }
        
        // 현재 행의 각 열에 퀸 배치 시도
        for (int col = 0; col < n; col++) {
            if (isValid(board, row, col, n)) {
                board[row][col] = 'Q';
                backtrack(solutions, board, row + 1, n);
                board[row][col] = '.'; // 백트랙
            }
        }
    }
    
    private static boolean isValid(char[][] board, int row, int col, int n) {
        // 같은 열 확인
        for (int i = 0; i < row; i++) {
            if (board[i][col] == 'Q') {
                return false;
            }
        }
        
        // 왼쪽 대각선 확인
        for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--) {
            if (board[i][j] == 'Q') {
                return false;
            }
        }
        
        // 오른쪽 대각선 확인
        for (int i = row - 1, j = col + 1; i >= 0 && j < n; i--, j++) {
            if (board[i][j] == 'Q') {
                return false;
            }
        }
        
        return true;
    }
    
    private static List<String> construct(char[][] board) {
        List<String> result = new ArrayList<>();
        for (char[] row : board) {
            result.add(new String(row));
        }
        return result;
    }
    
    // 최적화된 N-Queens (비트마스크 사용)
    public static int totalNQueens(int n) {
        return solveNQueensBitwise(0, 0, 0, 0, n);
    }
    
    private static int solveNQueensBitwise(int row, int cols, 
                                          int diag1, int diag2, int n) {
        if (row == n) {
            return 1;
        }
        
        int count = 0;
        int availablePositions = ((1 << n) - 1) & ~(cols | diag1 | diag2);
        
        while (availablePositions != 0) {
            int position = availablePositions & -availablePositions;
            availablePositions -= position;
            
            count += solveNQueensBitwise(
                row + 1,
                cols | position,
                (diag1 | position) << 1,
                (diag2 | position) >> 1,
                n
            );
        }
        
        return count;
    }
    
    // N-Queens 해의 개수만 구하기 (최적화)
    public static int countNQueensSolutions(int n) {
        boolean[] cols = new boolean[n];
        boolean[] diag1 = new boolean[2 * n - 1];
        boolean[] diag2 = new boolean[2 * n - 1];
        return countSolutions(0, n, cols, diag1, diag2);
    }
    
    private static int countSolutions(int row, int n, boolean[] cols,
                                     boolean[] diag1, boolean[] diag2) {
        if (row == n) {
            return 1;
        }
        
        int count = 0;
        for (int col = 0; col < n; col++) {
            int d1 = row - col + n - 1;
            int d2 = row + col;
            
            if (!cols[col] && !diag1[d1] && !diag2[d2]) {
                cols[col] = diag1[d1] = diag2[d2] = true;
                count += countSolutions(row + 1, n, cols, diag1, diag2);
                cols[col] = diag1[d1] = diag2[d2] = false;
            }
        }
        
        return count;
    }
}
```

#### C 구현
```c
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

bool isValid(int board[], int row, int col) {
    for (int i = 0; i < row; i++) {
        // 같은 열 또는 대각선에 퀸이 있는지 확인
        if (board[i] == col || 
            board[i] - i == col - row ||
            board[i] + i == col + row) {
            return false;
        }
    }
    return true;
}

void printSolution(int board[], int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (board[i] == j) {
                printf("Q ");
            } else {
                printf(". ");
            }
        }
        printf("\n");
    }
    printf("\n");
}

void solveNQueens(int board[], int row, int n, int* count) {
    if (row == n) {
        printSolution(board, n);
        (*count)++;
        return;
    }
    
    for (int col = 0; col < n; col++) {
        if (isValid(board, row, col)) {
            board[row] = col;
            solveNQueens(board, row + 1, n, count);
        }
    }
}

int nQueens(int n) {
    int* board = (int*)malloc(n * sizeof(int));
    int count = 0;
    solveNQueens(board, 0, n, &count);
    free(board);
    return count;
}
```

## 8.3 스도쿠 해결

### 9×9 스도쿠 퍼즐 해결

#### Java 구현
```java
public class SudokuSolver {
    private static final int SIZE = 9;
    private static final int SUBGRID_SIZE = 3;
    
    public static boolean solveSudoku(int[][] board) {
        return solve(board);
    }
    
    private static boolean solve(int[][] board) {
        // 빈 셀 찾기
        int[] emptyCell = findEmptyCell(board);
        if (emptyCell == null) {
            return true; // 모든 셀이 채워짐
        }
        
        int row = emptyCell[0];
        int col = emptyCell[1];
        
        // 1부터 9까지 시도
        for (int num = 1; num <= SIZE; num++) {
            if (isValid(board, row, col, num)) {
                board[row][col] = num;
                
                if (solve(board)) {
                    return true;
                }
                
                board[row][col] = 0; // 백트랙
            }
        }
        
        return false;
    }
    
    private static int[] findEmptyCell(int[][] board) {
        for (int row = 0; row < SIZE; row++) {
            for (int col = 0; col < SIZE; col++) {
                if (board[row][col] == 0) {
                    return new int[]{row, col};
                }
            }
        }
        return null;
    }
    
    private static boolean isValid(int[][] board, int row, int col, int num) {
        // 행 확인
        for (int x = 0; x < SIZE; x++) {
            if (board[row][x] == num) {
                return false;
            }
        }
        
        // 열 확인
        for (int x = 0; x < SIZE; x++) {
            if (board[x][col] == num) {
                return false;
            }
        }
        
        // 3×3 서브그리드 확인
        int startRow = row - row % SUBGRID_SIZE;
        int startCol = col - col % SUBGRID_SIZE;
        
        for (int i = 0; i < SUBGRID_SIZE; i++) {
            for (int j = 0; j < SUBGRID_SIZE; j++) {
                if (board[startRow + i][startCol + j] == num) {
                    return false;
                }
            }
        }
        
        return true;
    }
    
    // 최적화된 스도쿠 솔버 (후보 추적)
    public static boolean solveSudokuOptimized(int[][] board) {
        boolean[][][] candidates = new boolean[SIZE][SIZE][SIZE + 1];
        
        // 후보 초기화
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                if (board[i][j] == 0) {
                    for (int num = 1; num <= SIZE; num++) {
                        candidates[i][j][num] = true;
                    }
                } else {
                    updateCandidates(candidates, i, j, board[i][j]);
                }
            }
        }
        
        return solveWithCandidates(board, candidates);
    }
    
    private static boolean solveWithCandidates(int[][] board, 
                                              boolean[][][] candidates) {
        // 최소 후보를 가진 셀 찾기
        int[] cell = findBestCell(board, candidates);
        if (cell == null) {
            return true;
        }
        
        int row = cell[0];
        int col = cell[1];
        
        for (int num = 1; num <= SIZE; num++) {
            if (candidates[row][col][num]) {
                board[row][col] = num;
                
                // 후보 백업
                boolean[][][] backup = copyCandidates(candidates);
                updateCandidates(candidates, row, col, num);
                
                if (solveWithCandidates(board, candidates)) {
                    return true;
                }
                
                // 백트랙
                board[row][col] = 0;
                candidates = backup;
            }
        }
        
        return false;
    }
    
    private static int[] findBestCell(int[][] board, boolean[][][] candidates) {
        int minCandidates = SIZE + 1;
        int[] bestCell = null;
        
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                if (board[i][j] == 0) {
                    int count = countCandidates(candidates[i][j]);
                    if (count < minCandidates) {
                        minCandidates = count;
                        bestCell = new int[]{i, j};
                    }
                }
            }
        }
        
        return bestCell;
    }
    
    private static void updateCandidates(boolean[][][] candidates, 
                                        int row, int col, int num) {
        // 같은 행의 후보 제거
        for (int j = 0; j < SIZE; j++) {
            candidates[row][j][num] = false;
        }
        
        // 같은 열의 후보 제거
        for (int i = 0; i < SIZE; i++) {
            candidates[i][col][num] = false;
        }
        
        // 같은 서브그리드의 후보 제거
        int startRow = row - row % SUBGRID_SIZE;
        int startCol = col - col % SUBGRID_SIZE;
        
        for (int i = 0; i < SUBGRID_SIZE; i++) {
            for (int j = 0; j < SUBGRID_SIZE; j++) {
                candidates[startRow + i][startCol + j][num] = false;
            }
        }
        
        // 현재 셀의 모든 후보 제거
        for (int n = 1; n <= SIZE; n++) {
            candidates[row][col][n] = false;
        }
    }
}
```

## 8.4 부분집합 합 문제

### 주어진 집합에서 합이 특정 값이 되는 부분집합 찾기

#### Java 구현
```java
public class SubsetSum {
    // 모든 부분집합 생성
    public static List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        backtrack(result, new ArrayList<>(), nums, 0);
        return result;
    }
    
    private static void backtrack(List<List<Integer>> result, 
                                 List<Integer> tempList, 
                                 int[] nums, int start) {
        result.add(new ArrayList<>(tempList));
        
        for (int i = start; i < nums.length; i++) {
            tempList.add(nums[i]);
            backtrack(result, tempList, nums, i + 1);
            tempList.remove(tempList.size() - 1);
        }
    }
    
    // 합이 target인 부분집합 찾기
    public static List<List<Integer>> subsetSum(int[] nums, int target) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        findSubsets(result, new ArrayList<>(), nums, target, 0);
        return result;
    }
    
    private static void findSubsets(List<List<Integer>> result,
                                   List<Integer> tempList,
                                   int[] nums, int remain, int start) {
        if (remain < 0) return; // 가지치기
        
        if (remain == 0) {
            result.add(new ArrayList<>(tempList));
            return;
        }
        
        for (int i = start; i < nums.length; i++) {
            tempList.add(nums[i]);
            findSubsets(result, tempList, nums, remain - nums[i], i + 1);
            tempList.remove(tempList.size() - 1);
        }
    }
    
    // 중복 원소가 있는 경우
    public static List<List<Integer>> subsetsWithDup(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        backtrackWithDup(result, new ArrayList<>(), nums, 0);
        return result;
    }
    
    private static void backtrackWithDup(List<List<Integer>> result,
                                        List<Integer> tempList,
                                        int[] nums, int start) {
        result.add(new ArrayList<>(tempList));
        
        for (int i = start; i < nums.length; i++) {
            // 중복 건너뛰기
            if (i > start && nums[i] == nums[i - 1]) continue;
            
            tempList.add(nums[i]);
            backtrackWithDup(result, tempList, nums, i + 1);
            tempList.remove(tempList.size() - 1);
        }
    }
    
    // 분할 문제: 배열을 같은 합의 두 부분집합으로 나누기
    public static boolean canPartition(int[] nums) {
        int totalSum = 0;
        for (int num : nums) {
            totalSum += num;
        }
        
        if (totalSum % 2 != 0) {
            return false;
        }
        
        int target = totalSum / 2;
        return canFindSubsetSum(nums, 0, 0, target);
    }
    
    private static boolean canFindSubsetSum(int[] nums, int index, 
                                           int currentSum, int target) {
        if (currentSum == target) {
            return true;
        }
        
        if (index >= nums.length || currentSum > target) {
            return false;
        }
        
        // 현재 원소 포함
        if (canFindSubsetSum(nums, index + 1, currentSum + nums[index], target)) {
            return true;
        }
        
        // 현재 원소 미포함
        return canFindSubsetSum(nums, index + 1, currentSum, target);
    }
}
```

## 8.5 그래프 색칠 문제

### 인접한 정점이 다른 색을 갖도록 그래프 색칠

#### Java 구현
```java
public class GraphColoring {
    // m개의 색으로 그래프 색칠 가능한지 확인
    public static boolean graphColoring(int[][] graph, int m) {
        int n = graph.length;
        int[] colors = new int[n];
        Arrays.fill(colors, -1);
        
        return colorGraph(graph, colors, 0, m);
    }
    
    private static boolean colorGraph(int[][] graph, int[] colors, 
                                     int vertex, int m) {
        if (vertex == graph.length) {
            return true; // 모든 정점 색칠 완료
        }
        
        // 모든 색 시도
        for (int color = 0; color < m; color++) {
            if (isSafe(graph, colors, vertex, color)) {
                colors[vertex] = color;
                
                if (colorGraph(graph, colors, vertex + 1, m)) {
                    return true;
                }
                
                colors[vertex] = -1; // 백트랙
            }
        }
        
        return false;
    }
    
    private static boolean isSafe(int[][] graph, int[] colors, 
                                 int vertex, int color) {
        for (int i = 0; i < graph.length; i++) {
            if (graph[vertex][i] == 1 && colors[i] == color) {
                return false;
            }
        }
        return true;
    }
    
    // 최소 색상 수 찾기
    public static int chromaticNumber(int[][] graph) {
        int n = graph.length;
        
        // 최대 차수 + 1이 상한
        int maxDegree = 0;
        for (int i = 0; i < n; i++) {
            int degree = 0;
            for (int j = 0; j < n; j++) {
                if (graph[i][j] == 1) degree++;
            }
            maxDegree = Math.max(maxDegree, degree);
        }
        
        // 1부터 maxDegree + 1까지 시도
        for (int m = 1; m <= maxDegree + 1; m++) {
            if (graphColoring(graph, m)) {
                return m;
            }
        }
        
        return -1; // 불가능 (연결 그래프가 아닌 경우)
    }
    
    // 지도 색칠 문제 (평면 그래프는 4색으로 충분)
    static class Region {
        int id;
        List<Integer> neighbors;
        
        Region(int id) {
            this.id = id;
            this.neighbors = new ArrayList<>();
        }
    }
    
    public static boolean mapColoring(List<Region> regions, int maxColors) {
        int[] colors = new int[regions.size()];
        Arrays.fill(colors, -1);
        
        return colorMap(regions, colors, 0, maxColors);
    }
    
    private static boolean colorMap(List<Region> regions, int[] colors,
                                   int regionIndex, int maxColors) {
        if (regionIndex == regions.size()) {
            return true;
        }
        
        Region region = regions.get(regionIndex);
        
        for (int color = 0; color < maxColors; color++) {
            boolean canUseColor = true;
            
            // 인접 지역 확인
            for (int neighbor : region.neighbors) {
                if (colors[neighbor] == color) {
                    canUseColor = false;
                    break;
                }
            }
            
            if (canUseColor) {
                colors[region.id] = color;
                
                if (colorMap(regions, colors, regionIndex + 1, maxColors)) {
                    return true;
                }
                
                colors[region.id] = -1;
            }
        }
        
        return false;
    }
}
```

## 8.6 순열과 조합

### 순열과 조합 생성

#### Java 구현
```java
public class PermutationsAndCombinations {
    // 순열 생성
    public static List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        backtrackPermute(result, new ArrayList<>(), nums, new boolean[nums.length]);
        return result;
    }
    
    private static void backtrackPermute(List<List<Integer>> result,
                                        List<Integer> tempList,
                                        int[] nums, boolean[] used) {
        if (tempList.size() == nums.length) {
            result.add(new ArrayList<>(tempList));
            return;
        }
        
        for (int i = 0; i < nums.length; i++) {
            if (used[i]) continue;
            
            used[i] = true;
            tempList.add(nums[i]);
            backtrackPermute(result, tempList, nums, used);
            tempList.remove(tempList.size() - 1);
            used[i] = false;
        }
    }
    
    // 중복이 있는 순열
    public static List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        backtrackPermuteUnique(result, new ArrayList<>(), nums, new boolean[nums.length]);
        return result;
    }
    
    private static void backtrackPermuteUnique(List<List<Integer>> result,
                                              List<Integer> tempList,
                                              int[] nums, boolean[] used) {
        if (tempList.size() == nums.length) {
            result.add(new ArrayList<>(tempList));
            return;
        }
        
        for (int i = 0; i < nums.length; i++) {
            if (used[i]) continue;
            if (i > 0 && nums[i] == nums[i - 1] && !used[i - 1]) continue;
            
            used[i] = true;
            tempList.add(nums[i]);
            backtrackPermuteUnique(result, tempList, nums, used);
            tempList.remove(tempList.size() - 1);
            used[i] = false;
        }
    }
    
    // 조합 생성
    public static List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> result = new ArrayList<>();
        backtrackCombine(result, new ArrayList<>(), 1, n, k);
        return result;
    }
    
    private static void backtrackCombine(List<List<Integer>> result,
                                        List<Integer> tempList,
                                        int start, int n, int k) {
        if (tempList.size() == k) {
            result.add(new ArrayList<>(tempList));
            return;
        }
        
        for (int i = start; i <= n - (k - tempList.size()) + 1; i++) {
            tempList.add(i);
            backtrackCombine(result, tempList, i + 1, n, k);
            tempList.remove(tempList.size() - 1);
        }
    }
    
    // 조합의 합
    public static List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(candidates);
        backtrackCombSum(result, new ArrayList<>(), candidates, target, 0);
        return result;
    }
    
    private static void backtrackCombSum(List<List<Integer>> result,
                                        List<Integer> tempList,
                                        int[] candidates, int remain, int start) {
        if (remain < 0) return;
        if (remain == 0) {
            result.add(new ArrayList<>(tempList));
            return;
        }
        
        for (int i = start; i < candidates.length; i++) {
            tempList.add(candidates[i]);
            // 같은 원소 재사용 가능
            backtrackCombSum(result, tempList, candidates, 
                           remain - candidates[i], i);
            tempList.remove(tempList.size() - 1);
        }
    }
}
```

## 8.7 미로 찾기

### 미로에서 출구까지의 경로 찾기

#### Java 구현
```java
public class MazeSolver {
    private static final int[] dx = {0, 1, 0, -1}; // 오른쪽, 아래, 왼쪽, 위
    private static final int[] dy = {1, 0, -1, 0};
    
    // 미로 탈출 경로 찾기
    public static boolean solveMaze(int[][] maze, int startX, int startY, 
                                   int endX, int endY) {
        int m = maze.length;
        int n = maze[0].length;
        boolean[][] visited = new boolean[m][n];
        List<int[]> path = new ArrayList<>();
        
        return findPath(maze, startX, startY, endX, endY, visited, path);
    }
    
    private static boolean findPath(int[][] maze, int x, int y, 
                                   int endX, int endY,
                                   boolean[][] visited, List<int[]> path) {
        // 목적지 도달
        if (x == endX && y == endY) {
            path.add(new int[]{x, y});
            return true;
        }
        
        // 유효성 검사
        if (x < 0 || x >= maze.length || y < 0 || y >= maze[0].length ||
            maze[x][y] == 1 || visited[x][y]) {
            return false;
        }
        
        // 현재 위치 방문 표시
        visited[x][y] = true;
        path.add(new int[]{x, y});
        
        // 네 방향 탐색
        for (int i = 0; i < 4; i++) {
            int nx = x + dx[i];
            int ny = y + dy[i];
            
            if (findPath(maze, nx, ny, endX, endY, visited, path)) {
                return true;
            }
        }
        
        // 백트랙
        path.remove(path.size() - 1);
        return false;
    }
    
    // 모든 경로 찾기
    public static List<List<int[]>> findAllPaths(int[][] maze, 
                                                 int startX, int startY,
                                                 int endX, int endY) {
        List<List<int[]>> allPaths = new ArrayList<>();
        boolean[][] visited = new boolean[maze.length][maze[0].length];
        List<int[]> currentPath = new ArrayList<>();
        
        findAllPathsHelper(maze, startX, startY, endX, endY, 
                          visited, currentPath, allPaths);
        
        return allPaths;
    }
    
    private static void findAllPathsHelper(int[][] maze, int x, int y,
                                          int endX, int endY,
                                          boolean[][] visited,
                                          List<int[]> currentPath,
                                          List<List<int[]>> allPaths) {
        if (x < 0 || x >= maze.length || y < 0 || y >= maze[0].length ||
            maze[x][y] == 1 || visited[x][y]) {
            return;
        }
        
        visited[x][y] = true;
        currentPath.add(new int[]{x, y});
        
        if (x == endX && y == endY) {
            allPaths.add(new ArrayList<>(currentPath));
        } else {
            for (int i = 0; i < 4; i++) {
                findAllPathsHelper(maze, x + dx[i], y + dy[i], 
                                  endX, endY, visited, currentPath, allPaths);
            }
        }
        
        // 백트랙
        visited[x][y] = false;
        currentPath.remove(currentPath.size() - 1);
    }
    
    // 최단 경로 (BFS가 더 적합하지만 백트래킹으로 구현)
    public static List<int[]> shortestPath(int[][] maze, 
                                          int startX, int startY,
                                          int endX, int endY) {
        List<int[]> shortestPath = new ArrayList<>();
        boolean[][] visited = new boolean[maze.length][maze[0].length];
        List<int[]> currentPath = new ArrayList<>();
        int[] minLength = {Integer.MAX_VALUE};
        
        findShortestPath(maze, startX, startY, endX, endY, 
                        visited, currentPath, shortestPath, minLength);
        
        return shortestPath;
    }
    
    private static void findShortestPath(int[][] maze, int x, int y,
                                        int endX, int endY,
                                        boolean[][] visited,
                                        List<int[]> currentPath,
                                        List<int[]> shortestPath,
                                        int[] minLength) {
        if (x < 0 || x >= maze.length || y < 0 || y >= maze[0].length ||
            maze[x][y] == 1 || visited[x][y]) {
            return;
        }
        
        // 가지치기: 현재 경로가 이미 최단 경로보다 길면 중단
        if (currentPath.size() >= minLength[0]) {
            return;
        }
        
        visited[x][y] = true;
        currentPath.add(new int[]{x, y});
        
        if (x == endX && y == endY) {
            if (currentPath.size() < minLength[0]) {
                minLength[0] = currentPath.size();
                shortestPath.clear();
                shortestPath.addAll(new ArrayList<>(currentPath));
            }
        } else {
            for (int i = 0; i < 4; i++) {
                findShortestPath(maze, x + dx[i], y + dy[i], 
                               endX, endY, visited, currentPath, 
                               shortestPath, minLength);
            }
        }
        
        visited[x][y] = false;
        currentPath.remove(currentPath.size() - 1);
    }
}
```

## 8.8 백트래킹 최적화 기법

### 1. 가지치기 (Pruning)
조건에 맞지 않는 경로를 조기에 차단

### 2. 휴리스틱 (Heuristics)
가능성이 높은 경로를 우선 탐색

### 3. 메모이제이션
중복 계산 결과 저장

#### 최적화 예제
```java
public class BacktrackingOptimization {
    // 최적화된 부분집합 합 (가지치기 + 정렬)
    public static boolean optimizedSubsetSum(int[] nums, int target) {
        Arrays.sort(nums); // 정렬로 가지치기 효율 향상
        return findSubsetSum(nums, 0, 0, target);
    }
    
    private static boolean findSubsetSum(int[] nums, int index, 
                                        int currentSum, int target) {
        if (currentSum == target) return true;
        if (index >= nums.length || currentSum > target) return false;
        
        // 남은 모든 원소를 더해도 target에 못 미치면 중단
        int remainingSum = 0;
        for (int i = index; i < nums.length; i++) {
            remainingSum += nums[i];
        }
        if (currentSum + remainingSum < target) return false;
        
        // 포함
        if (findSubsetSum(nums, index + 1, currentSum + nums[index], target)) {
            return true;
        }
        
        // 미포함
        return findSubsetSum(nums, index + 1, currentSum, target);
    }
    
    // 메모이제이션을 활용한 백트래킹
    public static boolean wordBreak(String s, List<String> wordDict) {
        Set<String> wordSet = new HashSet<>(wordDict);
        Map<Integer, Boolean> memo = new HashMap<>();
        return wordBreakHelper(s, 0, wordSet, memo);
    }
    
    private static boolean wordBreakHelper(String s, int start, 
                                          Set<String> wordSet,
                                          Map<Integer, Boolean> memo) {
        if (start == s.length()) return true;
        if (memo.containsKey(start)) return memo.get(start);
        
        for (int end = start + 1; end <= s.length(); end++) {
            String word = s.substring(start, end);
            if (wordSet.contains(word) && 
                wordBreakHelper(s, end, wordSet, memo)) {
                memo.put(start, true);
                return true;
            }
        }
        
        memo.put(start, false);
        return false;
    }
}
```

## 8.9 실습 문제

1. **나이트의 여행**: 체스판에서 나이트가 모든 칸을 한 번씩 방문
2. **암호문 해독**: 숫자를 문자로 변환하는 모든 방법
3. **표현식 계산**: 괄호를 추가하여 다른 결과 얻기
4. **타일링 문제**: 2×n 보드를 타일로 채우는 방법

## 8.10 요약

- 백트래킹은 가능한 모든 해를 체계적으로 탐색
- 조건에 맞지 않으면 이전 단계로 되돌아감
- 가지치기로 불필요한 탐색을 줄여 효율성 향상
- N-Queens, 스도쿠, 부분집합, 순열/조합 등에 활용
- DFS와 유사하지만 상태 복원이 핵심