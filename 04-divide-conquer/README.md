# Chapter 4: 분할 정복 (Divide and Conquer)

## 4.1 분할 정복이란?

### 정의
분할 정복은 큰 문제를 작은 부분 문제로 나누어 해결한 후, 그 결과를 합쳐서 전체 문제의 답을 구하는 알고리즘 설계 패러다임입니다.

### 분할 정복의 3단계
1. **분할(Divide)**: 문제를 더 작은 부분 문제로 분할
2. **정복(Conquer)**: 부분 문제를 재귀적으로 해결
3. **결합(Combine)**: 부분 문제의 해를 결합하여 원래 문제의 해 구성

### 분할 정복의 조건
- 문제를 더 작은 부분 문제로 분할 가능
- 부분 문제들이 원래 문제와 동일한 형태
- 부분 문제의 해를 결합하여 원래 문제 해결 가능

## 4.2 병합 정렬 심화

### 병합 정렬의 원리
배열을 반으로 나누고, 각각을 정렬한 후 병합

#### Java 구현
```java
public class MergeSortAdvanced {
    // 기본 병합 정렬
    public static void mergeSort(int[] arr) {
        if (arr.length > 1) {
            mergeSort(arr, 0, arr.length - 1);
        }
    }
    
    private static void mergeSort(int[] arr, int left, int right) {
        if (left < right) {
            int mid = left + (right - left) / 2;
            
            // 분할
            mergeSort(arr, left, mid);
            mergeSort(arr, mid + 1, right);
            
            // 결합
            merge(arr, left, mid, right);
        }
    }
    
    private static void merge(int[] arr, int left, int mid, int right) {
        int n1 = mid - left + 1;
        int n2 = right - mid;
        
        int[] L = new int[n1];
        int[] R = new int[n2];
        
        System.arraycopy(arr, left, L, 0, n1);
        System.arraycopy(arr, mid + 1, R, 0, n2);
        
        int i = 0, j = 0, k = left;
        
        while (i < n1 && j < n2) {
            if (L[i] <= R[j]) {
                arr[k++] = L[i++];
            } else {
                arr[k++] = R[j++];
            }
        }
        
        while (i < n1) arr[k++] = L[i++];
        while (j < n2) arr[k++] = R[j++];
    }
    
    // 역순 쌍 개수 구하기
    public static long countInversions(int[] arr) {
        return mergeSortAndCount(arr, 0, arr.length - 1);
    }
    
    private static long mergeSortAndCount(int[] arr, int left, int right) {
        long count = 0;
        
        if (left < right) {
            int mid = left + (right - left) / 2;
            
            count += mergeSortAndCount(arr, left, mid);
            count += mergeSortAndCount(arr, mid + 1, right);
            count += mergeAndCount(arr, left, mid, right);
        }
        
        return count;
    }
    
    private static long mergeAndCount(int[] arr, int left, int mid, int right) {
        int[] temp = new int[right - left + 1];
        int i = left, j = mid + 1, k = 0;
        long invCount = 0;
        
        while (i <= mid && j <= right) {
            if (arr[i] <= arr[j]) {
                temp[k++] = arr[i++];
            } else {
                temp[k++] = arr[j++];
                invCount += (mid - i + 1); // 역순 쌍 개수
            }
        }
        
        while (i <= mid) temp[k++] = arr[i++];
        while (j <= right) temp[k++] = arr[j++];
        
        System.arraycopy(temp, 0, arr, left, temp.length);
        return invCount;
    }
    
    // K개 정렬된 배열 병합
    public static int[] mergeKSortedArrays(int[][] arrays) {
        if (arrays.length == 0) return new int[0];
        return mergeKArraysHelper(arrays, 0, arrays.length - 1);
    }
    
    private static int[] mergeKArraysHelper(int[][] arrays, int left, int right) {
        if (left == right) {
            return arrays[left];
        }
        
        int mid = left + (right - left) / 2;
        int[] leftMerged = mergeKArraysHelper(arrays, left, mid);
        int[] rightMerged = mergeKArraysHelper(arrays, mid + 1, right);
        
        return mergeTwoArrays(leftMerged, rightMerged);
    }
    
    private static int[] mergeTwoArrays(int[] arr1, int[] arr2) {
        int[] result = new int[arr1.length + arr2.length];
        int i = 0, j = 0, k = 0;
        
        while (i < arr1.length && j < arr2.length) {
            if (arr1[i] <= arr2[j]) {
                result[k++] = arr1[i++];
            } else {
                result[k++] = arr2[j++];
            }
        }
        
        while (i < arr1.length) result[k++] = arr1[i++];
        while (j < arr2.length) result[k++] = arr2[j++];
        
        return result;
    }
}
```

#### C 구현
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 병합 함수
void merge(int arr[], int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;
    
    int* L = (int*)malloc(n1 * sizeof(int));
    int* R = (int*)malloc(n2 * sizeof(int));
    
    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];
    
    int i = 0, j = 0, k = left;
    
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k++] = L[i++];
        } else {
            arr[k++] = R[j++];
        }
    }
    
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
    
    free(L);
    free(R);
}

// 병합 정렬
void mergeSort(int arr[], int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

// 역순 쌍 개수 구하기
long long mergeAndCount(int arr[], int temp[], int left, int mid, int right) {
    int i = left, j = mid + 1, k = left;
    long long invCount = 0;
    
    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
            invCount += (mid - i + 1);
        }
    }
    
    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];
    
    for (i = left; i <= right; i++)
        arr[i] = temp[i];
    
    return invCount;
}

long long mergeSortAndCount(int arr[], int temp[], int left, int right) {
    long long invCount = 0;
    
    if (left < right) {
        int mid = left + (right - left) / 2;
        
        invCount += mergeSortAndCount(arr, temp, left, mid);
        invCount += mergeSortAndCount(arr, temp, mid + 1, right);
        invCount += mergeAndCount(arr, temp, left, mid, right);
    }
    
    return invCount;
}
```

## 4.3 퀵 정렬 심화

### 다양한 피벗 선택 전략

#### Java 구현
```java
public class QuickSortAdvanced {
    // 무작위 피벗 퀵 정렬
    public static void randomizedQuickSort(int[] arr) {
        randomizedQuickSort(arr, 0, arr.length - 1);
    }
    
    private static void randomizedQuickSort(int[] arr, int low, int high) {
        if (low < high) {
            int pi = randomizedPartition(arr, low, high);
            randomizedQuickSort(arr, low, pi - 1);
            randomizedQuickSort(arr, pi + 1, high);
        }
    }
    
    private static int randomizedPartition(int[] arr, int low, int high) {
        int random = low + (int)(Math.random() * (high - low + 1));
        swap(arr, random, high);
        return partition(arr, low, high);
    }
    
    // 중간값 피벗 (Median of Three)
    public static void medianOfThreeQuickSort(int[] arr) {
        medianOfThreeQuickSort(arr, 0, arr.length - 1);
    }
    
    private static void medianOfThreeQuickSort(int[] arr, int low, int high) {
        if (low < high) {
            int pivotIndex = medianOfThree(arr, low, high);
            swap(arr, pivotIndex, high);
            int pi = partition(arr, low, high);
            medianOfThreeQuickSort(arr, low, pi - 1);
            medianOfThreeQuickSort(arr, pi + 1, high);
        }
    }
    
    private static int medianOfThree(int[] arr, int low, int high) {
        int mid = low + (high - low) / 2;
        
        if (arr[low] > arr[mid]) swap(arr, low, mid);
        if (arr[low] > arr[high]) swap(arr, low, high);
        if (arr[mid] > arr[high]) swap(arr, mid, high);
        
        return mid;
    }
    
    private static int partition(int[] arr, int low, int high) {
        int pivot = arr[high];
        int i = low - 1;
        
        for (int j = low; j < high; j++) {
            if (arr[j] < pivot) {
                i++;
                swap(arr, i, j);
            }
        }
        
        swap(arr, i + 1, high);
        return i + 1;
    }
    
    private static void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
    
    // K번째 작은 원소 찾기 (Quick Select)
    public static int quickSelect(int[] arr, int k) {
        return quickSelect(arr, 0, arr.length - 1, k - 1);
    }
    
    private static int quickSelect(int[] arr, int low, int high, int k) {
        if (low == high) {
            return arr[low];
        }
        
        int pivotIndex = randomizedPartition(arr, low, high);
        
        if (k == pivotIndex) {
            return arr[k];
        } else if (k < pivotIndex) {
            return quickSelect(arr, low, pivotIndex - 1, k);
        } else {
            return quickSelect(arr, pivotIndex + 1, high, k);
        }
    }
    
    // 3-way 퀵 정렬 (Dutch National Flag)
    public static void threeWayQuickSort(int[] arr) {
        threeWayQuickSort(arr, 0, arr.length - 1);
    }
    
    private static void threeWayQuickSort(int[] arr, int low, int high) {
        if (low >= high) return;
        
        int[] partitionIndices = threeWayPartition(arr, low, high);
        int lt = partitionIndices[0];
        int gt = partitionIndices[1];
        
        threeWayQuickSort(arr, low, lt - 1);
        threeWayQuickSort(arr, gt + 1, high);
    }
    
    private static int[] threeWayPartition(int[] arr, int low, int high) {
        int pivot = arr[low];
        int i = low, lt = low, gt = high;
        
        while (i <= gt) {
            if (arr[i] < pivot) {
                swap(arr, lt++, i++);
            } else if (arr[i] > pivot) {
                swap(arr, i, gt--);
            } else {
                i++;
            }
        }
        
        return new int[]{lt, gt};
    }
}
```

## 4.4 이진 탐색의 응용

### 다양한 이진 탐색 문제

#### Java 구현
```java
public class BinarySearchApplications {
    // 제곱근 구하기
    public static double sqrt(double x, double epsilon) {
        if (x < 0) throw new IllegalArgumentException();
        if (x == 0 || x == 1) return x;
        
        double low = 0, high = x;
        if (x < 1) high = 1;
        
        while (high - low > epsilon) {
            double mid = low + (high - low) / 2;
            double square = mid * mid;
            
            if (square > x) {
                high = mid;
            } else {
                low = mid;
            }
        }
        
        return low + (high - low) / 2;
    }
    
    // 피크 찾기 (2D)
    public static int[] findPeakElement2D(int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
        int low = 0, high = n - 1;
        
        while (low <= high) {
            int midCol = low + (high - low) / 2;
            int maxRow = findMaxInColumn(matrix, midCol);
            
            boolean leftBigger = midCol > 0 && 
                matrix[maxRow][midCol - 1] > matrix[maxRow][midCol];
            boolean rightBigger = midCol < n - 1 && 
                matrix[maxRow][midCol + 1] > matrix[maxRow][midCol];
            
            if (!leftBigger && !rightBigger) {
                return new int[]{maxRow, midCol};
            } else if (leftBigger) {
                high = midCol - 1;
            } else {
                low = midCol + 1;
            }
        }
        
        return new int[]{-1, -1};
    }
    
    private static int findMaxInColumn(int[][] matrix, int col) {
        int maxRow = 0;
        for (int i = 1; i < matrix.length; i++) {
            if (matrix[i][col] > matrix[maxRow][col]) {
                maxRow = i;
            }
        }
        return maxRow;
    }
    
    // 행렬에서 검색 (행과 열이 정렬됨)
    public static boolean searchMatrix(int[][] matrix, int target) {
        if (matrix.length == 0 || matrix[0].length == 0) return false;
        
        int m = matrix.length;
        int n = matrix[0].length;
        int row = 0, col = n - 1;
        
        while (row < m && col >= 0) {
            if (matrix[row][col] == target) {
                return true;
            } else if (matrix[row][col] > target) {
                col--;
            } else {
                row++;
            }
        }
        
        return false;
    }
    
    // 중앙값 찾기 (두 정렬된 배열)
    public static double findMedianSortedArrays(int[] nums1, int[] nums2) {
        if (nums1.length > nums2.length) {
            return findMedianSortedArrays(nums2, nums1);
        }
        
        int m = nums1.length;
        int n = nums2.length;
        int low = 0, high = m;
        
        while (low <= high) {
            int partitionX = (low + high) / 2;
            int partitionY = (m + n + 1) / 2 - partitionX;
            
            int maxLeftX = (partitionX == 0) ? Integer.MIN_VALUE : 
                           nums1[partitionX - 1];
            int minRightX = (partitionX == m) ? Integer.MAX_VALUE : 
                            nums1[partitionX];
            
            int maxLeftY = (partitionY == 0) ? Integer.MIN_VALUE : 
                           nums2[partitionY - 1];
            int minRightY = (partitionY == n) ? Integer.MAX_VALUE : 
                            nums2[partitionY];
            
            if (maxLeftX <= minRightY && maxLeftY <= minRightX) {
                if ((m + n) % 2 == 0) {
                    return (Math.max(maxLeftX, maxLeftY) + 
                            Math.min(minRightX, minRightY)) / 2.0;
                } else {
                    return Math.max(maxLeftX, maxLeftY);
                }
            } else if (maxLeftX > minRightY) {
                high = partitionX - 1;
            } else {
                low = partitionX + 1;
            }
        }
        
        throw new IllegalArgumentException();
    }
}
```

## 4.5 최근접 점 쌍 문제

### 분할 정복으로 해결

#### Java 구현
```java
import java.util.*;

public class ClosestPairOfPoints {
    static class Point {
        double x, y;
        
        Point(double x, double y) {
            this.x = x;
            this.y = y;
        }
    }
    
    // 두 점 사이의 거리
    private static double distance(Point p1, Point p2) {
        double dx = p1.x - p2.x;
        double dy = p1.y - p2.y;
        return Math.sqrt(dx * dx + dy * dy);
    }
    
    // 브루트 포스 (작은 경우)
    private static double bruteForce(Point[] points, int start, int end) {
        double minDist = Double.MAX_VALUE;
        
        for (int i = start; i < end; i++) {
            for (int j = i + 1; j <= end; j++) {
                minDist = Math.min(minDist, distance(points[i], points[j]));
            }
        }
        
        return minDist;
    }
    
    // 스트립에서 최소 거리 찾기
    private static double stripClosest(Point[] strip, double d) {
        double minDist = d;
        
        // Y 좌표로 정렬
        Arrays.sort(strip, (a, b) -> Double.compare(a.y, b.y));
        
        for (int i = 0; i < strip.length; i++) {
            for (int j = i + 1; j < strip.length && 
                 (strip[j].y - strip[i].y) < minDist; j++) {
                minDist = Math.min(minDist, distance(strip[i], strip[j]));
            }
        }
        
        return minDist;
    }
    
    // 재귀적 분할 정복
    private static double closestUtil(Point[] px, Point[] py, int n) {
        // 작은 경우 브루트 포스
        if (n <= 3) {
            return bruteForce(px, 0, n - 1);
        }
        
        int mid = n / 2;
        Point midPoint = px[mid];
        
        // Y 좌표로 정렬된 배열을 두 부분으로 분할
        Point[] pyl = new Point[mid + 1];
        Point[] pyr = new Point[n - mid - 1];
        int li = 0, ri = 0;
        
        for (int i = 0; i < n; i++) {
            if (py[i].x <= midPoint.x) {
                pyl[li++] = py[i];
            } else {
                pyr[ri++] = py[i];
            }
        }
        
        // 재귀 호출
        double dl = closestUtil(px, pyl, mid + 1);
        double dr = closestUtil(Arrays.copyOfRange(px, mid + 1, n), 
                               pyr, n - mid - 1);
        
        double d = Math.min(dl, dr);
        
        // 중간선 근처의 점들
        List<Point> stripList = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (Math.abs(py[i].x - midPoint.x) < d) {
                stripList.add(py[i]);
            }
        }
        
        Point[] strip = stripList.toArray(new Point[0]);
        return Math.min(d, stripClosest(strip, d));
    }
    
    // 메인 함수
    public static double closestPair(Point[] points) {
        int n = points.length;
        Point[] px = Arrays.copyOf(points, n);
        Point[] py = Arrays.copyOf(points, n);
        
        Arrays.sort(px, (a, b) -> Double.compare(a.x, b.x));
        Arrays.sort(py, (a, b) -> Double.compare(a.y, b.y));
        
        return closestUtil(px, py, n);
    }
}
```

## 4.6 Strassen 행렬 곱셈

### 분할 정복을 이용한 행렬 곱셈 최적화

#### Java 구현
```java
public class StrassenMatrixMultiplication {
    // 일반 행렬 곱셈 - O(n³)
    public static int[][] multiply(int[][] A, int[][] B) {
        int n = A.length;
        int[][] C = new int[n][n];
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        
        return C;
    }
    
    // Strassen 알고리즘 - O(n^2.807)
    public static int[][] strassen(int[][] A, int[][] B) {
        int n = A.length;
        
        if (n <= 64) { // 작은 경우 일반 곱셈
            return multiply(A, B);
        }
        
        // 홀수 크기 처리
        if (n % 2 != 0) {
            int[][] newA = new int[n + 1][n + 1];
            int[][] newB = new int[n + 1][n + 1];
            
            for (int i = 0; i < n; i++) {
                System.arraycopy(A[i], 0, newA[i], 0, n);
                System.arraycopy(B[i], 0, newB[i], 0, n);
            }
            
            int[][] C = strassen(newA, newB);
            int[][] result = new int[n][n];
            
            for (int i = 0; i < n; i++) {
                System.arraycopy(C[i], 0, result[i], 0, n);
            }
            
            return result;
        }
        
        int newSize = n / 2;
        
        // 부분 행렬 생성
        int[][] A11 = new int[newSize][newSize];
        int[][] A12 = new int[newSize][newSize];
        int[][] A21 = new int[newSize][newSize];
        int[][] A22 = new int[newSize][newSize];
        
        int[][] B11 = new int[newSize][newSize];
        int[][] B12 = new int[newSize][newSize];
        int[][] B21 = new int[newSize][newSize];
        int[][] B22 = new int[newSize][newSize];
        
        // 부분 행렬로 분할
        split(A, A11, 0, 0);
        split(A, A12, 0, newSize);
        split(A, A21, newSize, 0);
        split(A, A22, newSize, newSize);
        
        split(B, B11, 0, 0);
        split(B, B12, 0, newSize);
        split(B, B21, newSize, 0);
        split(B, B22, newSize, newSize);
        
        // Strassen의 7개 곱셈
        int[][] M1 = strassen(add(A11, A22), add(B11, B22));
        int[][] M2 = strassen(add(A21, A22), B11);
        int[][] M3 = strassen(A11, subtract(B12, B22));
        int[][] M4 = strassen(A22, subtract(B21, B11));
        int[][] M5 = strassen(add(A11, A12), B22);
        int[][] M6 = strassen(subtract(A21, A11), add(B11, B12));
        int[][] M7 = strassen(subtract(A12, A22), add(B21, B22));
        
        // 결과 계산
        int[][] C11 = add(subtract(add(M1, M4), M5), M7);
        int[][] C12 = add(M3, M5);
        int[][] C21 = add(M2, M4);
        int[][] C22 = add(subtract(add(M1, M3), M2), M6);
        
        // 결과 합치기
        int[][] C = new int[n][n];
        join(C11, C, 0, 0);
        join(C12, C, 0, newSize);
        join(C21, C, newSize, 0);
        join(C22, C, newSize, newSize);
        
        return C;
    }
    
    // 행렬 분할
    private static void split(int[][] P, int[][] C, int iB, int jB) {
        for (int i = 0; i < C.length; i++) {
            for (int j = 0; j < C.length; j++) {
                C[i][j] = P[i + iB][j + jB];
            }
        }
    }
    
    // 행렬 합치기
    private static void join(int[][] C, int[][] P, int iB, int jB) {
        for (int i = 0; i < C.length; i++) {
            for (int j = 0; j < C.length; j++) {
                P[i + iB][j + jB] = C[i][j];
            }
        }
    }
    
    // 행렬 덧셈
    private static int[][] add(int[][] A, int[][] B) {
        int n = A.length;
        int[][] C = new int[n][n];
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = A[i][j] + B[i][j];
            }
        }
        
        return C;
    }
    
    // 행렬 뺄셈
    private static int[][] subtract(int[][] A, int[][] B) {
        int n = A.length;
        int[][] C = new int[n][n];
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = A[i][j] - B[i][j];
            }
        }
        
        return C;
    }
}
```

## 4.7 Karatsuba 알고리즘

### 큰 수의 곱셈

#### Java 구현
```java
import java.math.BigInteger;

public class KaratsubaMultiplication {
    // Karatsuba 곱셈
    public static BigInteger karatsuba(BigInteger x, BigInteger y) {
        int n = Math.max(x.bitLength(), y.bitLength());
        
        // 작은 경우 일반 곱셈
        if (n <= 32) {
            return x.multiply(y);
        }
        
        // n을 반으로 나누기
        n = (n / 2) + (n % 2);
        
        // x = a * 2^n + b, y = c * 2^n + d
        BigInteger b = x.and(BigInteger.ONE.shiftLeft(n).subtract(BigInteger.ONE));
        BigInteger a = x.shiftRight(n);
        BigInteger d = y.and(BigInteger.ONE.shiftLeft(n).subtract(BigInteger.ONE));
        BigInteger c = y.shiftRight(n);
        
        // 재귀 호출
        BigInteger ac = karatsuba(a, c);
        BigInteger bd = karatsuba(b, d);
        BigInteger abcd = karatsuba(a.add(b), c.add(d));
        
        return ac.shiftLeft(2 * n)
                 .add(abcd.subtract(ac).subtract(bd).shiftLeft(n))
                 .add(bd);
    }
    
    // 문자열로 표현된 큰 수의 곱셈
    public static String multiplyStrings(String num1, String num2) {
        BigInteger a = new BigInteger(num1);
        BigInteger b = new BigInteger(num2);
        return karatsuba(a, b).toString();
    }
}
```

## 4.8 분할 정복 문제 해결 전략

### 마스터 정리 (Master Theorem)
T(n) = aT(n/b) + f(n) 형태의 재귀식 분석

1. f(n) = O(n^(log_b(a) - ε)) → T(n) = Θ(n^(log_b(a)))
2. f(n) = Θ(n^(log_b(a))) → T(n) = Θ(n^(log_b(a)) * log n)
3. f(n) = Ω(n^(log_b(a) + ε)) → T(n) = Θ(f(n))

### 분할 정복 설계 패턴
1. **문제 분석**: 분할 가능성 확인
2. **분할 전략**: 어떻게 나눌 것인가
3. **기저 사례**: 언제 재귀를 멈출 것인가
4. **결합 방법**: 부분 해를 어떻게 합칠 것인가
5. **최적화**: 중복 계산 제거, 캐싱

## 4.9 실습 문제

1. **최대 부분 배열**: 분할 정복으로 최대 부분 배열 합 구하기
2. **거듭제곱**: O(log n) 시간에 x^n 계산하기
3. **역순 쌍**: 배열에서 i < j이면서 arr[i] > arr[j]인 쌍의 개수
4. **스카이라인 문제**: 건물들의 스카이라인 구하기

## 4.10 요약

- 분할 정복은 문제를 작은 부분으로 나누어 해결하는 강력한 패러다임
- 병합 정렬과 퀵 정렬은 대표적인 분할 정복 정렬 알고리즘
- 이진 탐색은 정렬된 데이터에서 효율적인 탐색을 제공
- 행렬 곱셈, 큰 수 곱셈 등 다양한 문제에 적용 가능
- 마스터 정리를 통해 시간 복잡도 분석 가능