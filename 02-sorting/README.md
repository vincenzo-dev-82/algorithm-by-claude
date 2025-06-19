# Chapter 2: 정렬 알고리즘 (Sorting Algorithms)

## 2.1 정렬이란?

### 정의
정렬(Sorting)은 데이터를 특정 순서대로 재배열하는 과정입니다. 일반적으로 오름차순이나 내림차순으로 정렬합니다.

### 정렬의 중요성
- 탐색 효율성 향상 (이진 탐색 가능)
- 데이터 분석 용이
- 중복 제거 간편
- 다른 알고리즘의 전처리 단계

### 정렬 알고리즘의 분류
1. **비교 기반 정렬**: 원소 간 비교를 통해 정렬
2. **비비교 기반 정렬**: 원소의 특성을 이용한 정렬
3. **안정 정렬**: 같은 값의 원소들의 상대적 순서 유지
4. **불안정 정렬**: 같은 값의 원소들의 순서가 바뀔 수 있음

## 2.2 O(n²) 정렬 알고리즘

### 버블 정렬 (Bubble Sort)
인접한 두 원소를 비교하여 정렬하는 가장 단순한 알고리즘

#### Java 구현
```java
public class BubbleSort {
    // 기본 버블 정렬
    public static void bubbleSort(int[] arr) {
        int n = arr.length;
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (arr[j] > arr[j + 1]) {
                    // swap
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                }
            }
        }
    }
    
    // 최적화된 버블 정렬 (조기 종료)
    public static void optimizedBubbleSort(int[] arr) {
        int n = arr.length;
        for (int i = 0; i < n - 1; i++) {
            boolean swapped = false;
            for (int j = 0; j < n - i - 1; j++) {
                if (arr[j] > arr[j + 1]) {
                    // swap
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                    swapped = true;
                }
            }
            // 교환이 없으면 이미 정렬됨
            if (!swapped) break;
        }
    }
}
```

#### C 구현
```c
#include <stdio.h>
#include <stdbool.h>

// 기본 버블 정렬
void bubbleSort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                // swap
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

// 최적화된 버블 정렬
void optimizedBubbleSort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        bool swapped = false;
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
                swapped = true;
            }
        }
        if (!swapped) break;
    }
}
```

### 선택 정렬 (Selection Sort)
최솟값을 찾아 맨 앞으로 이동시키는 정렬

#### Java 구현
```java
public class SelectionSort {
    public static void selectionSort(int[] arr) {
        int n = arr.length;
        
        for (int i = 0; i < n - 1; i++) {
            int minIdx = i;
            
            // 최솟값 찾기
            for (int j = i + 1; j < n; j++) {
                if (arr[j] < arr[minIdx]) {
                    minIdx = j;
                }
            }
            
            // 최솟값을 맨 앞으로 이동
            if (minIdx != i) {
                int temp = arr[i];
                arr[i] = arr[minIdx];
                arr[minIdx] = temp;
            }
        }
    }
}
```

#### C 구현
```c
void selectionSort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        int minIdx = i;
        
        // 최솟값 찾기
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[minIdx]) {
                minIdx = j;
            }
        }
        
        // 최솟값을 맨 앞으로 이동
        if (minIdx != i) {
            int temp = arr[i];
            arr[i] = arr[minIdx];
            arr[minIdx] = temp;
        }
    }
}
```

### 삽입 정렬 (Insertion Sort)
정렬된 부분에 새로운 원소를 적절한 위치에 삽입

#### Java 구현
```java
public class InsertionSort {
    public static void insertionSort(int[] arr) {
        int n = arr.length;
        
        for (int i = 1; i < n; i++) {
            int key = arr[i];
            int j = i - 1;
            
            // key보다 큰 원소들을 오른쪽으로 이동
            while (j >= 0 && arr[j] > key) {
                arr[j + 1] = arr[j];
                j--;
            }
            
            // key를 적절한 위치에 삽입
            arr[j + 1] = key;
        }
    }
    
    // 이진 삽입 정렬 (삽입 위치를 이진 탐색으로 찾기)
    public static void binaryInsertionSort(int[] arr) {
        int n = arr.length;
        
        for (int i = 1; i < n; i++) {
            int key = arr[i];
            int left = 0;
            int right = i;
            
            // 이진 탐색으로 삽입 위치 찾기
            while (left < right) {
                int mid = left + (right - left) / 2;
                if (arr[mid] > key) {
                    right = mid;
                } else {
                    left = mid + 1;
                }
            }
            
            // 원소들을 이동하고 삽입
            for (int j = i - 1; j >= left; j--) {
                arr[j + 1] = arr[j];
            }
            arr[left] = key;
        }
    }
}
```

#### C 구현
```c
void insertionSort(int arr[], int n) {
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        
        arr[j + 1] = key;
    }
}
```

## 2.3 O(n log n) 정렬 알고리즘

### 병합 정렬 (Merge Sort)
분할 정복 방식으로 배열을 반으로 나누고 정렬 후 병합

#### Java 구현
```java
public class MergeSort {
    public static void mergeSort(int[] arr) {
        if (arr.length > 1) {
            mergeSortHelper(arr, 0, arr.length - 1);
        }
    }
    
    private static void mergeSortHelper(int[] arr, int left, int right) {
        if (left < right) {
            int mid = left + (right - left) / 2;
            
            // 왼쪽 반 정렬
            mergeSortHelper(arr, left, mid);
            
            // 오른쪽 반 정렬
            mergeSortHelper(arr, mid + 1, right);
            
            // 병합
            merge(arr, left, mid, right);
        }
    }
    
    private static void merge(int[] arr, int left, int mid, int right) {
        // 임시 배열 생성
        int[] temp = new int[right - left + 1];
        int i = left, j = mid + 1, k = 0;
        
        // 두 부분 배열을 비교하며 병합
        while (i <= mid && j <= right) {
            if (arr[i] <= arr[j]) {
                temp[k++] = arr[i++];
            } else {
                temp[k++] = arr[j++];
            }
        }
        
        // 남은 원소들 복사
        while (i <= mid) {
            temp[k++] = arr[i++];
        }
        while (j <= right) {
            temp[k++] = arr[j++];
        }
        
        // 임시 배열을 원본에 복사
        for (i = 0; i < temp.length; i++) {
            arr[left + i] = temp[i];
        }
    }
}
```

#### C 구현
```c
#include <stdlib.h>

void merge(int arr[], int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;
    
    // 임시 배열 생성
    int* L = (int*)malloc(n1 * sizeof(int));
    int* R = (int*)malloc(n2 * sizeof(int));
    
    // 데이터 복사
    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];
    
    // 병합
    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k++] = L[i++];
        } else {
            arr[k++] = R[j++];
        }
    }
    
    // 남은 원소들 복사
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
    
    free(L);
    free(R);
}

void mergeSortHelper(int arr[], int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        
        mergeSortHelper(arr, left, mid);
        mergeSortHelper(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

void mergeSort(int arr[], int n) {
    mergeSortHelper(arr, 0, n - 1);
}
```

### 퀵 정렬 (Quick Sort)
피벗을 기준으로 작은 값은 왼쪽, 큰 값은 오른쪽으로 분할

#### Java 구현
```java
public class QuickSort {
    // 기본 퀵 정렬
    public static void quickSort(int[] arr) {
        if (arr.length > 1) {
            quickSortHelper(arr, 0, arr.length - 1);
        }
    }
    
    private static void quickSortHelper(int[] arr, int low, int high) {
        if (low < high) {
            int pi = partition(arr, low, high);
            
            quickSortHelper(arr, low, pi - 1);
            quickSortHelper(arr, pi + 1, high);
        }
    }
    
    private static int partition(int[] arr, int low, int high) {
        int pivot = arr[high];  // 마지막 원소를 피벗으로
        int i = low - 1;
        
        for (int j = low; j < high; j++) {
            if (arr[j] < pivot) {
                i++;
                // swap
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
        
        // 피벗을 올바른 위치로
        i++;
        int temp = arr[i];
        arr[i] = arr[high];
        arr[high] = temp;
        
        return i;
    }
    
    // 3-way 퀵 정렬 (중복이 많을 때 효율적)
    public static void quickSort3Way(int[] arr) {
        quickSort3WayHelper(arr, 0, arr.length - 1);
    }
    
    private static void quickSort3WayHelper(int[] arr, int low, int high) {
        if (low >= high) return;
        
        int lt = low, gt = high;
        int pivot = arr[low];
        int i = low + 1;
        
        while (i <= gt) {
            if (arr[i] < pivot) {
                swap(arr, lt++, i++);
            } else if (arr[i] > pivot) {
                swap(arr, i, gt--);
            } else {
                i++;
            }
        }
        
        quickSort3WayHelper(arr, low, lt - 1);
        quickSort3WayHelper(arr, gt + 1, high);
    }
    
    private static void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}
```

#### C 구현
```c
void swap(int* a, int* b) {
    int t = *a;
    *a = *b;
    *b = t;
}

int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = low - 1;
    
    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    
    swap(&arr[i + 1], &arr[high]);
    return i + 1;
}

void quickSortHelper(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        
        quickSortHelper(arr, low, pi - 1);
        quickSortHelper(arr, pi + 1, high);
    }
}

void quickSort(int arr[], int n) {
    quickSortHelper(arr, 0, n - 1);
}
```

### 힙 정렬 (Heap Sort)
최대 힙을 구성하고 루트를 제거하며 정렬

#### Java 구현
```java
public class HeapSort {
    public static void heapSort(int[] arr) {
        int n = arr.length;
        
        // 최대 힙 구성
        for (int i = n / 2 - 1; i >= 0; i--) {
            heapify(arr, n, i);
        }
        
        // 힙에서 원소를 하나씩 추출
        for (int i = n - 1; i > 0; i--) {
            // 루트(최댓값)를 끝으로 이동
            int temp = arr[0];
            arr[0] = arr[i];
            arr[i] = temp;
            
            // 힙 속성 복구
            heapify(arr, i, 0);
        }
    }
    
    private static void heapify(int[] arr, int n, int i) {
        int largest = i;
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        
        if (left < n && arr[left] > arr[largest]) {
            largest = left;
        }
        
        if (right < n && arr[right] > arr[largest]) {
            largest = right;
        }
        
        if (largest != i) {
            int temp = arr[i];
            arr[i] = arr[largest];
            arr[largest] = temp;
            
            heapify(arr, n, largest);
        }
    }
}
```

#### C 구현
```c
void heapify(int arr[], int n, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;
    
    if (left < n && arr[left] > arr[largest])
        largest = left;
    
    if (right < n && arr[right] > arr[largest])
        largest = right;
    
    if (largest != i) {
        swap(&arr[i], &arr[largest]);
        heapify(arr, n, largest);
    }
}

void heapSort(int arr[], int n) {
    // 최대 힙 구성
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i);
    
    // 힙에서 원소를 하나씩 추출
    for (int i = n - 1; i > 0; i--) {
        swap(&arr[0], &arr[i]);
        heapify(arr, i, 0);
    }
}
```

## 2.4 O(n) 정렬 알고리즘

### 계수 정렬 (Counting Sort)
원소의 값 범위가 제한적일 때 사용하는 정렬

#### Java 구현
```java
public class CountingSort {
    public static void countingSort(int[] arr) {
        if (arr.length == 0) return;
        
        // 최댓값과 최솟값 찾기
        int max = arr[0], min = arr[0];
        for (int num : arr) {
            max = Math.max(max, num);
            min = Math.min(min, num);
        }
        
        // 계수 배열 생성
        int range = max - min + 1;
        int[] count = new int[range];
        
        // 각 원소의 개수 세기
        for (int num : arr) {
            count[num - min]++;
        }
        
        // 누적합 계산
        for (int i = 1; i < range; i++) {
            count[i] += count[i - 1];
        }
        
        // 결과 배열 생성
        int[] output = new int[arr.length];
        for (int i = arr.length - 1; i >= 0; i--) {
            output[count[arr[i] - min] - 1] = arr[i];
            count[arr[i] - min]--;
        }
        
        // 원본 배열에 복사
        System.arraycopy(output, 0, arr, 0, arr.length);
    }
}
```

#### C 구현
```c
void countingSort(int arr[], int n) {
    if (n == 0) return;
    
    // 최댓값과 최솟값 찾기
    int max = arr[0], min = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > max) max = arr[i];
        if (arr[i] < min) min = arr[i];
    }
    
    int range = max - min + 1;
    int* count = (int*)calloc(range, sizeof(int));
    int* output = (int*)malloc(n * sizeof(int));
    
    // 각 원소의 개수 세기
    for (int i = 0; i < n; i++) {
        count[arr[i] - min]++;
    }
    
    // 누적합 계산
    for (int i = 1; i < range; i++) {
        count[i] += count[i - 1];
    }
    
    // 결과 배열 생성
    for (int i = n - 1; i >= 0; i--) {
        output[count[arr[i] - min] - 1] = arr[i];
        count[arr[i] - min]--;
    }
    
    // 원본 배열에 복사
    for (int i = 0; i < n; i++) {
        arr[i] = output[i];
    }
    
    free(count);
    free(output);
}
```

### 기수 정렬 (Radix Sort)
자릿수별로 정렬하는 방법

#### Java 구현
```java
public class RadixSort {
    public static void radixSort(int[] arr) {
        if (arr.length == 0) return;
        
        // 최댓값 찾기
        int max = arr[0];
        for (int num : arr) {
            max = Math.max(max, num);
        }
        
        // 각 자릿수에 대해 계수 정렬 수행
        for (int exp = 1; max / exp > 0; exp *= 10) {
            countingSortByDigit(arr, exp);
        }
    }
    
    private static void countingSortByDigit(int[] arr, int exp) {
        int n = arr.length;
        int[] output = new int[n];
        int[] count = new int[10];  // 0-9
        
        // 각 자릿수의 개수 세기
        for (int i = 0; i < n; i++) {
            count[(arr[i] / exp) % 10]++;
        }
        
        // 누적합 계산
        for (int i = 1; i < 10; i++) {
            count[i] += count[i - 1];
        }
        
        // 결과 배열 생성
        for (int i = n - 1; i >= 0; i--) {
            int digit = (arr[i] / exp) % 10;
            output[count[digit] - 1] = arr[i];
            count[digit]--;
        }
        
        // 원본 배열에 복사
        System.arraycopy(output, 0, arr, 0, n);
    }
}
```

## 2.5 정렬 알고리즘 비교

### 시간 복잡도 비교
| 알고리즘 | 최선 | 평균 | 최악 | 공간 | 안정성 |
|----------|------|------|------|------|--------|
| 버블 정렬 | O(n) | O(n²) | O(n²) | O(1) | 안정 |
| 선택 정렬 | O(n²) | O(n²) | O(n²) | O(1) | 불안정 |
| 삽입 정렬 | O(n) | O(n²) | O(n²) | O(1) | 안정 |
| 병합 정렬 | O(n log n) | O(n log n) | O(n log n) | O(n) | 안정 |
| 퀵 정렬 | O(n log n) | O(n log n) | O(n²) | O(log n) | 불안정 |
| 힙 정렬 | O(n log n) | O(n log n) | O(n log n) | O(1) | 불안정 |
| 계수 정렬 | O(n+k) | O(n+k) | O(n+k) | O(k) | 안정 |
| 기수 정렬 | O(nk) | O(nk) | O(nk) | O(n+k) | 안정 |

### 정렬 알고리즘 선택 가이드
1. **작은 데이터**: 삽입 정렬 (간단하고 효율적)
2. **큰 데이터**: 퀵 정렬, 병합 정렬, 힙 정렬
3. **거의 정렬된 데이터**: 삽입 정렬
4. **안정성이 필요한 경우**: 병합 정렬
5. **메모리 제약이 있는 경우**: 힙 정렬
6. **정수 데이터**: 계수 정렬, 기수 정렬

## 2.6 실습 예제

### 정렬 알고리즘 성능 비교
```java
public class SortingComparison {
    public static void main(String[] args) {
        int[] sizes = {100, 1000, 10000};
        
        for (int size : sizes) {
            System.out.println("\n배열 크기: " + size);
            
            // 랜덤 배열 생성
            int[] arr = generateRandomArray(size);
            
            // 각 정렬 알고리즘 테스트
            testSort(arr.clone(), "버블 정렬", BubbleSort::bubbleSort);
            testSort(arr.clone(), "선택 정렬", SelectionSort::selectionSort);
            testSort(arr.clone(), "삽입 정렬", InsertionSort::insertionSort);
            testSort(arr.clone(), "병합 정렬", MergeSort::mergeSort);
            testSort(arr.clone(), "퀵 정렬", QuickSort::quickSort);
            testSort(arr.clone(), "힙 정렬", HeapSort::heapSort);
        }
    }
    
    private static int[] generateRandomArray(int size) {
        int[] arr = new int[size];
        for (int i = 0; i < size; i++) {
            arr[i] = (int)(Math.random() * 1000);
        }
        return arr;
    }
    
    private static void testSort(int[] arr, String name, Consumer<int[]> sortMethod) {
        long start = System.nanoTime();
        sortMethod.accept(arr);
        long end = System.nanoTime();
        
        System.out.printf("%s: %.3f ms%n", name, (end - start) / 1_000_000.0);
        
        // 정렬 확인
        if (!isSorted(arr)) {
            System.out.println("정렬 실패!");
        }
    }
    
    private static boolean isSorted(int[] arr) {
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] < arr[i-1]) return false;
        }
        return true;
    }
}
```

## 2.7 요약

- 정렬은 데이터를 순서대로 재배열하는 기본적인 연산
- O(n²) 정렬은 구현이 간단하지만 큰 데이터에는 비효율적
- O(n log n) 정렬은 대부분의 경우에 효율적
- O(n) 정렬은 특정 조건에서만 사용 가능
- 데이터의 특성과 요구사항에 따라 적절한 정렬 알고리즘 선택이 중요