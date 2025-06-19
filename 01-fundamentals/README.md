# Chapter 1: 알고리즘 기초 (Algorithm Fundamentals)

## 1.1 알고리즘이란?

### 정의
알고리즘(Algorithm)은 특정 문제를 해결하기 위한 명확하고 유한한 단계들의 집합입니다. 입력을 받아 원하는 출력을 생성하는 계산 절차입니다.

### 알고리즘의 특성
1. **입력(Input)**: 0개 이상의 입력
2. **출력(Output)**: 1개 이상의 출력
3. **명확성(Definiteness)**: 각 단계는 명확하고 모호하지 않음
4. **유한성(Finiteness)**: 유한한 단계 후 종료
5. **효율성(Effectiveness)**: 각 단계는 기본적이고 실행 가능

## 1.2 복잡도 분석

### 시간 복잡도 (Time Complexity)
알고리즘이 문제를 해결하는 데 걸리는 시간을 입력 크기에 대한 함수로 표현합니다.

### 공간 복잡도 (Space Complexity)
알고리즘이 실행되는 동안 필요한 메모리 공간을 나타냅니다.

### Big-O 표기법
최악의 경우(Worst Case)의 성능을 나타내는 표기법입니다.

| 표기법 | 명칭 | 예시 | 성능 |
|--------|------|------|------|
| O(1) | 상수 시간 | 배열 접근 | 최고 |
| O(log n) | 로그 시간 | 이진 탐색 | 우수 |
| O(n) | 선형 시간 | 선형 탐색 | 양호 |
| O(n log n) | 선형 로그 시간 | 효율적인 정렬 | 양호 |
| O(n²) | 이차 시간 | 중첩 반복문 | 나쁨 |
| O(2ⁿ) | 지수 시간 | 재귀적 피보나치 | 최악 |

### 예제: 복잡도 분석

#### Java 구현
```java
public class ComplexityExamples {
    
    // O(1) - 상수 시간
    public static int constantTime(int[] arr) {
        if (arr.length == 0) return -1;
        return arr[0];  // 첫 번째 요소 반환
    }
    
    // O(log n) - 로그 시간
    public static int binarySearch(int[] arr, int target) {
        int left = 0;
        int right = arr.length - 1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            
            if (arr[mid] == target) {
                return mid;
            } else if (arr[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return -1;
    }
    
    // O(n) - 선형 시간
    public static int linearSearch(int[] arr, int target) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == target) {
                return i;
            }
        }
        return -1;
    }
    
    // O(n²) - 이차 시간
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
    
    // O(2ⁿ) - 지수 시간 (비효율적)
    public static int fibonacci(int n) {
        if (n <= 1) {
            return n;
        }
        return fibonacci(n - 1) + fibonacci(n - 2);
    }
    
    // O(n) - 선형 시간 (효율적인 피보나치)
    public static int fibonacciEfficient(int n) {
        if (n <= 1) return n;
        
        int prev = 0, curr = 1;
        for (int i = 2; i <= n; i++) {
            int temp = curr;
            curr = prev + curr;
            prev = temp;
        }
        return curr;
    }
}
```

#### C 구현
```c
#include <stdio.h>

// O(1) - 상수 시간
int constantTime(int arr[], int size) {
    if (size == 0) return -1;
    return arr[0];  // 첫 번째 요소 반환
}

// O(log n) - 로그 시간
int binarySearch(int arr[], int size, int target) {
    int left = 0;
    int right = size - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return -1;
}

// O(n) - 선형 시간
int linearSearch(int arr[], int size, int target) {
    for (int i = 0; i < size; i++) {
        if (arr[i] == target) {
            return i;
        }
    }
    return -1;
}

// O(n²) - 이차 시간
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

// O(2ⁿ) - 지수 시간 (비효율적)
int fibonacci(int n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

// O(n) - 선형 시간 (효율적인 피보나치)
int fibonacciEfficient(int n) {
    if (n <= 1) return n;
    
    int prev = 0, curr = 1;
    for (int i = 2; i <= n; i++) {
        int temp = curr;
        curr = prev + curr;
        prev = temp;
    }
    return curr;
}
```

## 1.3 재귀와 반복

### 재귀 (Recursion)
함수가 자기 자신을 호출하는 프로그래밍 기법입니다.

#### 재귀의 조건
1. **기저 사례(Base Case)**: 재귀를 종료하는 조건
2. **재귀 사례(Recursive Case)**: 자기 자신을 호출하는 부분

### 재귀 vs 반복 비교

#### Java 예제
```java
public class RecursionVsIteration {
    
    // 팩토리얼 - 재귀
    public static int factorialRecursive(int n) {
        if (n <= 1) {  // 기저 사례
            return 1;
        }
        return n * factorialRecursive(n - 1);  // 재귀 사례
    }
    
    // 팩토리얼 - 반복
    public static int factorialIterative(int n) {
        int result = 1;
        for (int i = 2; i <= n; i++) {
            result *= i;
        }
        return result;
    }
    
    // 하노이의 탑 - 재귀
    public static void hanoi(int n, char from, char to, char aux) {
        if (n == 1) {
            System.out.println("Move disk 1 from " + from + " to " + to);
            return;
        }
        
        hanoi(n - 1, from, aux, to);
        System.out.println("Move disk " + n + " from " + from + " to " + to);
        hanoi(n - 1, aux, to, from);
    }
    
    // 이진 트리 순회 - 재귀
    static class TreeNode {
        int val;
        TreeNode left, right;
        
        TreeNode(int val) {
            this.val = val;
        }
    }
    
    public static void inorderTraversal(TreeNode root) {
        if (root == null) return;
        
        inorderTraversal(root.left);
        System.out.print(root.val + " ");
        inorderTraversal(root.right);
    }
}
```

#### C 예제
```c
#include <stdio.h>

// 팩토리얼 - 재귀
int factorialRecursive(int n) {
    if (n <= 1) {  // 기저 사례
        return 1;
    }
    return n * factorialRecursive(n - 1);  // 재귀 사례
}

// 팩토리얼 - 반복
int factorialIterative(int n) {
    int result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

// 하노이의 탑 - 재귀
void hanoi(int n, char from, char to, char aux) {
    if (n == 1) {
        printf("Move disk 1 from %c to %c\n", from, to);
        return;
    }
    
    hanoi(n - 1, from, aux, to);
    printf("Move disk %d from %c to %c\n", n, from, to);
    hanoi(n - 1, aux, to, from);
}

// 이진 트리 순회 - 재귀
typedef struct TreeNode {
    int val;
    struct TreeNode* left;
    struct TreeNode* right;
} TreeNode;

void inorderTraversal(TreeNode* root) {
    if (root == NULL) return;
    
    inorderTraversal(root->left);
    printf("%d ", root->val);
    inorderTraversal(root->right);
}
```

## 1.4 기본 자료구조 복습

### 배열 (Array)
연속된 메모리 공간에 동일한 타입의 데이터를 저장하는 자료구조

#### Java
```java
public class ArrayBasics {
    public static void main(String[] args) {
        // 배열 선언과 초기화
        int[] arr = new int[5];  // 크기 5인 정수 배열
        int[] arr2 = {1, 2, 3, 4, 5};  // 초기값과 함께 선언
        
        // 배열 접근 - O(1)
        int first = arr2[0];  // 1
        arr2[2] = 10;  // 3을 10으로 변경
        
        // 배열 순회 - O(n)
        for (int i = 0; i < arr2.length; i++) {
            System.out.println(arr2[i]);
        }
    }
}
```

#### C
```c
#include <stdio.h>

int main() {
    // 배열 선언과 초기화
    int arr[5];  // 크기 5인 정수 배열
    int arr2[] = {1, 2, 3, 4, 5};  // 초기값과 함께 선언
    
    // 배열 접근 - O(1)
    int first = arr2[0];  // 1
    arr2[2] = 10;  // 3을 10으로 변경
    
    // 배열 순회 - O(n)
    for (int i = 0; i < 5; i++) {
        printf("%d ", arr2[i]);
    }
    
    return 0;
}
```

### 연결 리스트 (Linked List)
노드들이 포인터로 연결된 선형 자료구조

#### Java
```java
public class LinkedListBasics {
    static class Node {
        int data;
        Node next;
        
        Node(int data) {
            this.data = data;
            this.next = null;
        }
    }
    
    static class LinkedList {
        Node head;
        
        // 맨 앞에 삽입 - O(1)
        void addFirst(int data) {
            Node newNode = new Node(data);
            newNode.next = head;
            head = newNode;
        }
        
        // 출력
        void printList() {
            Node current = head;
            while (current != null) {
                System.out.print(current.data + " -> ");
                current = current.next;
            }
            System.out.println("null");
        }
    }
}
```

#### C
```c
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int data;
    struct Node* next;
} Node;

// 새 노드 생성
Node* createNode(int data) {
    Node* newNode = (Node*)malloc(sizeof(Node));
    newNode->data = data;
    newNode->next = NULL;
    return newNode;
}

// 맨 앞에 삽입 - O(1)
void addFirst(Node** head, int data) {
    Node* newNode = createNode(data);
    newNode->next = *head;
    *head = newNode;
}

// 출력
void printList(Node* head) {
    Node* current = head;
    while (current != NULL) {
        printf("%d -> ", current->data);
        current = current->next;
    }
    printf("NULL\n");
}
```

## 1.5 최선, 평균, 최악의 경우

### 복잡도 분석의 세 가지 경우
1. **최선의 경우 (Best Case)**: Ω(빅 오메가) 표기
2. **평균의 경우 (Average Case)**: Θ(빅 세타) 표기
3. **최악의 경우 (Worst Case)**: O(빅 오) 표기

### 예제: 선형 탐색의 경우
- 최선: O(1) - 첫 번째 원소가 찾는 값
- 평균: O(n/2) = O(n) - 중간쯤에서 발견
- 최악: O(n) - 마지막 원소이거나 없는 경우

## 1.6 연습 문제

1. **두 수의 합**: 배열에서 합이 특정 값이 되는 두 수를 찾기
2. **배열 회전**: 배열을 k번 오른쪽으로 회전
3. **팰린드롬 확인**: 재귀를 사용하여 문자열이 팰린드롬인지 확인
4. **최대 부분 배열 합**: 연속된 부분 배열의 최대 합 구하기

## 1.7 요약

- 알고리즘은 문제를 해결하는 명확한 절차
- 복잡도 분석을 통해 알고리즘의 효율성 평가
- Big-O 표기법으로 최악의 경우 성능 표현
- 재귀와 반복은 각각 장단점이 있음
- 기본 자료구조의 특성과 연산 복잡도 이해가 중요