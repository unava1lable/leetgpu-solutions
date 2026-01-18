# Matrix Multiplication

Write a program that multiplies two matrices of 32-bit floating point numbers on a GPU. Given matrix A of dimensions `M x N` and matrix B of dimensions `N x K`, compute the product matrix C = A * B, which will have dimensions `M x K`. All matrices are stored in row-major format.

## Implementation Requirements
* Use only native features (external libraries are not permitted)
* The `solve` function signature must remain unchanged
* The final result must be stored in vector `C`

## Example 1:
```
Input:  A[2x2] = [1.0, 2.0, 3.0, 4.0]
        B[2x2] = [5.0, 6.0, 7.0, 8.0]
Output: C[2x2] = [19.0, 22.0, 43.0, 50.0]
```

## Example 2:
```
Input:  A[1x3] = [1.0, 2.0, 3.0]
        B[3x1] = [4.0, 5.0, 6.0]
Output: C[1x1] = [32.0]
```

## Constraints
* 1 ≤ `M`, `N`, `K` ≤ 8192
* Performance is measured with `M = 8192`, `N = 6144`, `K = 4096`