#include <stdint.h>
#include <string.h>

#ifndef SPIKE
#include "printf.h"
#else
#include "util.h"
#include <stdio.h>
#endif

#define MIN(a, b) ((a) < (b) ? (a) : (b))

int verify_matrix(int64_t *result, int64_t *gold, size_t R, size_t C);
void imatmul(int64_t *c, const int64_t *a, const int64_t *b,
             const unsigned long int M, const unsigned long int N,
             const unsigned long int P);
void imatmul_4x4(int64_t *c, const int64_t *a, const int64_t *b,
                 const unsigned long int M, const unsigned long int N,
                 const unsigned long int P);
void imatmul_vec_4x4_slice_init();
void imatmul_vec_4x4(int64_t *c, const int64_t *a, const int64_t *b,
                     const unsigned long int N, const unsigned long int P);


// Verify the matrix
int verify_matrix(int64_t *result, int64_t *gold, size_t R, size_t C) {
  for (uint64_t i = 0; i < R; ++i) {
    for (uint64_t j = 0; j < C; ++j) {
      uint64_t idx = i * C + j;
      printf("result[%ld]=%ld, gold[%ld]=%ld\t", idx, result[idx], idx, gold[idx]);
      if (result[idx] != gold[idx]) {
        return (i + j) == 0 ? -1 : idx;
      }
    }
    printf("\n");
  }
  return 0;
}

void imatmul(int64_t *c, const int64_t *a, const int64_t *b,
             const unsigned long int M, const unsigned long int N,
             const unsigned long int P) {
  if (M <= 4) {
    imatmul_4x4(c, a, b, M, N, P);
  }
  else {
    // Vector length is 64 elements. With an 4x4 matmul,
    // we can use LMUL=4, having a vl of 256.
    imatmul_4x4(c, a, b, M, N, P);
  }
}

// ---------------
// 4x4
// ---------------

void imatmul_4x4(int64_t *c, const int64_t *a, const int64_t *b,
                 const unsigned long int M, const unsigned long int N,
                 const unsigned long int P) {
  // We work on 4 rows of the matrix at once
  const unsigned long int block_size = 4;
  unsigned long int block_size_p;

  // Set the vector configuration
  asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(block_size_p) : "r"(P));

  // Slice the matrix into a manageable number of columns p_
  for (unsigned long int p = 0; p < P; p += block_size_p) {
    // Set the vector length
    const unsigned long int p_ = MIN(P - p, block_size_p);

    // Find pointers to the submatrices
    const int64_t *b_ = b + p;
    int64_t *c_ = c + p;

    asm volatile("vsetvli zero, %0, e64, m4, ta, ma" ::"r"(p_));

    // Iterate over the rows
    for (unsigned long int m = 0; m < M; m += block_size) {
      // Find pointer to the submatrices
      const int64_t *a_ = a + m * N;
      int64_t *c__ = c_ + m * P;

      imatmul_vec_4x4_slice_init();
      imatmul_vec_4x4(c__, a_, b_, N, P);
    }
  }
}

void imatmul_vec_4x4_slice_init() {
  asm volatile("vmv.v.i v0,  0");
  asm volatile("vmv.v.i v4,  0");
  asm volatile("vmv.v.i v8,  0");
  asm volatile("vmv.v.i v12, 0");
}

void imatmul_vec_4x4(int64_t *c, const int64_t *a, const int64_t *b,
                     const unsigned long int N, const unsigned long int P) {
  // Temporary variables
  int64_t t0, t1, t2, t3;
    int64_t v0_[4], v4_[4], v8_[4], v12_[4],v16_[4],v20_[4];
    int64_t *v0__ = v0_, *v4__ = v4_, *v8__ = v8_, *v12__ = v12_, *v16__ = v16_, *v20__ = v20_;
  // Original pointer
  const int64_t *a_ = a;

  // Prefetch one row of matrix B
  asm volatile("vle64.v v16, (%0);" ::"r"(b));
  b += P;

  // Prefetch one row of scalar values
  t0 = *a, a += N;
  t1 = *a, a += N;
  t2 = *a, a += N;
  t3 = *a;

  // Compute the multiplication
  unsigned long int n = 0;

  while (n < N) {
#ifdef VCD_DUMP
    // Start dumping VCD
    if (n == 8)
      event_trigger = +1;
    // Stop dumping VCD
    if (n == 12)
      event_trigger = -1;
#endif

    // Calculate pointer to the matrix A
    a = a_ + ++n;

    printf("t0=%ld, t1=%ld, t2=%ld, t3=%ld\n", t0, t1, t2, t3);

    asm volatile("vse64.v v16, (%0);" ::"r"(v16__));
    printf("v16={%ld, %ld, %ld, %ld}\n", v16_[0], v16_[1], v16_[2], v16_[3]);

    asm volatile("vmacc.vx v0, %0, v16" ::"r"(t0));
    t0 = *a, a += N;

    // Load one row of B
    asm volatile("vle64.v v20, (%0);" ::"r"(b));
    b += P;

    asm volatile("vmacc.vx v4, %0, v16" ::"r"(t1));
    t1 = *a, a += N;
    asm volatile("vmacc.vx v8, %0, v16" ::"r"(t2));
    t2 = *a, a += N;
    asm volatile("vmacc.vx v12, %0, v16" ::"r"(t3));
    t3 = *a;


    asm volatile("vse64.v v0, (%0);" ::"r"(v0__));
    asm volatile("vse64.v v4, (%0);" ::"r"(v4__));
    asm volatile("vse64.v v8, (%0);" ::"r"(v8__));
    asm volatile("vse64.v v12, (%0);" ::"r"(v12__));
    printf("v0={%ld, %ld, %ld, %ld}, v4={%ld, %ld, %ld, %ld}, v8={%ld, %ld, %ld, %ld}, v12={%ld, %ld, %ld, %ld}\n", v0_[0], v0_[1], v0_[2], v0_[3], v4_[0], v4_[1], v4_[2], v4_[3], v8_[0], v8_[1], v8_[2], v8_[3], v12_[0], v12_[1], v12_[2], v12_[3]);

    a = a_ + ++n;

    printf("t0=%ld, t1=%ld, t2=%ld, t3=%ld\n", t0, t1, t2, t3);

    asm volatile("vse64.v v20, (%0);" ::"r"(v20__));
    printf("v20={%ld, %ld, %ld, %ld}\n", v20_[0], v20_[1], v20_[2], v20_[3]);
    if (n == N)
      break;

    asm volatile("vmacc.vx v0, %0, v20" ::"r"(t0));
    t0 = *a, a += N;

    // Load one row of B
    asm volatile("vle64.v v16, (%0);" ::"r"(b));
    b += P;

    asm volatile("vmacc.vx v4, %0, v20" ::"r"(t1));
    t1 = *a, a += N;
    asm volatile("vmacc.vx v8, %0, v20" ::"r"(t2));
    t2 = *a, a += N;
    asm volatile("vmacc.vx v12, %0, v20" ::"r"(t3));
    t3 = *a;


    asm volatile("vse64.v v0, (%0);" ::"r"(v0__));
    asm volatile("vse64.v v4, (%0);" ::"r"(v4__));
    asm volatile("vse64.v v8, (%0);" ::"r"(v8__));
    asm volatile("vse64.v v12, (%0);" ::"r"(v12__));
    printf("v0={%ld, %ld, %ld, %ld}, v4={%ld, %ld, %ld, %ld}, v8={%ld, %ld, %ld, %ld}, v12={%ld, %ld, %ld, %ld}\n", v0_[0], v0_[1], v0_[2], v0_[3], v4_[0], v4_[1], v4_[2], v4_[3], v8_[0], v8_[1], v8_[2], v8_[3], v12_[0], v12_[1], v12_[2], v12_[3]);

  }

  // Last iteration: store results
  asm volatile("vmacc.vx v0, %0, v20" ::"r"(t0));
  asm volatile("vse64.v v0, (%0);" ::"r"(c));
  c += P;
  asm volatile("vmacc.vx v4, %0, v20" ::"r"(t1));
  asm volatile("vse64.v v4, (%0);" ::"r"(c));
  c += P;
  asm volatile("vmacc.vx v8, %0, v20" ::"r"(t2));
  asm volatile("vse64.v v8, (%0);" ::"r"(c));
  c += P;
  asm volatile("vmacc.vx v12, %0, v20" ::"r"(t3));
  asm volatile("vse64.v v12, (%0);" ::"r"(c));

  asm volatile("vse64.v v0, (%0);" ::"r"(v0__));
    asm volatile("vse64.v v4, (%0);" ::"r"(v4__));
    asm volatile("vse64.v v8, (%0);" ::"r"(v8__));
    asm volatile("vse64.v v12, (%0);" ::"r"(v12__));
    printf("v0={%ld, %ld, %ld, %ld}, v4={%ld, %ld, %ld, %ld}, v8={%ld, %ld, %ld, %ld}, v12={%ld, %ld, %ld, %ld}\n", v0_[0], v0_[1], v0_[2], v0_[3], v4_[0], v4_[1], v4_[2], v4_[3], v8_[0], v8_[1], v8_[2], v8_[3], v12_[0], v12_[1], v12_[2], v12_[3]);

}

// int main() {
//   unsigned long int dest;
//   asm volatile("vsetvli %0, %1, e64, m4, ta, ma" :  "=r"(dest): "r"(4));
//   int64_t a[4] = {1, 2, 3, 4};
//   int64_t b[4] = {1, 2, 3, 4};
//   int64_t c[4] = {1, 1, 1, 1};
//   int64_t *a_ = a;
//   int64_t *b_ = b;
//   int64_t *c_ = c;
//   printf("*a_=%ld, *b_=%ld, *c_=%ld\n", *a_, *b_, *c_);
//   asm volatile("vle64.v v4, (%0)" :: "r"(a_));
//   asm volatile("vle64.v v8, (%0)" :: "r"(b_));
//   asm volatile("vle64.v v12, (%0)" :: "r"(c_));
//   asm volatile("vmacc.vx v12, %0, v4" :: "r"(3333));
//   // asm volatile("vmacc.vv v12, v4, v8");
//   asm volatile("vse64.v v12, (%0)" :: "r"(c_));
//   printf("c[4]={%ld, %ld, %ld, %ld}\n", c[0], c[1], c[2], c[3]);
//   return 0;
// }

int main()
{
  printf("\n");
  printf("=============\n");
  printf("=  IMATMUL  =\n");
  printf("=============\n");
  printf("\n");
  printf("\n");

  for (int s = 4; s <= 4; s *= 2) {
    printf("\n");
    printf("------------------------------------------------------------\n");
    printf("Calculating a (%d x %d) x (%d x %d) matrix multiplication...\n", s,
           s, s, s);
    printf("------------------------------------------------------------\n");
    printf("\n");

    // Initialize matrices
    printf("Initializing matrices...\n");
    int64_t a[16]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    int64_t b[16]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    int64_t c[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    int64_t g[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    // Initialize gold matrix
    for (int i = 0; i < s; i++) {
      for (int j = 0; j < s; j++) {
        for (int k = 0; k < s; k++) {
          g[i * s + j] += a[i * s + k] * b[k * s + j];
        }
      }
    }

    // Perform the matrix multiplication
    imatmul(c, a, b, s, s, s);

    printf("c[][]:\n");
    for(int i=0;i<s;i++)
    {
      for(int j=0;j<s;j++)
      {
        printf("%ld\t",c[i*s+j]);
      }
      printf("\n");
    }

    printf("g[][]:\n");
    for(int i=0;i<s;i++)
    {
      for(int j=0;j<s;j++)
      {
        printf("%ld\t",g[i*s+j]);
      }
      printf("\n");
    }
    // Verify the result
    if (verify_matrix(c, g, s, s) == 0) {
      printf("Matrix multiplication successful!\n");
    } else {
      printf("Matrix multiplication failed!\n");
    }
  }

  return 0;
}