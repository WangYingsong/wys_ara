#include <stdint.h>
#include <string.h>

#ifndef SPIKE
#include "printf.h"
#else
#include "util.h"
#include <stdio.h>
#endif

#define MIN(a, b) ((a) < (b) ? (a) : (b))

void imatmul(int64_t *c, const int64_t *a, const int64_t *b,
             const unsigned long int m, const unsigned long int n,
             const unsigned long int p);

void imatmul_4x4(int64_t *c, const int64_t *a, const int64_t *b,
                 const unsigned long int m, const unsigned long int n,
                 const unsigned long int p);
void imatmul_vec_4x4_slice_init();
void imatmul_vec_4x4(int64_t *c, const int64_t *a, const int64_t *b,
                     const unsigned long int n, const unsigned long int p);
void imatmul_8x8(int64_t *c, const int64_t *a, const int64_t *b,
                 const unsigned long int m, const unsigned long int n,
                 const unsigned long int p);
void imatmul_vec_8x8_slice_init();
void imatmul_vec_8x8(int64_t *c, const int64_t *a, const int64_t *b,
                     const unsigned long int n, const unsigned long int p);

void imatmul(int64_t *c, const int64_t *a, const int64_t *b,
             const unsigned long int M, const unsigned long int N,
             const unsigned long int P) {
  if (M <= 4) {
    imatmul_4x4(c, a, b, M, N, P);
  }else if (M <= 128) {
    imatmul_8x8(c, a, b, M, N, P);
  }
  else {
    // Vector length is 64 elements. With an 4x4 matmul,
    // we can use LMUL=4, having a vl of 256.
    imatmul_4x4(c, a, b, M, N, P);
  }
}

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

    a = a_ + ++n;

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
}
void imatmul_8x8(int64_t *c, const int64_t *a, const int64_t *b,
                 const unsigned long int M, const unsigned long int N,
                 const unsigned long int P) {
  // We work on 4 rows of the matrix at once
  const unsigned long int block_size = 8;
  unsigned long int block_size_p;

  // Set the vector configuration
  asm volatile("vsetvli %0, %1, e64, m2, ta, ma" : "=r"(block_size_p) : "r"(P));

  // Slice the matrix into a manageable number of columns p_
  for (unsigned long int p = 0; p < P; p += block_size_p) {
    // Set the vector length
    const unsigned long int p_ = MIN(P - p, block_size_p);

    // Find pointers to the submatrices
    const int64_t *b_ = b + p;
    int64_t *c_ = c + p;

    asm volatile("vsetvli zero, %0, e64, m2, ta, ma" ::"r"(p_));

    // Iterate over the rows
    for (unsigned long int m = 0; m < M; m += block_size) {
      // Find pointer to the submatrices
      const int64_t *a_ = a + m * N;
      int64_t *c__ = c_ + m * P;

      imatmul_vec_8x8_slice_init();
      imatmul_vec_8x8(c__, a_, b_, N, P);
    }
  }
}

void imatmul_vec_8x8_slice_init() {
  asm volatile("vmv.v.i v0,  0");
  asm volatile("vmv.v.i v2,  0");
  asm volatile("vmv.v.i v4,  0");
  asm volatile("vmv.v.i v6,  0");
  asm volatile("vmv.v.i v8,  0");
  asm volatile("vmv.v.i v10, 0");
  asm volatile("vmv.v.i v12, 0");
  asm volatile("vmv.v.i v14, 0");
}

void imatmul_vec_8x8(int64_t *c, const int64_t *a, const int64_t *b,
                     const unsigned long int N, const unsigned long int P) {
  // Temporary variables
  int64_t t0, t1, t2, t3, t4, t5, t6, t7;

  // Original pointer
  const int64_t *a_ = a;

  // Prefetch one row of matrix B
  asm volatile("vle64.v v18, (%0);" ::"r"(b));
  b += P;

  // Prefetch one row of scalar values
  t0 = *a, a += N;
  t1 = *a, a += N;
  t2 = *a, a += N;
  t3 = *a, a += N;
  t4 = *a, a += N;
  t5 = *a, a += N;
  t6 = *a, a += N;
  t7 = *a;

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

    asm volatile("vmacc.vx v0, %0, v18" ::"r"(t0));
    t0 = *a, a += N;

    // Load one row of B
    asm volatile("vle64.v v20, (%0);" ::"r"(b));
    b += P;

    asm volatile("vmacc.vx v2, %0, v18" ::"r"(t1));
    t1 = *a, a += N;
    asm volatile("vmacc.vx v4, %0, v18" ::"r"(t2));
    t2 = *a, a += N;
    asm volatile("vmacc.vx v6, %0, v18" ::"r"(t3));
    t3 = *a, a += N;
    asm volatile("vmacc.vx v8, %0, v18" ::"r"(t4));
    t4 = *a, a += N;
    asm volatile("vmacc.vx v10, %0, v18" ::"r"(t5));
    t5 = *a, a += N;
    asm volatile("vmacc.vx v12, %0, v18" ::"r"(t6));
    t6 = *a, a += N;
    asm volatile("vmacc.vx v14, %0, v18" ::"r"(t7));
    t7 = *a;

    a = a_ + ++n;

    if (n == N)
      break;

    asm volatile("vmacc.vx v0, %0, v20" ::"r"(t0));
    t0 = *a, a += N;

    // Load one row of B
    asm volatile("vle64.v v18, (%0);" ::"r"(b));
    b += P;

    asm volatile("vmacc.vx v2, %0, v20" ::"r"(t1));
    t1 = *a, a += N;
    asm volatile("vmacc.vx v4, %0, v20" ::"r"(t2));
    t2 = *a, a += N;
    asm volatile("vmacc.vx v6, %0, v20" ::"r"(t3));
    t3 = *a, a += N;
    asm volatile("vmacc.vx v8, %0, v20" ::"r"(t4));
    t4 = *a, a += N;
    asm volatile("vmacc.vx v10, %0, v20" ::"r"(t5));
    t5 = *a, a += N;
    asm volatile("vmacc.vx v12, %0, v20" ::"r"(t6));
    t6 = *a, a += N;
    asm volatile("vmacc.vx v14, %0, v20" ::"r"(t7));
    t7 = *a;
  }

  // Last iteration: store results
  asm volatile("vmacc.vx v0, %0, v20" ::"r"(t0));
  asm volatile("vse64.v v0, (%0);" ::"r"(c));
  c += P;
  asm volatile("vmacc.vx v2, %0, v20" ::"r"(t1));
  asm volatile("vse64.v v2, (%0);" ::"r"(c));
  c += P;
  asm volatile("vmacc.vx v4, %0, v20" ::"r"(t2));
  asm volatile("vse64.v v4, (%0);" ::"r"(c));
  c += P;
  asm volatile("vmacc.vx v6, %0, v20" ::"r"(t3));
  asm volatile("vse64.v v6, (%0);" ::"r"(c));
  c += P;
  asm volatile("vmacc.vx v8, %0, v20" ::"r"(t4));
  asm volatile("vse64.v v8, (%0);" ::"r"(c));
  c += P;
  asm volatile("vmacc.vx v10, %0, v20" ::"r"(t5));
  asm volatile("vse64.v v10, (%0);" ::"r"(c));
  c += P;
  asm volatile("vmacc.vx v12, %0, v20" ::"r"(t6));
  asm volatile("vse64.v v12, (%0);" ::"r"(c));
  c += P;
  asm volatile("vmacc.vx v14, %0, v20" ::"r"(t7));
  asm volatile("vse64.v v14, (%0);" ::"r"(c));
}

int main() {
  unsigned long int dest;
  asm volatile("vsetvli %0, %1, e64, m4, ta, ma" :  "=r"(dest): "r"(4));
    int64_t t0=0,t1=4,t2=8,t3=12;
    int64_t v16[4]={0,1,2,3},v0[4]={0,0,0,0}, v4[4]={0,0,0,0},v8[4]={0,0,0,0},v12[4]={0,0,0,0}; 
    int64_t *v16_=v16,*v0_=v0, *v4_=v4, *v8_=v8, *v12_=v12;
    asm volatile("vle64.v v16, (%0);" ::"r"(v16_));

    // asm volatile("vmacc.vx v0, %0, v16" ::"r"(t0));
    asm volatile("vmacc.vx v4, %0, v16" ::"r"(t1));
    // asm volatile("vmacc.vx v8, %0, v16" ::"r"(t2));
    // asm volatile("vmacc.vx v12, %0, v16" ::"r"(t3));
    // asm volatile("vse64.v v0, (%0);" ::"r"(v0_));
    asm volatile("vse64.v v4, (%0);" ::"r"(v4_));
    // asm volatile("vse64.v v8, (%0);" ::"r"(v8_));
    // asm volatile("vse64.v v12, (%0);" ::"r"(v12_));
    // asm volatile("vse64.v v16, (%0);" ::"r"(v0_));
    // printf("v0={%ld, %ld, %ld, %ld}\n", v0[0], v0[1], v0[2], v0[3]);
    printf("v4={%ld, %ld, %ld, %ld}\n", v4[0], v4[1], v4[2], v4[3]);
    // printf("v8={%ld, %ld, %ld, %ld}\n", v8[0], v8[1], v8[2], v8[3]);
    // printf("v12={%ld, %ld, %ld, %ld}\n", v12[0], v12[1], v12[2], v12[3]);
  return 0;
}

// int main()
// {
//   int64_t a[256] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
//                    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
//                    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
//                    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
//                    64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
//                    80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
//                    96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
//                    112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
//                    128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
//                    144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
//                    160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
//                    176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
//                    192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
//                    208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
//                    224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
//                    240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255};
//   int64_t b[256] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
//                    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
//                    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
//                    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
//                    64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
//                    80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
//                    96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
//                    112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
//                    128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
//                    144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
//                    160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
//                    176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
//                    192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
//                    208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
//                    224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
//                    240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255};
//   int64_t c[256] = {0};
//   imatmul(c, a, b, 16, 16, 16);
//   printf("a:\n");
//   for(int i=0;i<16;i++)
//   {
//     for(int j=0;j<16;j++)
//     {
//       printf("%ld ", a[i*16+j]);
//     }
//     printf("\n");
//   }
//   printf("b:\n");
//   for(int i=0;i<16;i++)
//   {
//     for(int j=0;j<16;j++)
//     {
//       printf("%ld ", b[i*16+j]);
//     }
//     printf("\n");
//   }
//   printf("c:\n");
//   for(int i=0;i<16;i++)
//   {
//     for(int j=0;j<16;j++)
//     {
//       printf("%ld ", c[i*16+j]);
//     }
//     printf("\n");
//   }
//   int64_t axb[256];
//       for(int i=0;i<16;i++)
//     {
//       for(int j=0;j<16;j++)
//       {
//         for(int k=0;k<16;k++)
//         {
//           axb[i*16+j] += a[i*16+k] * b[k*16+j];
//         }
//       }
//     }
//     printf("axb:\n");
//     for(int i=0;i<16;i++)
//     {
//       for(int j=0;j<16;j++)
//       {
//         printf("%ld\t",axb[i*16+j]);
//       }
//       printf("\n");
//     }
//     printf("\n");
// }