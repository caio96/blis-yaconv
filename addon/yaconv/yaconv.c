/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2019, Advanced Micro Devices, Inc.
   Copyright (C) 2021 - 2022, Ivan Korostelev

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"

// Align x to the next multiple of align
static inline int align_to(int x, int align) {
  return (x + align - 1) & ~(align - 1);
}

// Convenience page-aligned allocation with return check
static float *aligned_alloc(int size) {
  float *data = NULL;
  int ret =
      posix_memalign((void **)&data, BLIS_PAGE_SIZE, size * sizeof(float));

  switch (ret) {
  case 0:
    return data;
  case EINVAL:
    fprintf(stderr, "\033[31mBad alignment in posix_memalign!\033[0m\n");
  case ENOMEM:
    fprintf(stderr, "\033[31mNo memory in posix_memalign!\033[0m\n");
  default:
    exit(ret);
  }
}

// Offset required after the output image
static int yaconv_extra_size_after(int H, int FH, int PH, int OW, int M,
                                   const cntx_t *cntx) {
  int NR = bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_NR, cntx);
  int extra_h = H % NR ? NR - H % NR : 0;
  return bli_max(0, extra_h + FH - 1 - PH) * OW * M;
}

// Offset required before the output image
static int yaconv_extra_size_before(int FH, int PH, int OW, int M) {
  return bli_max(0, FH - 1 - PH) * OW * M;
}

// Packing function
static void yaconv_pack(float *src, int rss, int css, float *dst, int MN, int k,
                        int MNR, const cntx_t *cntx) {
  for (int mn = 0; mn < MN; mn += MNR)
    bls_spackm_cxk(BLIS_NO_CONJUGATE, BLIS_PACKED_ROW_PANELS,
                   bli_min(MN - mn, MNR), MNR, k, k,
                   bli_s1, src + mn * rss, rss, css,
                   dst + mn * k, MNR,
                   (cntx_t*)cntx);
}

// The main yaconv function that computes convolution on a single image
static void yaconv_single_image(float *image, int H, int W, int C,
                                float *filter, int FH, int FW, int M,
                                float *output, int PH, int PW, int MC, int NC,
                                int KC, int MR, int NR, float *image_buf,
                                float *filter_buf, float *output_buf,
                                auxinfo_t *auxinfo, const cntx_t *cntx) {

  // First, compute the spatial width of the output
  const int OH = H + 2 * PH - FH + 1;
  const int OW = W + 2 * PW - FW + 1;

  output += yaconv_extra_size_before(FH, PH, OW, M);
  bli_ssetv(BLIS_NO_CONJUGATE, OH * OW * M, bli_s0, output, 1);

  for (int nc = 0; nc < H; nc += NC) {

    int nc_curr = bli_min(H - nc, NC);

    yaconv_pack(image + nc * W * C, W * C, 1, image_buf, nc_curr, W * C, NR,
                cntx);

    for (int fh = 0; fh < FH; ++fh) {
      for (int m = 0; m < M; m += MC) {

        int mc_curr = bli_min(M - m, MC);

        for (int kc = 0; kc < FW * C; kc += KC) {

          int kc_curr = bli_min(FW * C - kc, KC);
          yaconv_pack(filter + (fh * FW * C + kc) * M + m, 1, M, filter_buf,
                      mc_curr, kc_curr, MR, cntx);

          for (int nr = 0; nr < nc_curr; nr += NR) {
            for (int ow = 0; ow < OW; ++ow) {

              int image_start = (ow - PW) * C + kc;
              int image_end = bli_min(W * C, image_start + kc_curr);

              float *ar = filter_buf;
              if (image_start < 0) {
                ar -= image_start * MR;
                image_start = 0;
              }

              int K = image_end - image_start;
              if (K <= 0)
                continue;

              float *br = image_buf + nr * W * C + image_start * NR;
              float *cr = output + ((nc + nr - fh + PH) * OW + ow) * M + m;

              for (int mr = 0; mr < mc_curr; mr += MR) {
                if (mr + MR <= mc_curr)
                  bli_sgemm_ukernel(MR, NR, K, bli_s1, ar, br, bli_s1, cr, 1,
                                    OW * M, auxinfo, cntx);
                else {
                  bli_sgemm_ukernel(MR, NR, K, bli_s1, ar, br, bli_s0,
                                    output_buf, NR, 1, auxinfo, cntx);
                  bli_sxpbys_mxn(mc_curr - mr, NR, output_buf, NR, 1, bli_s1,
                                 cr, 1, OW * M);
                }

                ar += MR * kc_curr;
                cr += MR;
              }
            }
          }
        }
      }
    }
  }
}

void yaconv_ex(float *images, int N, int H, int W, int C, float *filter, int FH,
               int FW, int M, float *outputs, int PH, int PW, cntx_t *cntx) {
  // Get valid context
  if (cntx == NULL)
    cntx = (cntx_t *)bli_gks_query_cntx();

  // Allocate memory for auxiliary buffers
  auxinfo_t *auxinfo = (auxinfo_t *)malloc(sizeof(auxinfo_t));

  // Get block sizes
  int MR = bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_MR, cntx);
  int NR = bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_NR, cntx);
  int MC = bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_MC, cntx);
  int KC = bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_KC, cntx);
  int NC = bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_NC, cntx);

  // Adjust NC at run-time so that the packed image buffer fits in L3 and
  // NC is the nearest multiple of NR greater than NC in BLIS
  NC = KC * NC / W / C;
  NC += (NC % NR) ? NR - NC % NR : 0;
  KC = bli_min(FW * C, KC); // to use less buffer space for small inputs

  // Output dimensions
  int OH = H + 2 * PH - FH + 1;
  int OW = W + 2 * PW - FW + 1;

  // Extra offset required by yaconv
  int extra_before = yaconv_extra_size_before(FH, PH, OW, M);
  int extra_after = yaconv_extra_size_after(H, FH, PH, OW, M, cntx);

  // Allocate memory for filter, image, and output buffers
  float *filter_buf = aligned_alloc(MC * KC);
  float *image_buf = aligned_alloc(W * C * NC);
  float *output_buf = aligned_alloc(MR * NR);

  // Allocate memory for single output
  float *single_output =
      aligned_alloc(OH * OW * M + extra_before + extra_after);

  // Run yaconv on each image
  for (int i = 0; i < N; ++i) {
    yaconv_single_image(&images[i * H * W * C], H, W, C, filter, FH, FW, M,
                        single_output, PH, PW, MC, NC, KC, MR, NR, image_buf,
                        filter_buf, output_buf, auxinfo, cntx);
    // Convert single output to NHWC
    for (int j = 0; j < OH * OW * M; ++j) {
      outputs[i * OH * OW * M + j] = single_output[j + extra_before];
    }
  }

  // Deallocate buffers
  free(single_output);
  free(filter_buf);
  free(image_buf);
  free(output_buf);
  free(auxinfo);
}

void yaconv(float *images, int N, int H, int W, int C, float *filter, int FH,
            int FW, int M, float *outputs, int PH, int PW) {
  yaconv_ex(images, N, H, W, C, filter, FH, FW, M, outputs, PH, PW, NULL);
}
