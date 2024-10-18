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

// Packing function based on the bli_spackm_cxk function used in the original
// yaconv implementation
static void yaconv_pack(conj_t conja, pack_t schema, dim_t panel_dim,
                        dim_t panel_dim_max, dim_t panel_len,
                        dim_t panel_len_max, float *kappa, float *a, inc_t inca,
                        inc_t lda, float *p, inc_t ldp, cntx_t *cntx) {
  num_t dt = PASTEMAC(s, type);
  ukr_t ker_id =
      bli_is_col_packed(schema) ? BLIS_PACKM_NRXK_KER : BLIS_PACKM_MRXK_KER;

  // Query the context for the packm kernel corresponding to the current panel
  // dimension, or kernel id. If the id is invalid, the function will return
  // NULL.
  packm_cxk_ker_ft f = bli_cntx_get_ukr_dt(dt, ker_id, cntx);

  if (f != NULL) {
    f(conja, schema, panel_dim, panel_len, panel_len_max, kappa, a, inca, lda,
      p, ldp, cntx);
  } else {
    // Treat the micro-panel as panel_dim x panel_len and column-stored (unit
    // row stride). The rntm_t* can safely be NULL as long as it's not used by
    // scal2m_ex().
    PASTEMAC2(s, scal2m, BLIS_TAPI_EX_SUF)
    (0, BLIS_NONUNIT_DIAG, BLIS_DENSE, (trans_t)conja, panel_dim, panel_len,
     kappa, a, inca, lda, p, 1, ldp, cntx, NULL);

    // If panel_dim < panel_dim_max, then we zero those unused rows.
    if (panel_dim < panel_dim_max) {
      const dim_t i = panel_dim;
      const dim_t m_edge = panel_dim_max - panel_dim;
      const dim_t n_edge = panel_len_max;
      float *restrict p_edge = p + (i) * 1;

      PASTEMAC(s, set0s_mxn)
      (m_edge, n_edge, p_edge, 1, ldp);
    }

    // If panel_len < panel_len_max, then we zero those unused columns.
    if (panel_len < panel_len_max) {
      const dim_t j = panel_len;
      const dim_t m_edge = panel_dim_max;
      const dim_t n_edge = panel_len_max - panel_len;
      float *restrict p_edge = p + (j)*ldp;

      PASTEMAC(s, set0s_mxn)
      (m_edge, n_edge, p_edge, 1, ldp);
    }
  }
}

// The main yaconv function that computes convolution on a single image
// Image is in NHWC format
// Filter is in HWCM format
static void yaconv_single_image(float *image, int H, int W, int C,
                                float *filter, int FH, int FW, int M,
                                float *output, int PH, int PW, int MC, int NC,
                                int KC, int MR, int NR, float *image_buf,
                                float *filter_buf, float *output_buf,
                                auxinfo_t *auxinfo, const cntx_t *cntx) {

  // First, compute the spatial width of the output
  const int OH = H + 2 * PH - FH + 1;
  const int OW = W + 2 * PW - FW + 1;

  // Skip extra offset at the start of the output
  output += yaconv_extra_size_before(FH, PH, OW, M);
  // Initialize output to zeros
  bli_ssetv(BLIS_NO_CONJUGATE, OH * OW * M, bli_s0, output, 1);

  // NC divides the image height into block
  for (int nc = 0; nc < H; nc += NC) {
    // Get block of size NC, or, if the last block is smaller, the remaining
    int nc_curr = bli_min(H - nc, NC);

    // Pack block of the image of size W*C x nc_curr
    // where nc_curr is divided into blocks of size NR.
    // This packing transposes the image (inca is not 1)
    for (int nr = 0; nr < nc_curr; nr += NR) {
      // bls_spackm_cxk also works here if gemm-like sandbox is enabled
      yaconv_pack(BLIS_NO_CONJUGATE, BLIS_PACKED_ROW_PANELS,
                  bli_min(nc_curr - nr, NR), NR, W * C, W * C, bli_s1,
                  &image[(nc + nr) * W * C], W * C, 1, &image_buf[nr * W * C],
                  NR, (cntx_t *)cntx);
    }

    // For every element in the filter height
    for (int fh = 0; fh < FH; ++fh) {

      // MC divides the number of output channels into blocks
      for (int mc = 0; mc < M; mc += MC) {
        // Get block of size MC, or, if the last block is smaller, the
        // remaining
        int mc_curr = bli_min(M - mc, MC);

        // TODO: check if the kc loop should be moved outside the mc loop
        //       because M is the innermost dimension of the filter
        //
        // KC divides the filter width times input channels into blocks
        for (int kc = 0; kc < FW * C; kc += KC) {
          // Get block of size KC, or, if the last block is smaller, the
          // remaining
          int kc_curr = bli_min(FW * C - kc, KC);

          // Pack block of the filter of size kc_curr x mc_curr
          // where mc_curr is divided into blocks of size MR.
          for (int mr = 0; mr < mc_curr; mr += MR) {
            // bls_spackm_cxk also works here if gemm-like sandbox is enabled
            yaconv_pack(BLIS_NO_CONJUGATE, BLIS_PACKED_ROW_PANELS,
                        bli_min(mc_curr - mr, MR), MR, kc_curr, kc_curr, bli_s1,
                        &filter[(fh * FW * C + kc) * M + mc + mr], 1, M,
                        &filter_buf[mr * kc_curr], MR, (cntx_t *)cntx);
          }

          // NR subdivides the block of size NC into smaller blocks
          for (int nr = 0; nr < nc_curr; nr += NR) {
            // Get corresponding output height. The filter height is subtracted
            // because instead of shifting the image one element in the H
            // dimension for the next filter height and shortening nr, the image
            // tile stays the same and the first elements of the output are
            // stores in the extra space before as trash. Filter height can make
            // this value negative, this is why the output has some extra space
            // allocated before.
            int oh = nc + nr - fh + PH;

            // For every output width element
            for (int ow = 0; ow < OW; ++ow) {

              // Start of the filter block of size kc_curr * mc_curr.
              // This tile was fixed before loop nr, but ow check if some
              // weights need to be skipped due to padding.
              float *ar = filter_buf;

              // Get a slice of the W*C dimension of the image of size
              // kc_curr. This slice slides down the image panel for every
              // element of ow.
              int ow_padding = ow - PW;
              int image_start = ow_padding * C + kc;
              int image_end = bli_min(W * C, image_start + kc_curr);

              // The start may be negative if it starts in padding, if so, the
              // end already was calculated accordingly (kc_curr is
              // shortened), start is set to zero, and the kc_curr of the
              // filter buf is also shortened. That is, the filter_buf skips
              // the elements that would multiply with the padding.
              if (ow_padding < 0) {
                ar = &filter_buf[-1 * image_start * MR];
                image_start = 0;
              }

              // Get k dimension of the micro kernel
              int K = image_end - image_start;
              if (K <= 0)
                continue;

              // Start of the image block of size kc_curr * NR
              float *br = &image_buf[nr * W * C + image_start * NR];
              // Start of the output block of size NR * mc_curr
              // Here ow varies for every output element and oh varies in
              // blocks of NR minus elements of the filter height. cr starts
              // in arbitrary places of the OH*OW dimension. But results of
              // the multiplication are not fully store contiguously here.
              float *cr = &output[(oh * OW + ow) * M + mc];

              // MR subdivides the block of size MC into smaller blocks
              for (int mr = 0; mr < mc_curr; mr += MR) {

                // In the gemm calls, the image tile is fixed and the filter
                // and output vary in the M dimension
                if (mr + MR <= mc_curr) {
                  // The row and column stride of 1, OW * M make the output be
                  // stored in a strided fashion. Ignoring the output
                  // channels, it is as if output elements were stored in a
                  // single column of the output, which is not contiguous.
                  // Spill elements are stored in the extra space after
                  // because of this stride.
                  //
                  // TODO: make stores contiguous
                  bli_sgemm_ukernel(MR, NR, K, bli_s1, &ar[mr * kc_curr], br,
                                    bli_s1, &cr[mr], 1, OW * M, auxinfo, cntx);
                } else {
                  // beta is 0, meaning C is not accumulated, only written to
                  bli_sgemm_ukernel(MR, NR, K, bli_s1, &ar[mr * kc_curr], br,
                                    bli_s0, output_buf, NR, 1, auxinfo, cntx);
                  bli_sxpbys_mxn(mc_curr - mr, NR, output_buf, NR, 1, bli_s1,
                                 &cr[mr], 1, OW * M);
                }
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

  // TODO: NC could be smaller for the input 1, 64, 64, 64, 128, 3, 3
  // TODO: NC blows up for input 1,1,3,3,1,2,2,1,1,0,0
  //
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
    // Convert single output to NHWC (remove extra space before and after)
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
