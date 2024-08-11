/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

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


//
// Prototype object-based check functions.
//

#undef  GENPROT
#define GENPROT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
       FILE*  file, \
       obj_t* x  \
     );

GENPROT( fscanv )

// ---

void bli_utilv_fscan_check
     (
       FILE*  file,
       obj_t* x
     );

// -----------------------------------------------------------------------------

#undef  GENTDEF
#define GENTDEF( ctype, ch, opname, tsuf ) \
\
typedef void (*PASTECH(ch,opname,tsuf)) \
     ( \
       FILE*  file, \
       dim_t  m, \
       ctype* x, inc_t incx  \
     );

INSERT_GENTDEF( fscanv )

// -----------------------------------------------------------------------------

#undef  GENPROT
#define GENPROT( opname ) \
\
PASTECH(opname,_vft) \
PASTEMAC(opname,_qfp)( num_t dt );

GENPROT( fscanv )

// -----------------------------------------------------------------------------

#undef  GENPROT
#define GENPROT( opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(opname) \
     ( \
       FILE*   file, \
       obj_t*  x  \
     );

GENPROT( fscanv )


#undef  GENPROT
#define GENPROT( opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(opname) \
     ( \
       obj_t*  x  \
     );

GENPROT( scanv )

// -----------------------------------------------------------------------------

#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(ch,opname) \
     ( \
       dim_t  m, \
       void*  x, inc_t incx  \
     );

INSERT_GENTPROT_BASIC_I( scanv )

// -----------------------------------------------------------------------------

#undef  GENTPROTR
#define GENTPROTR( ctype, ctype_r, ch, chr, opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(ch,opname) \
     ( \
       FILE*  file, \
       dim_t  m, \
       ctype* x, inc_t incx  \
     );

INSERT_GENTPROTR_BASIC( fscanv )

