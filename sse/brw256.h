#pragma once

// ---------------------------------------------------------------------

#include <emmintrin.h>
#include <stdint.h>

// ---------------------------------------------------------------------

#ifndef BLOCKLEN
#define BLOCKLEN                 16
#endif

#define BRW_CHUNKLEN             16*BLOCKLEN
#define BRW_EXTENDED_CHUNKLEN    22*BLOCKLEN
#define BRW_PRECOMPUTED_POWERS   16 // K, K^2, K^4, K^8, K^16
#define BRW_HASHLEN              32
#define BRW_KEYLEN               32

#if __GNUC__
#define ALIGN(n)      __attribute__ ((aligned(n)))
#elif _MSC_VER
#define ALIGN(n)      __declspec(align(n))
#else
#define ALIGN(n)
#endif

// ---------------------------------------------------------------------

ALIGN(16)
typedef struct {
    __m128i k1[BRW_PRECOMPUTED_POWERS];
    __m128i k2[BRW_PRECOMPUTED_POWERS];
} brwhash_ctx_t;

// ---------------------------------------------------------------------

void brwhash(const brwhash_ctx_t* ctx, 
             const __m128i* header, 
             const size_t hlen, 
             const __m128i* message, 
             const size_t mlen, 
             __m128i result[2]);

// ---------------------------------------------------------------------

void brwhash_keysetup(brwhash_ctx_t* ctx, 
                      const __m128i key[2]);

// ---------------------------------------------------------------------
