#pragma once

// ---------------------------------------------------------------------

#include <stdint.h>

// ---------------------------------------------------------------------

#ifndef BLOCKLEN
#define BLOCKLEN                 16
#endif

#define BRW_PRECOMPUTED_POWERS   16 // K, K^2, K^4, K^8, K^16
#define BRW_HASHLEN              32
#define BRW_KEYLEN               32

// ---------------------------------------------------------------------

#ifndef DCT_BLOCK_TYPE
typedef uint8_t block[BLOCKLEN];
#define DCT_BLOCK_TYPE 1
#endif

typedef struct {
    uint8_t k1[BRW_PRECOMPUTED_POWERS * BLOCKLEN];
    uint8_t k2[BRW_PRECOMPUTED_POWERS * BLOCKLEN];
} brwhash_ctx_t;

// ---------------------------------------------------------------------

void brwhash(const brwhash_ctx_t* ctx, 
             const uint8_t* header, 
             const size_t hlen, 
             const uint8_t* message, 
             const size_t mlen, 
             uint8_t result[BRW_HASHLEN]);

// ---------------------------------------------------------------------

void brwhash_keysetup(brwhash_ctx_t* ctx, 
                      const uint8_t key[BRW_KEYLEN]);

// ---------------------------------------------------------------------
