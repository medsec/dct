#pragma once

// ---------------------------------------------------------------------

#include <stddef.h>
#include "aes.h"
#include "api.h"
#include "brw256.h"

// ---------------------------------------------------------------------

#define BLOCKLEN             16
#define KEYLEN               CRYPTO_KEYBYTES
#define KEYLEN_BITS          8*KEYLEN
#define TAGLEN               CRYPTO_ABYTES
#define SIMPIRA_BLOCKLEN     32
#define SIMPIRA_ROUNDS       15
#define DEOXYS_IVLEN         32
#define DEOXYS_ROUNDS        14
#define DEOXYS_ROUND_KEYS    15
#define HASH_KEYS             2

#define MIN_MESSAGE_LEN      16
#define NUM_KEYS              5

// ---------------------------------------------------------------------

#ifndef DCT_BLOCK_TYPE
typedef uint8_t block[BLOCKLEN];
#define DCT_BLOCK_TYPE 1
#endif

typedef unsigned char DEOXYS_KEY[DEOXYS_ROUND_KEYS * BLOCKLEN];

typedef struct {
    unsigned char secret_key[KEYLEN];
    brwhash_ctx_t hash_ctx;
    DEOXYS_KEY expanded_key;
    unsigned char simpira_key[BLOCKLEN];
} dct_context_t;

// ---------------------------------------------------------------------

void keysetup(dct_context_t* ctx, const unsigned char key[KEYLEN]);

// ---------------------------------------------------------------------

void encrypt_final(dct_context_t* ctx, 
                   unsigned char* c, size_t* clen, 
                   const unsigned char* h, const size_t hlen,
                   const unsigned char* m, const size_t mlen);

// ---------------------------------------------------------------------

int decrypt_final(dct_context_t* ctx, 
                  unsigned char* m, size_t* mlen, 
                  const unsigned char* h, const size_t hlen,
                  const unsigned char* c, const size_t clen);
