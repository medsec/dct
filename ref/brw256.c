#ifdef DEBUG
#include <stdio.h>
#endif
#include <stdint.h>
#include <string.h>
#include "brw256.h"

// ---------------------------------------------------------------------

inline 
static size_t ceil(const size_t len, const size_t step)
{
    if (len == 0 || step == 0) {
        return 0;
    }

    return step * (1 + ((len - 1) / step));
}

// ---------------------------------------------------------------------

#ifdef DEBUG
static void print_hex(const char *label, const uint8_t *c, const size_t len)
{
    printf("%s: \n", label);
    size_t i;

    for (i = 0; i < len; i++) {
        if ((i != 0) && (i % 16 == 0)) {
            puts("");
        }
        printf("%02x ", c[i]);
    }

    puts("\n");
}

// ---------------------------------------------------------------------

inline
static void print_block(const char *label, const uint8_t *c)
{
    print_hex(label, c, BLOCKLEN);
}
#endif

// ---------------------------------------------------------------------

inline 
static void to_big_endian_array(uint8_t* result, const uint64_t src)
{
    for (size_t i = 0; i < 8; ++i) {
        result[7-i] = (src >> (8*i)) & 0xFF;
    }
}

// ---------------------------------------------------------------------

inline
static void to_array(uint8_t* result, const uint64_t src)
{
    for (size_t i = 0; i < 8; ++i) {
        result[i] = (src >> (8*i)) & 0xFF;
    }
}

// ---------------------------------------------------------------------

inline
static uint64_t to_uint64(const uint8_t* x)
{
    uint64_t result = 0L;

    for (size_t i = 0; i < 8; ++i) {
        result |= (uint64_t)x[i] << (i*8);
    }

    return result;
}

// ---------------------------------------------------------------------

inline
static void vxor(block result, const block a, const block b) 
{
    for (size_t i = 0; i < BLOCKLEN; ++i) {
        result[i] = a[i] ^ b[i];
    }
}

// ---------------------------------------------------------------------

static void gf64_multiply(const uint64_t a, const uint64_t b, uint64_t r[2])
{
    uint8_t w = 64;
    uint8_t s = 4; // Window size
    uint64_t two_s = 1 << s; // 2^s
    uint64_t smask = two_s-1; // s-1 bits
    uint64_t u[two_s];
    uint64_t tmp;
    uint64_t ifmask;

    // -----------------------------------------------------------------
    // Precomputation
    // -----------------------------------------------------------------
    uint64_t i;
    u[0] = 0;
    u[1] = b;

    for(i = 2; i < two_s; i += 2) {
        u[i] = u[i >> 1] << 1; // Get even numbers by left shift.
        u[i + 1] = u[i] ^ b;   // Get odd numbers by xoring with b.
    }

    // -----------------------------------------------------------------
    // Multiplication
    // -----------------------------------------------------------------
    
    // The first window affects only the least-significant word.
    r[0] = u[a & smask] ^ (u[(a >> s) & smask] << s); 
    r[1] = 0;

    for(i = 2*s; i < w; i += 2*s) {
        tmp = u[(a >> i) & smask] ^ (u[(a >> (i+s)) & smask] << s);
        r[0] ^= tmp << i;
        r[1] ^= tmp >> (w - i);
    }

    // -----------------------------------------------------------------
    // Reparation
    // -----------------------------------------------------------------
    
    uint64_t m = 0xFEFEFEFEFEFEFEFEL; // s = 2*4
    
    for(i = 1 ; i < 2*s ; i++) {
        tmp = ((a & m) >> i);
        m &= m << 1;
        ifmask = -((b >> (w-i)) & 1); // If the (w-i)-th bit of b is 1.
        r[1] ^= (tmp & ifmask);
    }
}

// ---------------------------------------------------------------------

inline 
static void reduce(const uint64_t x[4], uint64_t r[2])
{
     //reduction of x with polynomial x^128+x^7+x^2+x+1
    const uint64_t ra = x[3] >> (64-1);
    const uint64_t rb = x[3] >> (64-2);
    const uint64_t rc = x[3] >> (64-7);

    const uint64_t rd = x[2] ^ ra ^ rb ^ rc;
    //shift 128 bit word [x3;d] to the left by 1,2,7
    const uint64_t re1 = (x[3] << 1) ^ (rd >> (64-1));
    const uint64_t re0 = rd << 1;
    const uint64_t f1 = (x[3] << 2) ^ (rd >> (64-2));
    const uint64_t f0 = rd << 2;
    const uint64_t g1 = (x[3] << 7) ^ (rd >> (64-7));
    const uint64_t g0 = rd << 7;

    const uint64_t e1Xf1Xg1 = re1 ^ f1 ^ g1;
    const uint64_t e0Xf0Xg0 = re0 ^ f0 ^ g0;

    const uint64_t h1 = x[3] ^ e1Xf1Xg1;
    const uint64_t h0 = rd ^ e0Xf0Xg0;
    r[1] = h1 ^ x[1];
    r[0] = h0 ^ x[0];
}

// ---------------------------------------------------------------------

inline
static void gf128_multiply(const uint64_t a[2], 
                           const uint64_t b[2], 
                           uint64_t r[2])
{
    uint64_t e[2];
    uint64_t x[4];

    gf64_multiply(a[0], b[0], x); // lower = d
    gf64_multiply(a[1], b[1], x+2); // higher = c
    gf64_multiply(a[0] ^ a[1], b[0] ^ b[1], e); // mid = e

    const uint64_t d1Xc0 = x[1] ^ x[2]; // d1 ^ c0
    x[1] = x[0] ^ e[0] ^ d1Xc0; // d0 ^ e0 ^ d1 ^ c0
    x[2] = x[3] ^ e[1] ^ d1Xc0; // d1 ^ e1 ^ d1 ^ c0

    reduce(x, r);
}

// ---------------------------------------------------------------------

inline
static void gfmul(const block a, const block b, block res) 
{
    uint64_t a_[2];
    uint64_t b_[2];
    uint64_t c_[2];
    a_[0] = to_uint64(a);
    a_[1] = to_uint64(a+8);
    b_[0] = to_uint64(b);
    b_[1] = to_uint64(b+8);
    gf128_multiply(a_, b_, c_);
    to_array(res, c_[0]);
    to_array(res+8, c_[1]);
}

// ---------------------------------------------------------------------

inline
static void gf_square(const block a, block res)
{
    uint64_t a_[2];
    uint64_t c_[2];
    a_[0] = to_uint64(a);
    a_[1] = to_uint64(a+8);
    gf128_multiply(a_, a_, c_);
    to_array(res, c_[0]);
    to_array(res+8, c_[1]);
}

// ---------------------------------------------------------------------

static const uint32_t MultiplyDeBruijnBitPosition[32] = 
{
  0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8, 
  31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9
};

inline
static uint32_t find_lowest_set_bit_index(const uint64_t value)
{
    return MultiplyDeBruijnBitPosition[
        ((uint32_t)((value & -value) * 0x077CB531U)) >> 27
    ];
}

// ---------------------------------------------------------------------

inline
static void encode_lengths(uint8_t* final_chunk, 
                           const uint64_t hlen, 
                           const uint64_t mlen)
{
    to_big_endian_array(final_chunk,   hlen);
    to_big_endian_array(final_chunk+8, mlen);
}

// ---------------------------------------------------------------------

inline
static void pad_with_zeroes(uint8_t* final_chunk, const uint64_t len)
{
    if (len % BLOCKLEN == 0) {
        return;
    }

    const uint64_t num_pad_bytes = BLOCKLEN - (len % BLOCKLEN);
    memset(final_chunk, 0x00, num_pad_bytes);
}

// ---------------------------------------------------------------------

static void BRW_n(const uint8_t* k, const uint8_t* m, 
                  const uint64_t len, uint8_t* result)
{
    block left;
    block right;

    // H(X_1,X_2,X_3) = (X_1 xor K) * (X_2 xor K^2) xor X_3
    if (len == 3*BLOCKLEN) { 
        const uint8_t* m2 = m+BLOCKLEN;
        const uint8_t* k2 = k+BLOCKLEN;
        const uint8_t* m3 = m+2*BLOCKLEN;

        vxor(left, m, k);
        vxor(right, m2, k2);
        gfmul(left, right, left);
        vxor(result, left, m3);
    } else if (len == 2*BLOCKLEN) { // H(X_1,X_2) = (X_1 * K) xor X_2
        const uint8_t* m2 = m+BLOCKLEN;

        gfmul(m, k, left);
        vxor(result, m2, left);
    } else if (len == BLOCKLEN) { // H(X_1) = X_1
        memcpy(result, m, BLOCKLEN);
    }
}

// ---------------------------------------------------------------------

/**
 * Computes y = (X_1 xor K) * (X_2 xor K^2) xor X_3 
 * and stores y into result.
 */
static void BRW3(const uint8_t* k, const uint8_t* m, uint8_t* result)
{
    const uint8_t* k2 = k+BLOCKLEN;
    const uint8_t* m2 = m+BLOCKLEN;
    const uint8_t* m3 = m+2*BLOCKLEN;
    
    block left;
    vxor(left, m, k);
    
    block right;
    vxor(right, m2, k2);

    gfmul(left, right, right);
    vxor(result, right, m3);
}

// ---------------------------------------------------------------------

/** 
 * Computes y = [(X_1 xor K) * (X_2 xor K^2) xor X_3] * (X_4 xor K^4)
 * and stores y in result.
 */
static void BRW4(const uint8_t* k, const uint8_t* m, uint8_t* result)
{
    const uint8_t* k4 = k+2*BLOCKLEN;
    const uint8_t* m4 = m+3*BLOCKLEN;
    
    block left;
    block right;
    BRW3(k, m, left);     // X_1,...,X_3

    vxor(right, m4, k4);  // X_4
    gfmul(left, right, result); // X_1,...,X_4
}

// ---------------------------------------------------------------------

/**
 * Computes 
 * y =     [(X_1 xor K) * (X_2 xor K^2) xor X_3]  * (X_4 xor K^4) 
 *     xor [(X_5 xor K) * (X_6 xor K^2) xor X_7]
 * and stores y in result.
 */
static void BRW7(const uint8_t* k, const uint8_t* m, uint8_t* result)
{
    const uint8_t* m5 = m+4*BLOCKLEN;
    
    block left;
    block right;

    BRW4(k, m, left);   // X_1,...,X_4
    BRW3(k, m5, right); // X_5,...,X_7
    vxor(result, right, left); // X_1,...,X_7
}

// ---------------------------------------------------------------------

/**
 * Computes 
 * y =    [[(X_1 xor K) * (X_2 xor K^2) xor X_3]  * (X_4 xor K^4) 
 *     xor [(X_5 xor K) * (X_6 xor K^2) xor X_7]] * (X_8 xor K^8)
 * and stores y in result.
 */
static void BRW8(const uint8_t* k, const uint8_t* m, uint8_t* result)
{

    block left;
    BRW7(k, m, left);
    
    const uint8_t* k8 = k+3*BLOCKLEN;
    const uint8_t* m8 = m+7*BLOCKLEN;
    block right;
    vxor(right, m8, k8);

    gfmul(left, right, result);
}

// ---------------------------------------------------------------------

/**
 * Computes 
 * y =    [[[( X_1 xor K) * ( X_2 xor K^2) xor  X_3]   * ( X_4 xor  K^4) 
 *     xor  [( X_5 xor K) * ( X_6 xor K^2) xor  X_7]]  * ( X_8 xor  K^8)
 *     xor [[(X_9  xor K) * (X_10 xor K^2) xor X_11]   * (X_12 xor  K^4) 
 *     xor  [(X_13 xor K) * (X_14 xor K^2) xor X_15]]] * (X_16 xor K^16)
 * and stores y in result.
 */
static void BRW16(const uint8_t* k, 
                  const block k_last, 
                  const uint8_t* m, 
                  uint8_t* result)
{
    const uint8_t* m9  = m+8*BLOCKLEN;
    const uint8_t* m16 = m+15*BLOCKLEN;

    block left;
    block right;

    BRW8(k, m, left);   // X_1,...,X_8
    BRW7(k, m9, right); // X_9,...,X_15

    vxor(right, right, left);   // X_1,...,X_15
    vxor(left, m16, k_last);    // X_16
    gfmul(left, right, result); // X_1,...,X_16
}

// ---------------------------------------------------------------------

static void hash_tail(const uint8_t* k, 
                      const uint8_t* m,
                      uint64_t i, 
                      uint64_t len, 
                      uint8_t* result)
{
    // ---------------------------------------------------------------------
    // We split the message into chunks of 16 blocks each. The 16-th block of
    // each chunk, M_i, for i = 16 * k, is XORed with a different key power K^j
    // depending on its block index. The key exponent j is determined by j =
    // 2^{Lowest_1_Bit_Index_of(i)}
    // Example: i  16  32  48  64  80  96 112 128 144 160 176
    //          j  16  32  16  64  16  32  16 128  16  32  16
    // ---------------------------------------------------------------------

    uint32_t lowest_set_bit_index;
    block term;
    const uint8_t* k_last;

    if (len >= 16*BLOCKLEN) {
        lowest_set_bit_index = find_lowest_set_bit_index(i);
        k_last = k + (lowest_set_bit_index * BLOCKLEN);

        BRW16(k, k_last, m, term);
        vxor(result, result, term);

        len -= 16*BLOCKLEN;
        m += 16*BLOCKLEN;
        i += 16;
    }

    if (len >= 8*BLOCKLEN) {
        BRW8(k, m, term);
        vxor(result, result, term);

        len -= 8*BLOCKLEN;
        m += 8*BLOCKLEN;
    }

    if (len >= 4*BLOCKLEN) {
        BRW4(k, m, term);
        vxor(result, result, term);

        len -= 4*BLOCKLEN;
        m += 4*BLOCKLEN;
    }

    if (len > 0) {
        BRW_n(k, m, len, term);
        vxor(result, result, term);
    }
}

// ---------------------------------------------------------------------

inline
static void do_hash_ae(const uint8_t* k, 
                       const uint8_t* h,
                       uint64_t hlen, 
                       const uint8_t* m,
                       uint64_t mlen, 
                       uint8_t result[BRW_HASHLEN])
{
    const uint64_t original_hlen = hlen;
    const uint64_t original_mlen = mlen;

    // ---------------------------------------------------------------------
    // Important: Only zeroize the BLOCKLEN bytes that are produced in this 
    // do_hash call.
    // ---------------------------------------------------------------------
    
    memset(result, 0x00, BLOCKLEN); 
    block term;
    
    // ---------------------------------------------------------------------
    // To determine which key the last block of each 16-block chunk must be
    // XORed with. i is the last index of the next chunk, thus it must be 16 at
    // the beginning. k_last must be K^{16} = K[4].
    // ---------------------------------------------------------------------

    uint64_t i = 16;
    uint32_t lowest_set_bit_index;

    const uint8_t* k_last;
    k_last = k + (4 * BLOCKLEN);

    while (hlen >= 16*BLOCKLEN) {
        lowest_set_bit_index = find_lowest_set_bit_index(i);
        k_last = k + (lowest_set_bit_index * BLOCKLEN);

        BRW16(k, k_last, h, term);
        vxor(result, result, term);

        hlen -= 16*BLOCKLEN;
        h += 16*BLOCKLEN;
        i += 16;
    }

    // ---------------------------------------------------------------------
    // Prepare junction chunk.
    // ---------------------------------------------------------------------
    
    uint8_t junction_chunk[17*BLOCKLEN];
    memcpy(junction_chunk, h, hlen);

    pad_with_zeroes(junction_chunk + hlen, hlen);

    uint64_t num_bytes = ceil(hlen, BLOCKLEN);
    uint64_t num_remaining_bytes = (16 * BLOCKLEN) - num_bytes;

    if (num_remaining_bytes >= mlen) {
        memcpy(junction_chunk + num_bytes, m, mlen);
        pad_with_zeroes(junction_chunk + num_bytes + mlen, mlen);
        num_bytes += ceil(mlen, BLOCKLEN);
        
        memset(junction_chunk + num_bytes, 0x00, BLOCKLEN);
        encode_lengths(junction_chunk + num_bytes, original_hlen, original_mlen);
        num_bytes += BLOCKLEN;
        
        hash_tail(k, junction_chunk, i, num_bytes, result);
        gfmul(k, result, result);
        return;
    } else {
        memcpy(junction_chunk + num_bytes, m, num_remaining_bytes);
        num_bytes = 16 * BLOCKLEN;
    }

    // ---------------------------------------------------------------------
    // Process junction chunk.
    // ---------------------------------------------------------------------

    lowest_set_bit_index = find_lowest_set_bit_index(i);
    k_last = k + (lowest_set_bit_index * BLOCKLEN);

    BRW16(k, k_last, junction_chunk, term);
    vxor(result, result, term);

    mlen -= num_remaining_bytes;
    m += num_remaining_bytes;
    i += 16;

    // ---------------------------------------------------------------------
    // Process message.
    // ---------------------------------------------------------------------

    while (mlen >= 16*BLOCKLEN) {
        lowest_set_bit_index = find_lowest_set_bit_index(i);
        k_last = k + (lowest_set_bit_index * BLOCKLEN);

        BRW16(k, k_last, m, term);
        vxor(result, result, term);

        mlen -= 16*BLOCKLEN;
        m += 16*BLOCKLEN;
        i += 16;
    }
    
    // ---------------------------------------------------------------------
    // Prepare final chunk. One extra block for encoding the message length.
    // ---------------------------------------------------------------------
    
    uint8_t final_chunk[17*BLOCKLEN];
    memcpy(final_chunk, m, mlen);

    pad_with_zeroes(final_chunk + mlen, mlen);
    num_bytes = ceil(mlen, BLOCKLEN);

    // ---------------------------------------------------------------------
    // Encode length
    // ---------------------------------------------------------------------
    
    memset(final_chunk + num_bytes, 0x00, BLOCKLEN);
    encode_lengths(final_chunk + num_bytes, original_hlen, original_mlen);
    num_bytes += BLOCKLEN;
    
    hash_tail(k, final_chunk, i, num_bytes, result);

    // ---------------------------------------------------------------------
    // For messages of m block lengths with m != (0 mod 4), pure BRW hashing
    // simply XORs the final message block. Since this would allow to predict
    // differences, one must add a final multiplication with the key afterwards.
    // ---------------------------------------------------------------------

    gfmul(k, result, result);
}

// ---------------------------------------------------------------------

void brwhash(const brwhash_ctx_t* ctx, 
             const uint8_t* header, 
             const size_t hlen, 
             const uint8_t* message, 
             const size_t mlen, 
             uint8_t result[BRW_HASHLEN])
{
    do_hash_ae(ctx->k1, header, hlen, message, mlen, result);
    do_hash_ae(ctx->k2, header, hlen, message, mlen, result+BLOCKLEN);
}

// ---------------------------------------------------------------------

inline
static void precompute_powers(uint8_t* key)
{
    for (size_t i = 1; i < BRW_PRECOMPUTED_POWERS; ++i) {
        uint8_t* next_k = key + (i * BLOCKLEN);
        uint8_t* current_k = key + ((i-1) * BLOCKLEN);
        gf_square(current_k, next_k);
    }
}

// ---------------------------------------------------------------------

void brwhash_keysetup(brwhash_ctx_t* ctx, 
                      const uint8_t key[BRW_KEYLEN])
{
    memcpy(ctx->k1, key, BLOCKLEN);
    memcpy(ctx->k2, key+BLOCKLEN, BLOCKLEN);
    precompute_powers(ctx->k1);
    precompute_powers(ctx->k2);
}
