#include <emmintrin.h>
#include <immintrin.h>
#include <tmmintrin.h>  // SSSE3 pshufb
#include <wmmintrin.h>
#include <stdint.h>
#ifdef DEBUG
#include <stdio.h>
#endif
#include <string.h>

#include "brw256.h"

// ---------------------------------------------------------------------

#define loadu(p)       _mm_loadu_si128((__m128i*)p)
#define load(p)        _mm_load_si128((__m128i*)p)
#define storeu(p,x)    _mm_storeu_si128((__m128i*)p, x)
#define store(p,x)     _mm_store_si128((__m128i*)p, x)

#define shuffle(x,mask) _mm_shuffle_epi8(x,mask)
#define zero           _mm_setzero_si128()
#define vand(x,y)      _mm_and_si128(x,y)
#define vxor(x,y)      _mm_xor_si128(x,y)
#define vxor3(x,y,z)   _mm_xor_si128(_mm_xor_si128(x,y),z)
#define clmul(x,y,z)   _mm_clmulepi64_si128(x,y,z)
#define ceil16(x)      ((x + 15) & ~15)
#define XMMMASK        _mm_setr_epi32(0xffffffff, 0x0, 0x0, 0x0)

// ---------------------------------------------------------------------

#define load256(p)     _mm256_load_si256((__m256i*)p)
#define loadu256(p)    _mm256_loadu_si256((__m256i*)p)
#define store256(p,x)  _mm256_storeu_si256((__m256i*)p, x)
#define storeu256(p,x) _mm256_store_si256((__m256i*)p, x)

#define zero256        _mm256_setzero_si256()
#define vxor256(x,y)   _mm256_xor_si256(x,y)
#define xmmmask256     _mm256_setr_epi32(0xffffffff, 0x0, 0x0, 0x0, 0xffffffff, 0x0, 0x0, 0x0)

// ---------------------------------------------------------------------

#ifdef DEBUG
static void print_128(const char* label, __m128i var)
{
    uint8_t val[BLOCKLEN];
    store((void*)val, var);
    printf("%s\n", label);
    size_t i;

    for (i = 0; i < BLOCKLEN; ++i) {
        printf("%02x ", val[i]);
    }

    puts("\n");
}

// ---------------------------------------------------------------------

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
#endif

// ---------------------------------------------------------------------
// Modified from 
// http://graphics.stanford.edu/~seander/bithacks.html
// http://www.blitzbasic.com/Community/posts.php?topic=72718
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
    storeu(final_chunk, shuffle(
        _mm_set_epi64((__m64)mlen, (__m64)hlen), 
        _mm_setr_epi8(7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8))
    );
}

// ---------------------------------------------------------------------

inline
static void pad_with_zeroes(uint8_t* final_chunk, const uint64_t len)
{
    const uint64_t num_pad_bytes = (BLOCKLEN - (len & 0xF)) & 0xF;
    memset(final_chunk, 0x00, num_pad_bytes);
}

// ---------------------------------------------------------------------

/**
 * lo, hi
 */
inline
static void REDUCE(__m128i tmp1, __m128i tmp2, __m128i* result)
{
    __m128i tmp3, tmp4;
    tmp3 = _mm_srli_epi32(tmp2, 31);         // [X3 >> 63:X3_lo >> 31:X2 >> 63:X2_lo >> 31] = [A:::] 
    tmp4 = _mm_srli_epi32(tmp2, 30);         // [X3 >> 62:X3_lo >> 30:X2 >> 62:X2_lo >> 30] = [B:::] 
    tmp3 = _mm_xor_si128(tmp3, tmp4);        // [A ^ B:::] 
    tmp4 = _mm_srli_epi32(tmp2, 25);         // [X3 >> 57:X3_lo >> 25:X2 >> 57:X2_lo >> 25] = [C:::] 
    tmp3 = _mm_xor_si128(tmp3, tmp4);        // [A ^ B ^ C:Y3_lo:Y2_hi:Y2_lo] 
    
    tmp4 = _mm_shuffle_epi32(tmp3, 0x93);    // [Y3_lo:Y2_hi:Y2_lo:A ^ B ^ C] 
    tmp3 = _mm_and_si128(XMMMASK, tmp4);     // [    0:    0:    0:A ^ B ^ C] 
    tmp4 = _mm_andnot_si128(XMMMASK, tmp4);  // [Y3_lo:Y2_hi:Y2_lo:        0] 
    tmp1 = _mm_xor_si128(tmp1, tmp4);        // [Y3_lo:Y2_hi:Y2_lo:        0] ^ [X1:X0] 
    tmp2 = _mm_xor_si128(tmp2, tmp3);        // [X3   :X2 ^ A ^ B ^ C] = [X3:D] 

    tmp3 = _mm_slli_epi32(tmp2, 1);          // [X3_hi << 1:X3_lo << 1:D_hi << 1:D_lo << 1] 
    tmp1 = _mm_xor_si128(tmp1, tmp3);        // [          X1 ^ E1:          X0 ^ E0] 
    tmp4 = _mm_slli_epi32(tmp2, 2);          // [X3_hi << 2:X3_lo << 2:D_hi << 2:D_lo << 2] 
    tmp1 = _mm_xor_si128(tmp1, tmp4);        // [     X1 ^ F1 ^ E1:     X0 ^ F0 ^ E0] 
    tmp4 = _mm_slli_epi32(tmp2, 7);          // [X3_hi << 7:X3_lo << 7:D_hi << 7:D_lo << 7] 
    tmp1 = _mm_xor_si128(tmp1, tmp4);        // [X1 ^ G1 ^ F1 ^ E1:X0 ^ G0 ^ F0 ^ E0] 
    *result = _mm_xor_si128(tmp1, tmp2);     // [X3 ^ X1 ^ G1 ^ F1 ^ E1:X0 ^ G0 ^ F0 ^ E0] 
}

// ---------------------------------------------------------------------

inline
static void REDUCE256(__m256i x1, __m256i x2, __m256i* result)
{
    __m256i x3, x4, x5;
    x3 = _mm256_srli_epi32(x2, 31);          // [X3 >> 63:X3_lo >> 31:X2 >> 63:X2_lo >> 31] = [A:::] 
    x4 = _mm256_srli_epi32(x2, 30);          // [X3 >> 62:X3_lo >> 30:X2 >> 62:X2_lo >> 30] = [B:::] 
    x5 = _mm256_srli_epi32(x2, 25);          // [X3 >> 57:X3_lo >> 25:X2 >> 57:X2_lo >> 25] = [C:::] 
    x3 = vxor256(x3, x4);                    // [A ^ B:::] 
    x3 = vxor256(x3, x5);                    // [A ^ B ^ C:Y3_lo:Y2_hi:Y2_lo] 
    
    x4 = _mm256_shuffle_epi32(x3, 0x93);     // [Y3_lo:Y2_hi:Y2_lo:A ^ B ^ C] 
    x3 = _mm256_and_si256(xmmmask256, x4);   // [    0:    0:    0:A ^ B ^ C] 
    x4 = _mm256_andnot_si256(xmmmask256, x4);// [Y3_lo:Y2_hi:Y2_lo:        0] 
    x1 = vxor256(x1, x4);                    // [Y3_lo:Y2_hi:Y2_lo:        0] ^ [X1:X0] 
    x2 = vxor256(x2, x3);                    // [X3   :X2 ^ A ^ B ^ C] = [X3:D] 

    x3 = _mm256_slli_epi32(x2, 1);           // [X3_hi << 1:X3_lo << 1:D_hi << 1:D_lo << 1] 
    x4 = _mm256_slli_epi32(x2, 2);           // [X3_hi << 2:X3_lo << 2:D_hi << 2:D_lo << 2] 
    x5 = _mm256_slli_epi32(x2, 7);           // [X3_hi << 7:X3_lo << 7:D_hi << 7:D_lo << 7] 
    x1 = vxor256(x1, x3);                    // [          X1 ^ E1:          X0 ^ E0] 
    x4 = vxor256(x4, x5);                    // [     X1 ^ F1 ^ E1:     X0 ^ F0 ^ E0] 
    x1 = vxor256(x1, x2);                    // [X1 ^ G1 ^ F1 ^ E1:X0 ^ G0 ^ F0 ^ E0] 
    *result = vxor256(x1, x4);               // [X3 ^ X1 ^ G1 ^ F1 ^ E1:X0 ^ G0 ^ F0 ^ E0] 
}

// ---------------------------------------------------------------------

inline
static void REDUCE256_2(__m256i x1_1, __m256i x1_2, __m256i* result1, 
                        __m256i x2_1, __m256i x2_2, __m256i* result2)
{
    __m256i x1_3, x1_4, x1_5;
    __m256i x2_3, x2_4, x2_5;
    x1_3 = _mm256_srli_epi32(x1_2, 31);
    x1_4 = _mm256_srli_epi32(x1_2, 30);
    x1_5 = _mm256_srli_epi32(x1_2, 25);
    x2_3 = _mm256_srli_epi32(x2_2, 31);
    x2_4 = _mm256_srli_epi32(x2_2, 30);
    x2_5 = _mm256_srli_epi32(x2_2, 25);
    x1_3 = vxor256(x1_3, x1_4);
    x1_3 = vxor256(x1_3, x1_5);
    x2_3 = vxor256(x2_3, x2_4);
    x2_3 = vxor256(x2_3, x2_5);
    
    x1_4 = _mm256_shuffle_epi32(x1_3, 0x93);
    x2_4 = _mm256_shuffle_epi32(x2_3, 0x93);
    x1_3 = _mm256_and_si256(xmmmask256, x1_4);
    x2_3 = _mm256_and_si256(xmmmask256, x2_4);
    x1_4 = _mm256_andnot_si256(xmmmask256, x1_4);
    x2_4 = _mm256_andnot_si256(xmmmask256, x2_4);
    x1_1 = vxor256(x1_1, x1_4);
    x1_2 = vxor256(x1_2, x1_3);
    x2_1 = vxor256(x2_1, x2_4);
    x2_2 = vxor256(x2_2, x2_3);

    x1_3 = _mm256_slli_epi32(x1_2, 1);
    x1_4 = _mm256_slli_epi32(x1_2, 2);
    x1_5 = _mm256_slli_epi32(x1_2, 7);
    x2_3 = _mm256_slli_epi32(x2_2, 1);
    x2_4 = _mm256_slli_epi32(x2_2, 2);
    x2_5 = _mm256_slli_epi32(x2_2, 7);
    x1_1 = vxor256(x1_1, x1_3);
    x1_4 = vxor256(x1_4, x1_5);
    x1_1 = vxor256(x1_1, x1_2);
    *result1 = vxor256(x1_1, x1_4);
    
    x2_1 = vxor256(x2_1, x2_3);
    x2_4 = vxor256(x2_4, x2_5);
    x2_1 = vxor256(x2_1, x2_2);
    *result2 = vxor256(x2_1, x2_4);
}

// ---------------------------------------------------------------------

#define COMPUTE_E(A, B, E) do {            \
    A = vxor(_mm_shuffle_epi32(A, 78), A); \
    B = vxor(_mm_shuffle_epi32(B, 78), B); \
    E = clmul(A, B, 0x00);                 \
} while (0)

// ---------------------------------------------------------------------

#define COMPUTE_X(lo, hi, C, D, E) do {    \
    lo = vxor3(C, D, E);                   \
    hi = _mm_srli_si128(lo, 0x08);         \
    lo = _mm_slli_si128(lo, 0x08);         \
    lo = vxor(lo, D);                      \
    hi = vxor(hi, C);                      \
} while (0)

// ---------------------------------------------------------------------

#define COMPUTE_X256(lo, hi, C1, D1, E1, C2, D2, E2) do {                         \
    E2 = vxor3(C2, D2, E2);                                                       \
    lo = _mm256_castsi128_si256(vxor3(C1, D1, E1));                               \
    lo = _mm256_insertf128_si256(lo, E2, 1);                                      \
    hi = _mm256_srli_si256(lo, 0x08);                                             \
    lo = _mm256_slli_si256(lo, 0x08);                                             \
    lo = vxor256(lo, _mm256_insertf128_si256(_mm256_castsi128_si256(D1), D2, 1)); \
    hi = vxor256(hi, _mm256_insertf128_si256(_mm256_castsi128_si256(C1), C2, 1)); \
} while (0)

// ---------------------------------------------------------------------

inline 
static void gfmul(__m128i A, __m128i B, __m128i* result) 
{
    __m128i C = clmul(A, B, 0x11);
    __m128i D = clmul(A, B, 0x00);
    __m128i E;
    COMPUTE_E(A, B, E);
    __m128i lo, hi;
    COMPUTE_X(lo, hi, C, D, E);
    REDUCE(lo, hi, result);
}

// ---------------------------------------------------------------------

inline
static void gf_square(__m128i a, __m128i* result)
{
    __m128i tmp1, tmp2, tmp4;
    __m128i maskl, maskh, table;

    table = _mm_set_epi32(0x55545150, 0x45444140, 0x15141110, 0x05040100);
    maskl = _mm_set1_epi32(0x0F0F0F0F);
    maskh = _mm_set1_epi32(0xF0F0F0F0);
    tmp1 = _mm_and_si128(a, maskh);
    tmp2 = _mm_and_si128(a, maskl);

    tmp1 = _mm_srli_epi64(tmp1, 0x04);
    
    tmp1 = _mm_shuffle_epi8(table, tmp1);
    tmp2 = _mm_shuffle_epi8(table, tmp2);

    tmp4 = _mm_unpackhi_epi8(tmp2, tmp1); // hi
    tmp1 = _mm_unpacklo_epi8(tmp2, tmp1); // low

    REDUCE(tmp1, tmp4, result);
}

// ---------------------------------------------------------------------

inline
static void gfmul4(__m128i A1, __m128i B1, __m128i* res1, 
                   __m128i A2, __m128i B2, __m128i* res2, 
                   __m128i A3, __m128i B3, __m128i* res3, 
                   __m128i A4, __m128i B4, __m128i* res4) 
{
    __m128i D1 = clmul(A1,B1,0x00);
    __m128i C1 = clmul(A1,B1,0x11);
    __m128i D2 = clmul(A2,B2,0x00);
    __m128i C2 = clmul(A2,B2,0x11);
    __m128i D3 = clmul(A3,B3,0x00);
    __m128i C3 = clmul(A3,B3,0x11);
    __m128i D4 = clmul(A4,B4,0x00);
    __m128i C4 = clmul(A4,B4,0x11);

    __m128i E1, E2, E3, E4;
    COMPUTE_E(A1, B1, E1);
    COMPUTE_E(A2, B2, E2);
    COMPUTE_E(A3, B3, E3);
    COMPUTE_E(A4, B4, E4);

    __m256i A12_lo, A12_hi, A34_lo, A34_hi;
    COMPUTE_X256(A12_lo, A12_hi, C1, D1, E1, C2, D2, E2);
    COMPUTE_X256(A34_lo, A34_hi, C3, D3, E3, C4, D4, E4);
    REDUCE256_2(A12_lo, A12_hi, &A12_lo, A34_lo, A34_hi, &A34_lo);

    *res1 = _mm256_castsi256_si128(A12_lo);
    *res2 = _mm256_extracti128_si256(A12_lo, 1);
    *res3 = _mm256_castsi256_si128(A34_lo);
    *res4 = _mm256_extracti128_si256(A34_lo, 1);
}

// ---------------------------------------------------------------------

inline
static void gfmul2(__m128i A1, __m128i B1, __m128i* res1, 
                   __m128i A2, __m128i B2, __m128i* res2) 
{
    __m128i D1 = clmul(A1,B1,0x00);
    __m128i C1 = clmul(A1,B1,0x11);
    __m128i D2 = clmul(A2,B2,0x00);
    __m128i C2 = clmul(A2,B2,0x11);

    __m128i E1, E2;
    COMPUTE_E(A1, B1, E1);
    COMPUTE_E(A2, B2, E2);
    
    __m256i A12_lo, A12_hi;
    COMPUTE_X256(A12_lo, A12_hi, C1, D1, E1, C2, D2, E2);
    REDUCE256(A12_lo, A12_hi, &A12_lo);
    *res1 = _mm256_castsi256_si128(A12_lo);
    *res2 = _mm256_extracti128_si256(A12_lo, 1);
}

// ---------------------------------------------------------------------

#define vxor2pairs(a,b,c,d) do { \
    a = _mm_xor_si128(a, b);     \
    c = _mm_xor_si128(c, d);     \
} while (0)

// ---------------------------------------------------------------------

#define vxor3pairs(a,b,c,d,e,f) do { \
    a = _mm_xor_si128(a, b);         \
    c = _mm_xor_si128(c, d);         \
    e = _mm_xor_si128(e, f);         \
} while (0)

// ---------------------------------------------------------------------

#define vxor4pairs(a,b,c,d,e,f,g,h) do { \
    a = _mm_xor_si128(a, b);             \
    c = _mm_xor_si128(c, d);             \
    e = _mm_xor_si128(e, f);             \
    g = _mm_xor_si128(g, h);             \
} while (0)

// ---------------------------------------------------------------------

/**
 * Computes the first four multiplications for a 32-block-chunk.
 * Assumes four variables __m128i y1_1, y1_2, y2_1, y2_2 be defined.
 */
#define BRW16_INITIAL_TWO_MULTS(m, k1, k2) do {\
    gfmul4(vxor(m[ 0], k1[0]), vxor(m[ 1], k1[1]), &y1_1, /* y1_1 = m_{ 1.. 2} */\
           vxor(m[ 4], k1[0]), vxor(m[ 5], k1[1]), &y1_2, /* y1_2 = m_{ 5.. 6} */\
           vxor(m[ 0], k2[0]), vxor(m[ 1], k2[1]), &y2_1, /* y2_1 = m_{ 1.. 2} */\
           vxor(m[ 4], k2[0]), vxor(m[ 5], k2[1]), &y2_2);/* y2_2 = m_{ 5.. 6} */\
    vxor4pairs(y1_1, m[ 2],  /* y1_1 = m_{ 1.. 3} */\
               y1_2, m[ 6],  /* y1_2 = m_{ 5.. 7} */\
               y2_1, m[ 2],  /* y2_1 = m_{ 1.. 3} */\
               y2_2, m[ 6]); /* y2_2 = m_{ 5.. 7} */\
} while (0)

// ---------------------------------------------------------------------

/**
 * Produces a BRW hash for 16 blocks, Uses a scheduling of 2-2-2-1-1
 * multiplications. For 256-bit hashing, this implies we do 4-4-4-2-2
 * multiplications. Assumes eight variables __m128i y1_1, .. y1_4, y2_1, ..,
 * y2_4 be defined. Assumes that y1_1, y1_2, y2_1, y2_2 are already initialized
 * with BRW16_INITIAL_TWO_MULTS.
 */
#define BRW16_22211(hash1, hash2, m, k1, k1_last, k2, k2_last) do {              \
    /* y1_1 = m_{ 1.. 3} */                                                      \
    /* y2_1 = m_{ 1.. 3} */                                                      \
    /* y1_2 = m_{ 5.. 7} */                                                      \
    /* y2_2 = m_{ 5.. 7} */                                                      \
    gfmul4(vxor(m[ 3], k1[2]), y1_1, &y1_1,              /* y1_1 = m_{ 1.. 4} */ \
           vxor(m[ 3], k2[2]), y2_1, &y2_1,              /* y2_1 = m_{ 1.. 4} */ \
           vxor(m[ 8], k1[0]), vxor(m[ 9], k1[1]), &y1_3, /* y1_3 = m_{ 9..10} */\
           vxor(m[ 8], k2[0]), vxor(m[ 9], k2[1]), &y2_3);/* y2_3 = m_{ 9..10} */\
    vxor4pairs(y1_1, y1_2,   /* y1_1 = m_{ 1.. 7} */                             \
               y2_1, y2_2,   /* y2_1 = m_{ 1.. 7} */                             \
               y1_3, m[10],  /* y1_3 = m_{ 9..11} */                             \
               y2_3, m[10]); /* y2_3 = m_{ 9..11} */                             \
    gfmul4(vxor(m[ 7], k1[3]), y1_1, &y1_1,              /* y1_1 = m_{ 1.. 8} */ \
           vxor(m[ 7], k2[3]), y2_1, &y2_1,              /* y2_1 = m_{ 1.. 8} */ \
           vxor(m[11], k1[2]), y1_3, &y1_3,              /* y1_3 = m_{ 9..12} */ \
           vxor(m[11], k2[2]), y2_3, &y2_3);             /* y2_3 = m_{ 9..12} */ \
    vxor2pairs(y1_1, y1_3,   /* y1_1 = m_{ 1..12} */                             \
               y2_1, y2_3);  /* y2_1 = m_{ 1..12} */                             \
    gfmul2(vxor(m[12], k1[0]), vxor(m[13], k1[1]), &y1_2, /* y1_2 = m_{13..14} */\
           vxor(m[12], k2[0]), vxor(m[13], k2[1]), &y2_2);/* y2_2 = m_{13..14} */\
    vxor2pairs(y1_2, m[14],  /* y1_2 = m_{13..15} */                             \
               y2_2, m[14]); /* y2_2 = m_{13..15} */                             \
    vxor2pairs(y1_1, y1_2,   /* y1_1 = m_{ 1..15} */                             \
               y2_1, y2_2);  /* y2_1 = m_{ 1..15} */                             \
    gfmul2(vxor(m[15], k1_last), y1_1, &hash1,                                   \
           vxor(m[15], k2_last), y2_1, &hash2);                                  \
} while (0)

// ---------------------------------------------------------------------

/**
 * Produces a BRW hash for 16 blocks. Uses a scheduling of 2-2-1(+1)-1(+1)
 * multiplications, where the #multiplications in parentheses are from the next
 * 16-block chunk. For 256-bit hashing, we perform 4-4-2(+2)-2(+2)
 * multiplications. Assumes eight variables __m128i y1_1, .. y1_4, y2_1, ..,
 * y2_4 be defined. Assumes that y1_1, y1_2, y2_1, y2_2 are already initialized
 * with BRW16_INITIAL_TWO_MULTS.
 */
#define BRW16_2222(hash1, hash2, m, k1, k1_last, k2, k2_last) do {                 \
    /* y1_1 = m_{ 1.. 3} */                                                        \
    /* y2_1 = m_{ 1.. 3} */                                                        \
    /* y1_2 = m_{ 5.. 7} */                                                        \
    /* y2_2 = m_{ 5.. 7} */                                                        \
    gfmul4(vxor(m[ 3], k1[2]), y1_1, &y1_1,               /* y1_1 = m_{ 1.. 4} */  \
           vxor(m[ 3], k2[2]), y2_1, &y2_1,               /* y2_1 = m_{ 1.. 4} */  \
           vxor(m[ 8], k1[0]), vxor(m[ 9], k1[1]), &y1_3, /* y1_3 = m_{ 9..10} */  \
           vxor(m[ 8], k2[0]), vxor(m[ 9], k2[1]), &y2_3);/* y2_3 = m_{ 9..10} */  \
    vxor4pairs(y1_1, y1_2,   /* y1_1 = m_{ 1.. 7} */                               \
               y2_1, y2_2,   /* y2_1 = m_{ 1.. 7} */                               \
               y1_3, m[10],  /* y1_3 = m_{ 9..11} */                               \
               y2_3, m[10]); /* y2_3 = m_{ 9..11} */                               \
    gfmul4(vxor(m[ 7], k1[3]), y1_1, &y1_1, /* y1_1 = m_{ 1.. 8} */                \
           vxor(m[ 7], k2[3]), y2_1, &y2_1, /* y2_1 = m_{ 1.. 8} */                \
           vxor(m[11], k1[2]), y1_3, &y1_3, /* y1_3 = m_{ 9..12} */                \
           vxor(m[11], k2[2]), y2_3, &y2_3);/* y2_3 = m_{ 9..12} */                \
    vxor2pairs(y1_1, y1_3,   /* y1_1 = m_{ 1..12} */                               \
               y2_1, y2_3);  /* y2_1 = m_{ 1..12} */                               \
    gfmul4(vxor(m[12], k1[0]), vxor(m[13], k1[1]), &y1_3, /* y1_3 = m_{13..14} */  \
           vxor(m[12], k2[0]), vxor(m[13], k2[1]), &y2_3, /* y2_3 = m_{13..14} */  \
           vxor(m[16], k1[0]), vxor(m[17], k1[1]), &y1_4, /* y1_4 = m_{17..18} */  \
           vxor(m[16], k2[0]), vxor(m[17], k2[1]), &y2_4);/* y2_4 = m_{17..18} */  \
    vxor4pairs(y1_3, m[14],  /* y1_3 = m_{13..15} */                               \
               y2_3, m[14],  /* y2_3 = m_{13..15} */                               \
               y1_4, m[18],  /* y1_4 = m_{17..19} */                               \
               y2_4, m[18]); /* y2_4 = m_{17..19} */                               \
    gfmul4(vxor(m[15], k1_last), vxor(y1_1, y1_3), &hash1,/* y1_3 = m_{13..14} */  \
           vxor(m[15], k2_last), vxor(y2_1, y2_3), &hash2,/* y1_3 = m_{13..14} */  \
           vxor(m[20], k1[0]), vxor(m[21], k1[1]), &y1_3,   /* y1_3 = m_{21..22} */\
           vxor(m[20], k2[0]), vxor(m[21], k2[1]), &y2_3);  /* y2_3 = m_{21..22} */\
    vxor2pairs(y1_3, m[22],  /* y1_3 = m_{21..23} */                               \
               y2_3, m[22]); /* y2_3 = m_{21..23} */                               \
    y1_1 = y1_4;             /* y1_1 = m_{17..19} */                               \
    y2_1 = y2_4;             /* y2_1 = m_{17..19} */                               \
    y1_2 = y1_3;             /* y1_2 = m_{21..23} */                               \
    y2_2 = y2_3;             /* y2_2 = m_{21..23} */                               \
} while (0)

// ---------------------------------------------------------------------

inline
static void BRW_n(const uint32_t num_blocks, const __m128i* k1, 
                 const __m128i* k2, const __m128i* m, 
                 __m128i* term1, __m128i* term2)
{
    if (num_blocks == 1) {
        *term1 = m[0];
        *term2 = m[0];
    } else if (num_blocks == 2) {
        gfmul2(m[0], k1[0], term1, 
               m[0], k2[0], term2); 
        *term1 = vxor(*term1, m[1]);
        *term2 = vxor(*term2, m[1]);
    } else {
        gfmul2(vxor(m[0], k1[0]), vxor(m[1], k1[1]), term1, 
               vxor(m[0], k2[0]), vxor(m[1], k2[1]), term2);

        *term1 = vxor(*term1, m[2]);
        *term2 = vxor(*term2, m[2]);
    }
}

// ---------------------------------------------------------------------

inline
static void BRW3(const __m128i* k1, const __m128i* k2, const __m128i* m, 
                 __m128i* term1, __m128i* term2)
{
    gfmul2(vxor(m[0], k1[0]), vxor(m[1], k1[1]), term1, 
           vxor(m[0], k2[0]), vxor(m[1], k2[1]), term2);
    *term1 = vxor(*term1, m[2]);
    *term2 = vxor(*term2, m[2]);
}

// ---------------------------------------------------------------------

inline
static void BRW4(const __m128i* k1, const __m128i* k2, 
                 const __m128i* m, __m128i* term1, __m128i* term2)
{
    BRW3(k1, k2, m, term1, term2);
    gfmul2(*term1, vxor(m[3], k1[2]), term1, 
           *term2, vxor(m[3], k2[2]), term2);
}

// ---------------------------------------------------------------------

inline
static void BRW8(const __m128i* k1, const __m128i* k2, 
                 const __m128i* m, __m128i* term1, __m128i* term2)
{
    __m128i x1_1to4, x2_1to4;
    BRW4(k1, k2, m, &x1_1to4, &x2_1to4);

    __m128i x1_5to7, x2_5to7;
    BRW3(k1, k2, m+4, &x1_5to7, &x2_5to7);

    gfmul2(vxor(x1_5to7, x1_1to4), vxor(m[7], k1[3]), term1, 
           vxor(x2_5to7, x2_1to4), vxor(m[7], k2[3]), term2);
}

// ---------------------------------------------------------------------

inline
static void hash_tail(const __m128i* k1, 
                      const __m128i* k2, 
                      const __m128i k1_last, 
                      const __m128i k2_last, 
                      __m128i* m,
                      uint64_t len, 
                      __m128i* hash1, 
                      __m128i* hash2)
{
    __m128i term1;
    __m128i term2;

    if (len >= BRW_CHUNKLEN) {
        __m128i y1_1, y1_2, y1_3, y2_1, y2_2, y2_3;
        BRW16_INITIAL_TWO_MULTS(m, k1, k2);
        BRW16_22211(term1, term2, m, k1, k1_last, k2, k2_last);

        *hash1 = vxor(*hash1, term1);
        *hash2 = vxor(*hash2, term2);

        len -= BRW_CHUNKLEN;
        m += 16;
    }

    if (len >= 8*BLOCKLEN) {
        BRW8(k1, k2, m, &term1, &term2);
        vxor2pairs(*hash1, term1, *hash2, term2);

        len -= 8*BLOCKLEN;
        m += 8;
    }

    if (len >= 4*BLOCKLEN) {
        BRW4(k1, k2, m, &term1, &term2);
        vxor2pairs(*hash1, term1, *hash2, term2);

        len -= 4*BLOCKLEN;
        m += 4;
    }

    if (len > 0) {
        const uint32_t num_blocks = len / BLOCKLEN;
        BRW_n(num_blocks, k1, k2, m, &term1, &term2);
        vxor2pairs(*hash1, term1, *hash2, term2);
    }
}

// ---------------------------------------------------------------------

#define process_junction(h, hlen, m, mlen) do {                                         \
    memcpy(chunk, h, hlen);                                                             \
    pad_with_zeroes(chunk + hlen, hlen);                                                \
    memcpy(chunk + num_bytes, m, num_remaining_bytes);                                  \
    num_bytes = BRW_CHUNKLEN;                                                           \
    BRW16_INITIAL_TWO_MULTS(((__m128i*)chunk), k1, k2); /* [0], k1[1], k2[0], k2[1]); */\
    BRW16_22211(term1, term2, ((__m128i*)chunk), k1, k1_last, k2, k2_last);             \
    hash1 = vxor(hash1, term1);                                                         \
    hash2 = vxor(hash2, term2);                                                         \
    mlen -= num_remaining_bytes;                                                        \
    m += num_remaining_bytes / BLOCKLEN;                                                \
    i += 16;                                                                            \
    lowest_set_bit_index = find_lowest_set_bit_index(i);                                \
    k1_last = k1[lowest_set_bit_index];                                                 \
    k2_last = k2[lowest_set_bit_index];                                                 \
} while (0)

// ---------------------------------------------------------------------

#define process_junction_tail(h, hlen, m, mlen) do {                                 \
    memcpy(chunk, h, hlen);                                                          \
    pad_with_zeroes(chunk + hlen, hlen);                                             \
    memcpy(chunk + num_bytes, m, mlen);                                              \
    pad_with_zeroes(chunk + num_bytes + mlen, mlen);                                 \
    num_bytes += ceil16(mlen);                                                       \
    encode_lengths(chunk + num_bytes, original_hlen, original_mlen);                 \
    num_bytes += BLOCKLEN;                                                           \
    hash_tail(k1, k2, k1_last, k2_last, (__m128i*)chunk, num_bytes, &hash1, &hash2); \
    gfmul(k1[0], hash1, result);                                                     \
    gfmul(k2[0], hash2, result+1);                                                   \
} while (0)

// ---------------------------------------------------------------------

#define process_tail(m, mlen) do {                                                   \
    memcpy(chunk, m, mlen);                                                          \
    pad_with_zeroes(chunk + mlen, mlen);                                             \
    num_bytes = ceil16(mlen);                                                        \
    encode_lengths(chunk + num_bytes, original_hlen, original_mlen);                 \
    num_bytes += BLOCKLEN;                                                           \
    hash_tail(k1, k2, k1_last, k2_last, (__m128i*)chunk, num_bytes, &hash1, &hash2); \
    gfmul(k1[0], hash1, result);                                                     \
    gfmul(k2[0], hash2, result+1);                                                   \
} while (0)

// ---------------------------------------------------------------------

inline
static void do_hash_ae(const __m128i* k1, 
                       const __m128i* k2, 
                       __m128i* h,
                       uint64_t hlen, 
                       __m128i* m,
                       uint64_t mlen, 
                       __m128i* result)
{
    // ---------------------------------------------------------------------
    // We must zeroize the result.
    // ---------------------------------------------------------------------
    
    __m128i hash1 = zero;
    __m128i hash2 = zero;
    __m128i term1;
    __m128i term2;
    __m128i y1_1, y1_2, y1_3, y1_4, y2_1, y2_2, y2_3, y2_4;

    // ---------------------------------------------------------------------
    // To determine which key the last block of each 16-block chunk must be
    // XORed with. i is the last index of the next chunk, thus it must be 16 at
    // the beginning. k_last must be K^{16} = K[4].
    // ---------------------------------------------------------------------

    const uint64_t original_hlen = hlen;
    const uint64_t original_mlen = mlen;

    __m128i k1_last = k1[4];
    __m128i k2_last = k2[4];
    uint64_t i = 16;
    uint32_t lowest_set_bit_index;
    
    // ---------------------------------------------------------------------
    // We split the header into 16-block chunks.
    // ---------------------------------------------------------------------

    while (hlen >= BRW_CHUNKLEN) {
        BRW16_INITIAL_TWO_MULTS(h, k1, k2);
        BRW16_22211(term1, term2, h, k1, k1_last, k2, k2_last);

        hash1 = vxor(hash1, term1);
        hash2 = vxor(hash2, term2);
        hlen -= BRW_CHUNKLEN;
        i += 16;
        h += 16;
        
        // ---------------------------------------------------------------------
        // The key used depends on the block index, for example:
        // i = 0111000 => use K^{2^3} = K[3]
        // i = 1100000 => use K^{2^5} = K[5]
        // ---------------------------------------------------------------------

        lowest_set_bit_index = find_lowest_set_bit_index(i);
        k1_last = k1[lowest_set_bit_index];
        k2_last = k2[lowest_set_bit_index];
    }

    // ---------------------------------------------------------------------
    // Process junction chunk.
    // ---------------------------------------------------------------------
    
    uint64_t num_bytes = ceil16(hlen);
    uint64_t num_remaining_bytes = BRW_CHUNKLEN - num_bytes;
    uint8_t chunk[BRW_CHUNKLEN + BLOCKLEN];
    
    if (num_remaining_bytes >= mlen) {
        process_junction_tail(h, hlen, m, mlen);
        return;
    } else {
        process_junction(h, hlen, m, mlen);
    }

    // ---------------------------------------------------------------------
    // Process message.
    // ---------------------------------------------------------------------
    
    // ---------------------------------------------------------------------
    // To exploit the 2-cycle inverse throughput and 7-cycle latency of the
    // pclmulqdq instruction on Haswell, we perform 4 multiplications in
    // parallel. We split the message into 16-block chunks. Since BRW
    // polynomials form a dependency tree, the last (2 x )2 multiplications of
    // each chunk are performed together with the first (2 x )2 multiplications
    // of the succeeding chunk, which concern the first 6 message blocks of the
    // next chunk. So, we must check if there are >= 22 next message blocks.
    // ---------------------------------------------------------------------
    
    int was_initialized = 0;

    if (mlen >= BRW_EXTENDED_CHUNKLEN) {
        BRW16_INITIAL_TWO_MULTS(m, k1, k2);
        was_initialized = 1;
    }
    
    while (mlen >= BRW_EXTENDED_CHUNKLEN) {
        BRW16_2222(term1, term2, m, k1, k1_last, k2, k2_last);
        hash1 = vxor(hash1, term1);
        hash2 = vxor(hash2, term2);
        mlen -= BRW_CHUNKLEN;
        i += 16;
        m += 16;

        // ---------------------------------------------------------------------
        // The key used depends on the block index, for example:
        // i = 0111000 => use K^{2^3} = K[3]
        // i = 1100000 => use K^{2^5} = K[5]
        // ---------------------------------------------------------------------

        lowest_set_bit_index = find_lowest_set_bit_index(i);
        k1_last = k1[lowest_set_bit_index];
        k2_last = k2[lowest_set_bit_index];
    }

    if (mlen >= BRW_CHUNKLEN) {
        if (!was_initialized) {
            BRW16_INITIAL_TWO_MULTS(m, k1, k2);
        }
        
        BRW16_22211(term1, term2, m, k1, k1_last, k2, k2_last);

        hash1 = vxor(hash1, term1);
        hash2 = vxor(hash2, term2);

        mlen -= BRW_CHUNKLEN;
        i += 16;
        m += 16;

        lowest_set_bit_index = find_lowest_set_bit_index(i);
        k1_last = k1[lowest_set_bit_index];
        k2_last = k2[lowest_set_bit_index];
    }

    // ---------------------------------------------------------------------
    // Process final chunk.
    // ---------------------------------------------------------------------
    
    process_tail(m, mlen);
}

// ---------------------------------------------------------------------

void brwhash(const brwhash_ctx_t* ctx, 
             const __m128i* header, 
             const size_t hlen, 
             const __m128i* message, 
             const size_t mlen, 
             __m128i result[2])
{
    do_hash_ae(ctx->k1, ctx->k2, 
      (__m128i*)header, hlen, 
      (__m128i*)message, mlen, result);
}

// ---------------------------------------------------------------------

inline
static void precompute_powers(__m128i key[BRW_PRECOMPUTED_POWERS])
{
    for (size_t i = 1; i < BRW_PRECOMPUTED_POWERS; ++i) {
        __m128i* next_k = key + i;
        gf_square(key[i-1], next_k);
    }
}

// ---------------------------------------------------------------------

void brwhash_keysetup(brwhash_ctx_t* ctx, 
                      const __m128i key[2])
{
    storeu(ctx->k1, loadu(key));
    storeu(ctx->k2, loadu(key+1));
    precompute_powers(ctx->k1);
    precompute_powers(ctx->k2);
}
