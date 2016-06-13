/**
 * AVX2 implementation of the Deterministic-Counter-in-Tweak 
 * deterministic authenticated encryption scheme DCT[H, Simpira, Pi[E]], where
 * - H denotes a double application of the BRW hash function under independent
 *   keys,
 * - CDMS denotes the Psi_3 construction by [CDMS'09].
 * - Pi denotes the CTRT mode of operation [JNP'15]
 * - E denotes the Deoxys-BC-128-128 tweakable block cipher [JNP'14]
 * Note: This version might be susceptible to side-channel attacks.
 * 
 * @author  Eik List
 * @created 2015-12-22
 * 
 * [Ber'07]  D. Bernstein: Polynomial evaluation and message 
 * authentication, 2007.
 * [GM'16]   S. Gueron and N. Mouha: Simpira: A Family of Efficient Permutations 
 * Using the AES Round Function, 2016.
 * [JNP'14]  J. Jean, I. Nikolic, and T. Peyrin. Tweaks and Keys for Block 
 * Ciphers: The TWEAKEY Framework, ASIACRYPT'2014.
 * [JNP'15]  J. Jean, I. Nikolic, and T. Peyrin. Counter-in-Tweak: Authenticated 
 * Encryption Modes for Tweakable Block Ciphers, ePrint, 2015.
*/
#ifdef DEBUG
  #include<stdio.h>
#endif
#include <emmintrin.h>
#include <immintrin.h> // AVX2 for blending 32-bit values
#include <smmintrin.h>
#include <tmmintrin.h> // SSSE3 for pshufb
#include <wmmintrin.h>
#include <stdint.h>
#include <string.h>
#include "brw256.h"
#include "dct.h"

// ---------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------

#define KEYGEN_TWEAK           _mm_setr_epi8(0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0)
#define H_PERMUTATION          _mm_setr_epi8(7,0,13,10, 11,4,1,14, 15,8,5,2, 3,12,9,6)
#define MSB_MASK               _mm_set1_epi8(0x80)
#define TRIVIAL_PERMUTATION    _mm_setr_epi8(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)
#define SIMPLY_1B              _mm_set1_epi8(0x1b)
#define KILL_SHIFT             _mm_set1_epi8(0xfe)
#define DOMAIN_MASK            _mm_setr_epi8(0x7f,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff)
#define DOMAIN_ENC             _mm_setr_epi8(0x80,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)

static const unsigned char RCON[17] = {
    0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 
    0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 
    0x72
};

// ---------------------------------------------------------------------
// Load, Store, Helpers
// ---------------------------------------------------------------------

#define loadu(p)         _mm_loadu_si128((__m128i*)p)
#define load(p)          _mm_load_si128((__m128i*)p)
#define storeu(p,x)      _mm_storeu_si128((__m128i*)p, x)
#define store(p,x)       _mm_store_si128((__m128i*)p, x)

#define zero             _mm_setzero_si128()
// _mm_setr_epi8(b0, ..., b15), with b0 the lowest, and b15 the highest byte.
#define one              _mm_setr_epi8(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1) 
#define eight            _mm_setr_epi8(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8) 
#define seight           _mm_setr_epi8(0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0) 
#define PERM_MASK        _mm_setr_epi8(0,1,2,3,4,5,6,7,15,14,13,12,11,10,9,8)

#define vaesenc(x,y)     _mm_aesenc_si128(x,y) 

#define vadd(x,y)        _mm_add_epi64(x,y)
#define vand(x,y)        _mm_and_si128(x,y)
#define vandnot(x,y)     _mm_andnot_si128(x,y)
#define vor(x,y)         _mm_or_si128(x,y)
#define vxor(x,y)        _mm_xor_si128(x,y)
#define vxor3(x,y,z)     _mm_xor_si128(x,_mm_xor_si128(y,z))

// ---------------------------------------------------------------------

static __m128i load_partial(const void *p, unsigned n)
{
    if (n == 0) {
        return zero;
    } else if (n % 16 == 0) {
        return _mm_loadu_si128((__m128i*)p);
    } else {
        __m128i tmp;
        unsigned i;

        for (i = 0; i < n; ++i) {
            ((char*)&tmp)[i] = ((char*)p)[i];
        }

        return tmp;
    }
}

// ---------------------------------------------------------------------

static void store_partial(const void *p, __m128i x, unsigned n)
{
    if (n == 0) {
        return;
    } else if (n >= BLOCKLEN) {
        storeu(p, x);
    } else {
        unsigned i;
        char* p_ = (char*)p;
        char* x_ = (char*)&x;

        for (i = 0; i < n; ++i) {
            p_[i] = x_[i];
        }
    }
}

// ---------------------------------------------------------------------
// Printing
// ---------------------------------------------------------------------

#ifdef DEBUG
static void print_128(const char* label, __m128i var)
{
    unsigned char val[BLOCKLEN];
    store((void*)val, var);
    printf("%s\n", label);
    int i;

    for (i = 0; i < BLOCKLEN; ++i) {
        printf("%02x ", val[i]);
    }

    puts("");
}

// ---------------------------------------------------------------------

static void print_hex_var(const char *label, const uint8_t *c, const int len)
{
    printf("%s: \n", label);
    int i;

    for (i = 0; i < len; i++) {
        printf("%02x ", c[i]);
    }

    puts("");
}
#endif

// ---------------------------------------------------------------------
// Utils
// ---------------------------------------------------------------------

static inline size_t ceil(const size_t a, const size_t b)
{
    if (a == 0 || b == 0) {
        return 0;
    }

    return ((a-1) / b) + 1;
}

// ---------------------------------------------------------------------
// Deoxys permutations
// ---------------------------------------------------------------------

#define permute(x)                       _mm_shuffle_epi8(x, PERM_MASK)
#define permute_tweak(x)                 _mm_shuffle_epi8(x, H_PERMUTATION)
#define set_domain_in_tweak(tweak,mask)  vor(vand(DOMAIN_MASK, tweak), mask)

#define add_to_tweak(t,x) do {        \
    t = permute(vadd(permute(t), x)); \
} while (0)

// y = permute(t, ( 0,1,2,3,4,5,6,7,  15,14,13,12,11,10,9,8 ));
// z = vadd   (y, ( 0,0,0,0, 0,0,0,0, 8,0,0,0, 0,0,0,0 ));
// t = permute(z, ( 0,1,2,3,4,5,6,7,  15,14,13,12,11,10,9,8 ));

// ---------------------------------------------------------------------
// Deoxys key/tweak setup
// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
// The bytes with their MSB (1*******) set must be XORed with 0x1b.
// Msbits = Those bytes with MSB set have 0x00, others 0x80.
// Multi_mask = Those bytes with MSB set have value <i>, others 0x8<i>.
// Only the to-be-XORed-with-0x1b-bytes are left after shuffle_epi8, others
// (with 1******* in multi_mask) are zeroized. 
// ---------------------------------------------------------------------

#define gf2_8_double_bytes(out, in) do {                         \
    __m128i msbits = vandnot(in, MSB_MASK);                      \
    __m128i multi_mask = vor(msbits, TRIVIAL_PERMUTATION);       \
    __m128i rot_cons = _mm_shuffle_epi8(SIMPLY_1B, multi_mask);  \
    __m128i tmp = _mm_slli_epi32(in, 1);                         \
    tmp = vand(tmp, KILL_SHIFT);                                 \
    out = vxor(tmp, rot_cons);                                   \
} while (0)

// ---------------------------------------------------------------------

#define tweakey_round(out, in) do {             \
    gf2_8_double_bytes(out, in);                \
    out = _mm_shuffle_epi8(out, H_PERMUTATION); \
} while (0)

// ---------------------------------------------------------------------

static void deoxys_keysetup(DEOXYS_KEY subkeys, const __m128i key)
{
    subkeys[0] = key;

    for (size_t i = 0; i < DEOXYS_ROUNDS; ++i) {
        tweakey_round(subkeys[i+1], subkeys[i]);
    }

    for (size_t i = 0; i <= DEOXYS_ROUNDS; ++i) {
        const __m128i rcon = _mm_setr_epi8(
            1,2,4,8,RCON[i],RCON[i],RCON[i],RCON[i],0,0,0,0, 0,0,0,0
        );
        subkeys[i] = vxor(subkeys[i], rcon);
    }
}

// ---------------------------------------------------------------------

static inline void prepare_tweak_counters(__m128i* tweak_ctrs) {
    __m128i tmp = one;

    for (size_t round = 0; round < DEOXYS_ROUND_KEYS; ++round) {
        tweak_ctrs[round*8]   = zero;
        tweak_ctrs[round*8+1] = vadd(tweak_ctrs[round*8+0], tmp);
        tweak_ctrs[round*8+2] = vadd(tweak_ctrs[round*8+1], tmp);
        tweak_ctrs[round*8+3] = vadd(tweak_ctrs[round*8+2], tmp);
        tweak_ctrs[round*8+4] = vadd(tweak_ctrs[round*8+3], tmp);
        tweak_ctrs[round*8+5] = vadd(tweak_ctrs[round*8+4], tmp);
        tweak_ctrs[round*8+6] = vadd(tweak_ctrs[round*8+5], tmp);
        tweak_ctrs[round*8+7] = vadd(tweak_ctrs[round*8+6], tmp);
        tmp = permute_tweak(tmp);
    }
}

// ---------------------------------------------------------------------

#define xor_eight(states, tmp, tweak_ctrs) do { \
    states[0] = vxor(tmp, tweak_ctrs[0]); \
    states[1] = vxor(tmp, tweak_ctrs[1]); \
    states[2] = vxor(tmp, tweak_ctrs[2]); \
    states[3] = vxor(tmp, tweak_ctrs[3]); \
    states[4] = vxor(tmp, tweak_ctrs[4]); \
    states[5] = vxor(tmp, tweak_ctrs[5]); \
    states[6] = vxor(tmp, tweak_ctrs[6]); \
    states[7] = vxor(tmp, tweak_ctrs[7]); \
} while (0)

// ---------------------------------------------------------------------

#define deoxys_enc_round_eight(states, tweak, tweak_ctrs, k) do { \
    tmp = vxor(tweak, k);                                         \
    states[0] = vaesenc(states[0], vxor(*(tweak_ctrs+0), tmp));   \
    states[1] = vaesenc(states[1], vxor(*(tweak_ctrs+1), tmp));   \
    states[2] = vaesenc(states[2], vxor(*(tweak_ctrs+2), tmp));   \
    states[3] = vaesenc(states[3], vxor(*(tweak_ctrs+3), tmp));   \
    states[4] = vaesenc(states[4], vxor(*(tweak_ctrs+4), tmp));   \
    states[5] = vaesenc(states[5], vxor(*(tweak_ctrs+5), tmp));   \
    states[6] = vaesenc(states[6], vxor(*(tweak_ctrs+6), tmp));   \
    states[7] = vaesenc(states[7], vxor(*(tweak_ctrs+7), tmp));   \
} while (0)

// ---------------------------------------------------------------------

#define deoxys_enc_eight(states, tweaks, tweak_ctrs, k, n) do {           \
    tmp = vxor(n, tweaks[0]);                                             \
    xor_eight(states, tmp, tweak_ctrs);                                   \
    deoxys_enc_round_eight(states, tweaks[1], tweak_ctrs+(1*8), k[1]);    \
    deoxys_enc_round_eight(states, tweaks[2], tweak_ctrs+(2*8), k[2]);    \
    deoxys_enc_round_eight(states, tweaks[3], tweak_ctrs+(3*8), k[3]);    \
    deoxys_enc_round_eight(states, tweaks[4], tweak_ctrs+(4*8), k[4]);    \
    deoxys_enc_round_eight(states, tweaks[5], tweak_ctrs+(5*8), k[5]);    \
    deoxys_enc_round_eight(states, tweaks[6], tweak_ctrs+(6*8), k[6]);    \
    deoxys_enc_round_eight(states, tweaks[7], tweak_ctrs+(7*8), k[7]);    \
    deoxys_enc_round_eight(states, tweaks[8], tweak_ctrs+(8*8), k[8]);    \
    deoxys_enc_round_eight(states, tweaks[9], tweak_ctrs+(9*8), k[9]);    \
    deoxys_enc_round_eight(states, tweaks[10], tweak_ctrs+(10*8), k[10]); \
    deoxys_enc_round_eight(states, tweaks[11], tweak_ctrs+(11*8), k[11]); \
    deoxys_enc_round_eight(states, tweaks[12], tweak_ctrs+(12*8), k[12]); \
    deoxys_enc_round_eight(states, tweaks[13], tweak_ctrs+(13*8), k[13]); \
    deoxys_enc_round_eight(states, tweaks[14], tweak_ctrs+(14*8), k[14]); \
} while (0)

// ---------------------------------------------------------------------

#define load_xor_and_store_eight(out, in, states) do { \
    out[0] = vxor(states[0], loadu(in  )); \
    out[1] = vxor(states[1], loadu(in+1)); \
    out[2] = vxor(states[2], loadu(in+2)); \
    out[3] = vxor(states[3], loadu(in+3)); \
    out[4] = vxor(states[4], loadu(in+4)); \
    out[5] = vxor(states[5], loadu(in+5)); \
    out[6] = vxor(states[6], loadu(in+6)); \
    out[7] = vxor(states[7], loadu(in+7)); \
} while (0)

// ---------------------------------------------------------------------

#define deoxys_enc_round(state, tweak, tweak_ctr, k) do { \
    state = vaesenc(state, vxor3(tweak, tweak_ctr, k));   \
    tweak = permute_tweak(tweak);                         \
} while (0)

// ---------------------------------------------------------------------

#define deoxys_enc(state, tweak, tweak_ctrs, k) do {            \
    state = vxor3(state, tweak, *(tweak_ctrs));                 \
    tweak = permute_tweak(tweak);                               \
    deoxys_enc_round(state, tweak, *(tweak_ctrs+1*8), k[1]);    \
    deoxys_enc_round(state, tweak, *(tweak_ctrs+2*8), k[2]);    \
    deoxys_enc_round(state, tweak, *(tweak_ctrs+3*8), k[3]);    \
    deoxys_enc_round(state, tweak, *(tweak_ctrs+4*8), k[4]);    \
    deoxys_enc_round(state, tweak, *(tweak_ctrs+5*8), k[5]);    \
    deoxys_enc_round(state, tweak, *(tweak_ctrs+6*8), k[6]);    \
    deoxys_enc_round(state, tweak, *(tweak_ctrs+7*8), k[7]);    \
    deoxys_enc_round(state, tweak, *(tweak_ctrs+8*8), k[8]);    \
    deoxys_enc_round(state, tweak, *(tweak_ctrs+9*8), k[9]);    \
    deoxys_enc_round(state, tweak, *(tweak_ctrs+10*8), k[10]);  \
    deoxys_enc_round(state, tweak, *(tweak_ctrs+11*8), k[11]);  \
    deoxys_enc_round(state, tweak, *(tweak_ctrs+12*8), k[12]);  \
    deoxys_enc_round(state, tweak, *(tweak_ctrs+13*8), k[13]);  \
    deoxys_enc_round(state, tweak, *(tweak_ctrs+14*8), k[14]);  \
} while (0)

// ---------------------------------------------------------------------

static inline void load_xor_store_n(__m128i* out, 
                                    const __m128i* in, 
                                    const __m128i* states, 
                                    const size_t num_blocks) 
{
    for (size_t i = 0; i < num_blocks; ++i) {
        out[i] = vxor(states[i], loadu(in+i));
    }
}

// ---------------------------------------------------------------------

static inline void deoxys_enc_n(__m128i* states, 
                                const __m128i tweak, 
                                const __m128i* tweak_ctrs, 
                                const __m128i* k, 
                                const int num_blocks, 
                                const __m128i n)
{
    int i, j;
    __m128i tmp = vxor(n, tweak);
    __m128i tmp_tweak = tweak;

    for(i = 0; i < num_blocks; i++) {
        states[i] = vxor(tmp, tweak_ctrs[i]);
    }

    tmp_tweak = permute_tweak(tmp_tweak);

    for(j = 1; j < DEOXYS_ROUND_KEYS; j++) {
        tmp = vxor(tmp_tweak, k[j]);

        for(i = 0; i< num_blocks; i++) {
            states[i] = vaesenc(states[i], vxor(tweak_ctrs[j*8+i], tmp));
        }

        tmp_tweak = permute_tweak(tmp_tweak);
    }
}

// ---------------------------------------------------------------------

#define deoxys_enc_single_round(state, tweak, k) do { \
    state = vaesenc(state, vxor(tweak, k));           \
    tweak = permute_tweak(tweak);                     \
} while (0)

// ---------------------------------------------------------------------

#define deoxys_encrypt(state, tweak, k) do {    \
    state = vxor3(state, tweak, k[0]);             \
    tweak = permute_tweak(tweak);                  \
    deoxys_enc_single_round(state, tweak, k[1]);   \
    deoxys_enc_single_round(state, tweak, k[2]);   \
    deoxys_enc_single_round(state, tweak, k[3]);   \
    deoxys_enc_single_round(state, tweak, k[4]);   \
    deoxys_enc_single_round(state, tweak, k[5]);   \
    deoxys_enc_single_round(state, tweak, k[6]);   \
    deoxys_enc_single_round(state, tweak, k[7]);   \
    deoxys_enc_single_round(state, tweak, k[8]);   \
    deoxys_enc_single_round(state, tweak, k[9]);   \
    deoxys_enc_single_round(state, tweak, k[10]);  \
    deoxys_enc_single_round(state, tweak, k[11]);  \
    deoxys_enc_single_round(state, tweak, k[12]);  \
    deoxys_enc_single_round(state, tweak, k[13]);  \
    deoxys_enc_single_round(state, tweak, k[14]);  \
} while (0)

// ---------------------------------------------------------------------
// Simpira v2
// ---------------------------------------------------------------------

#define simpira_const(c,b)  _mm_setr_epi32(c^b, c^b^0x10, c^b^0x20, c^b^0x30)

static void simpira_decrypt(__m128i k, const __m128i* c, __m128i* m) 
{
    __m128i l = vxor(c[0], k);
    __m128i r = c[1];
    r = vaesenc(vaesenc(l, simpira_const(15,2)), r);
    l = vaesenc(vaesenc(r, simpira_const(14,2)), l);
    r = vaesenc(vaesenc(l, simpira_const(13,2)), r);
    l = vaesenc(vaesenc(r, simpira_const(12,2)), l);
    r = vaesenc(vaesenc(l, simpira_const(11,2)), r);
    l = vaesenc(vaesenc(r, simpira_const(10,2)), l);
    r = vaesenc(vaesenc(l, simpira_const(9,2)), r);
    l = vaesenc(vaesenc(r, simpira_const(8,2)), l);
    r = vaesenc(vaesenc(l, simpira_const(7,2)), r);
    l = vaesenc(vaesenc(r, simpira_const(6,2)), l);
    r = vaesenc(vaesenc(l, simpira_const(5,2)), r);
    l = vaesenc(vaesenc(r, simpira_const(4,2)), l);
    r = vaesenc(vaesenc(l, simpira_const(3,2)), r);
    l = vaesenc(vaesenc(r, simpira_const(2,2)), l);
    r = vaesenc(vaesenc(l, simpira_const(1,2)), r);
    m[0] = vxor(l, k);
    m[1] = r;
}

// ---------------------------------------------------------------------

static void simpira_encrypt(const __m128i k, const __m128i* m, __m128i* c) 
{
    __m128i l = vxor(m[0], k);
    __m128i r = m[1];
    r = vaesenc(vaesenc(l, simpira_const(1,2)), r);
    l = vaesenc(vaesenc(r, simpira_const(2,2)), l);
    r = vaesenc(vaesenc(l, simpira_const(3,2)), r);
    l = vaesenc(vaesenc(r, simpira_const(4,2)), l);
    r = vaesenc(vaesenc(l, simpira_const(5,2)), r);
    l = vaesenc(vaesenc(r, simpira_const(6,2)), l);
    r = vaesenc(vaesenc(l, simpira_const(7,2)), r);
    l = vaesenc(vaesenc(r, simpira_const(8,2)), l);
    r = vaesenc(vaesenc(l, simpira_const(9,2)), r);
    l = vaesenc(vaesenc(r, simpira_const(10,2)), l);
    r = vaesenc(vaesenc(l, simpira_const(11,2)), r);
    l = vaesenc(vaesenc(r, simpira_const(12,2)), l);
    r = vaesenc(vaesenc(l, simpira_const(13,2)), r);
    l = vaesenc(vaesenc(r, simpira_const(14,2)), l);
    r = vaesenc(vaesenc(l, simpira_const(15,2)), r);
    c[0] = vxor(l, k);
    c[1] = r;
}

// ---------------------------------------------------------------------
// CTRT
// ---------------------------------------------------------------------

static inline void ctrt_mode(const DEOXYS_KEY k, 
                             const __m128i iv[2],
                             const __m128i* in,
                             uint64_t len, 
                             __m128i* out)
{
    // ---------------------------------------------------------------------
    // The nonce serves as input to each call of the block cipher.
    // ---------------------------------------------------------------------
    
    const __m128i n = vxor(loadu(iv), k[0]);
    
    // ---------------------------------------------------------------------
    // We use r+1 tweaks to store the tweaks t_0, t_1, ..., t_r for one block
    // for r rounds:
    // tweak_ctr[r][i] = pi^{r}(i)
    // tweak_ctr[r][0] = pi^{r}(0) = 0
    // In each round, we then simply have to have the subtweakey:
    // K[r] xor pi^r(T) xor pi^{r}(i)
    // ---------------------------------------------------------------------

    __m128i tweak_ctrs[DEOXYS_ROUND_KEYS*8];
    prepare_tweak_counters(tweak_ctrs);
    
    // ---------------------------------------------------------------------
    // T, the initial tweak. We encode the domain into the least significant
    // bit: tweak = (1 || tag).
    // ---------------------------------------------------------------------
    
    const __m128i initial_tweak = set_domain_in_tweak(iv[1], DOMAIN_ENC);
    __m128i tweak_ctr_base = zero;
    __m128i states[8];
    __m128i tmp;
    uint64_t j = 0;

    __m128i tweaks[15];
    
    while (len >= 8*BLOCKLEN) {
        tweaks[0] = vxor(initial_tweak, tweak_ctr_base);

        for (size_t i = 1; i < 8; ++i) {
            tweaks[i] = permute_tweak(tweaks[i-1]);
        }
        for (size_t i = 8; i < 15; ++i) {
            tweaks[i] = tweaks[i-8];
        }

        deoxys_enc_eight(states, tweaks, tweak_ctrs, k, n);
        load_xor_and_store_eight(out, in, states);

        len -= 8*BLOCKLEN;
        in += 8;
        out += 8;
        j += 8;

        // ---------------------------------------------------------------------
        // Every 256-th block, we have an overflow in the first byte and 
        // have to update the next highest bytes in the counter. 
        // ---------------------------------------------------------------------

        if ((j & 0xFF) == 0) { 
            add_to_tweak(tweak_ctr_base, seight);
        } else { // No overflow, increment only the lowest byte in the counter.
            tweak_ctr_base = vadd(tweak_ctr_base, eight);
        }
    }

    tweaks[0] = vxor(initial_tweak, tweak_ctr_base);
    
    const size_t ceiled_num_blocks = ceil(len, BLOCKLEN);
    const size_t num_blocks = len / BLOCKLEN;
    const size_t last_block = len % BLOCKLEN;

    deoxys_enc_n(states, tweaks[0], tweak_ctrs, k, ceiled_num_blocks, n);
    load_xor_store_n(out, in, states, num_blocks);

    if (last_block != 0) {    
        in += num_blocks;
        out += num_blocks;
        store_partial(out, 
            vxor(states[num_blocks], load_partial(in, last_block)), last_block);
    }
}

// ---------------------------------------------------------------------
// Encoding/Decoding
// ---------------------------------------------------------------------

/**
 * Assumes that m_left provides SIMPIRA_BLOCKLEN - TAGLEN of space.
 * Will only SET m_right to the start byte inside m.
 */
static void encode(const __m128i* m, 
                   const size_t mlen, 
                   __m128i* m_left, 
                   __m128i** m_right, 
                   size_t* m_left_len, 
                   size_t* m_right_len)
{
    const size_t m_left_start = SIMPIRA_BLOCKLEN - TAGLEN;

    // M_L
    m_left[0] = zero;
    m_left[1] = m[0];
    *m_left_len = SIMPIRA_BLOCKLEN;

    // M_R
    *m_right = (__m128i*)(m + 1);
    *m_right_len = mlen - m_left_start;
}

// ---------------------------------------------------------------------

/**
 * Will 
 * - copy m_left to the start of message, 
 * - extract the redundancy from m_left and verify it.
 * If the redundancy valid, sets the correct message length.
 * Otherwise, it zeroizes the complete message. 
 * Assumes that m_right points to the correct position inside the message.
 */
static int decode(const __m128i* m_left,
                  const size_t c_right_len,
                  __m128i* message, 
                  size_t* mlen)
{
    const size_t m_left_start = SIMPIRA_BLOCKLEN - TAGLEN;
    message[0] = m_left[1];
    *mlen = m_left_start + c_right_len;
    return _mm_testc_si128(m_left[0], zero) - 1;
}

// ---------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------

static void encrypt_long(dct_context_t* ctx, 
                         __m128i* c, size_t* clen, 
                         const __m128i* h, const size_t hlen,
                         const __m128i* m, const size_t mlen)
{
    // (M_L, M_R) = Encode(M)
    __m128i m_left[SIMPIRA_BLOCKS];
    __m128i* m_right;
    size_t m_left_len, m_right_len;
    encode(m, mlen, m_left, &m_right, &m_left_len, &m_right_len);

    // X = BRW(A, M_R)
    __m128i x[SIMPIRA_BLOCKS];
    brwhash(&ctx->hash_ctx, h, hlen, m_right, m_right_len, x);

    // Y = M_L xor X
    __m128i y[SIMPIRA_BLOCKS];
    y[0] = vxor(x[0], m_left[0]);
    y[1] = vxor(x[1], m_left[1]);

    __m128i* c_left = c;
    __m128i* c_right = c + SIMPIRA_BLOCKS;

    // C_L = Simpira(simpira_key, Y)
    simpira_encrypt(ctx->simpira_key, y, c_left);
    
    // C_R = CTRT(DEOXYS_KEY, C_L, M_R)
    ctrt_mode(ctx->expanded_key, c_left, m_right, m_right_len, c_right);
    
    // Set clen
    *clen = SIMPIRA_BLOCKLEN + m_right_len;
}

// ---------------------------------------------------------------------

static int decrypt_long(dct_context_t* ctx, 
                        __m128i* m, size_t* mlen, 
                        const __m128i* h, const size_t hlen,
                        const __m128i* c, const size_t clen)
{
    // (C_L, C_R) = C
    __m128i* c_left = (__m128i*)c;
    __m128i* c_right = (__m128i*)(c + SIMPIRA_BLOCKS);
    const size_t c_right_len = clen - SIMPIRA_BLOCKLEN;

    // M_R = CTRT^{-1}(DEOXYS_KEY, C_L)
    __m128i* m_right = m + 1;
    ctrt_mode(ctx->expanded_key, c_left, c_right, c_right_len, m_right);

    // X = BRW(A, M_R)
    __m128i x[SIMPIRA_BLOCKS];
    brwhash(&ctx->hash_ctx, h, hlen, m_right, c_right_len, x);

    // Y = Simpira^{-1}(simpira_key, C_L)
    __m128i y[SIMPIRA_BLOCKS];
    simpira_decrypt(ctx->simpira_key, c_left, y);

    // M_L = X xor Y
    x[0] = vxor(x[0], y[0]);
    x[1] = vxor(x[1], y[1]);

    // Decode(M_L)
    return decode(x, c_right_len, m, mlen);
}

// ---------------------------------------------------------------------
// API
// ---------------------------------------------------------------------

void keysetup(dct_context_t* ctx, const unsigned char key[KEYLEN])
{
    __m128i output[4];

    deoxys_keysetup(ctx->expanded_key, loadu(key));

    __m128i tweak;

    for (size_t i = 0; i < 4; ++i) {
        output[i] = _mm_set_epi8(i,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
        tweak = KEYGEN_TWEAK;
        deoxys_encrypt(output[i], tweak, ctx->expanded_key);
    }

    brwhash_keysetup(&ctx->hash_ctx, output);
    deoxys_keysetup(ctx->expanded_key, output[2]);
    store(&ctx->simpira_key, output[3]);
}

// ---------------------------------------------------------------------

void encrypt_final(dct_context_t* ctx, 
                   unsigned char* c, size_t* clen, 
                   const unsigned char* h, const size_t hlen,
                   const unsigned char* m, const size_t mlen)
{
    if (mlen < MIN_MESSAGE_LEN) {
        // TODO: encrypt_small(ctx, (__m128i*)c, clen, (const __m128i*)h, hlen, 
        //           (const __m128i*)m, mlen);
    } else {
        encrypt_long(ctx, (__m128i*)c, clen, (const __m128i*)h, hlen, 
            (const __m128i*)m, mlen);
    }
}

// ---------------------------------------------------------------------

int decrypt_final(dct_context_t* ctx, 
                  unsigned char* m, size_t* mlen, 
                  const unsigned char* h, const size_t hlen,
                  const unsigned char* c, const size_t clen)
{
    if (clen < SIMPIRA_BLOCKLEN) {
        return -1;
        // return decrypt_small(ctx, (__m128i*)m, mlen, (const __m128i*)h, hlen, 
        //     (const __m128i*)c, clen);
    } else {
        return decrypt_long(ctx, (__m128i*)m, mlen, (const __m128i*)h, hlen, 
            (const __m128i*)c, clen);
    }
}
