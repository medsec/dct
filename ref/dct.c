/**
 * Reference implementation of the Deterministic-Counter-in-Tweak 
 * deterministic authenticated encryption scheme DCT[H, Simpira, Pi[E]], where
 * - H denotes a double application of the BRW hash function under independent
 *   keys,
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
#include <stdint.h>
#include <string.h>
#include "brw256.h"
#include "dct.h"

// ---------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------

static const block KEYGEN_TWEAK = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
static const block H_PERMUTATION = { 
    7,0,13,10, 11,4,1,14, 15,8,5,2, 3,12,9,6 
};
static const block DOMAIN_MASK = { 
    0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff 
};
static const block DOMAIN_ENC = { 
    0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};
static const unsigned char RCON[17] = {
    0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 
    0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 
    0x72
};

// ---------------------------------------------------------------------
// Printing
// ---------------------------------------------------------------------

#ifdef DEBUG
static inline void print_hex_var(const char *label, 
                                 const uint8_t *x, 
                                 const int len)
{
    printf("%s: \n", label);
    int i;

    for (i = 0; i < len; i++) {
        if ((i != 0) && (i % 16 == 0)) {
            puts("");
        }

        printf("%02x ", x[i]);
    }

    puts("");
}

// ---------------------------------------------------------------------

static inline void print_hex(const char* label, const uint8_t *c)
{
    print_hex_var(label, c, BLOCKLEN);
}
#endif

// ---------------------------------------------------------------------
// Utils
// ---------------------------------------------------------------------

static int compare(const uint8_t* a, 
                   const uint8_t* b, 
                   const size_t num_bytes)
{
    uint8_t result = 0;
    
    for (size_t i = 0; i < num_bytes; i++) {
        result |= a[i] ^ b[i];
    }
    
    return result;
}

// ---------------------------------------------------------------------

static void mask_plaintext(uint8_t* message, 
                           const size_t mlen, 
                           const uint8_t mask)
{
    for (size_t i = 0; i < mlen; ++i) {
        message[i] &= mask;
    }
}

// ---------------------------------------------------------------------

static inline void to_be_array(uint8_t* result, const uint64_t src)
{
    for (size_t i = 0; i < 8; ++i) {
        result[7-i] = (src >> (8*i)) & 0xFF;
    }
}

// ---------------------------------------------------------------------

// static inline void to_array(uint8_t* result, const uint64_t src)
// {
//     for (size_t i = 0; i < 8; ++i) {
//         result[i] = (src >> (8*i)) & 0xFF;
//     }
// }

// ---------------------------------------------------------------------

static inline void vand(block out, const block a, const block b)
{
    for(size_t i = 0; i < BLOCKLEN; ++i) {
        out[i] = a[i] & b[i];
    }
}

// ---------------------------------------------------------------------

static inline void vor(block out, const block a, const block b)
{
    for(size_t i = 0; i < BLOCKLEN; ++i) {
        out[i] = a[i] | b[i];
    }
}

// ---------------------------------------------------------------------

static inline void vxor(uint8_t* out, const uint8_t* a, const uint8_t* b, 
                        const size_t len)
{
    for(size_t i = 0; i < len; ++i) {
        out[i] = a[i] ^ b[i];
    }
}

// ---------------------------------------------------------------------

static inline void vxor_block(block out, const block a, const block b)
{
    vxor(out, a, b, BLOCKLEN);
}

// ---------------------------------------------------------------------

static inline void zeroize(block x, const size_t len)
{
    memset(x, 0x00, len);
}

// ---------------------------------------------------------------------

static inline void zeroize_block(block x)
{
    zeroize(x, BLOCKLEN);
}

// ---------------------------------------------------------------------
// Deoxys permutations
// ---------------------------------------------------------------------

static inline void permute(block x, const block mask) 
{
    block tmp;

    for (size_t i = 0; i < BLOCKLEN; ++i) {
        tmp[i] = x[mask[i]];
    }

    memcpy(x, tmp, BLOCKLEN);
}

// ---------------------------------------------------------------------

static inline void permute_tweak(block x)
{
    permute(x, H_PERMUTATION);
}

// ---------------------------------------------------------------------
// Deoxys key/tweak setup
// ---------------------------------------------------------------------

static inline void gf2_8_double_bytes(block out, const block in)
{
    for (int i = 0; i < BLOCKLEN; ++i) {
        const uint8_t msb = (in[i] & 0x80) >> 7;
        out[i] = (in[i] << 1) ^ (msb * 0x1b);
    }
}

// ---------------------------------------------------------------------

static inline void tweakey_round(block out, const block in) 
{
    gf2_8_double_bytes(out, in);
    permute_tweak(out);
}

// ---------------------------------------------------------------------

static void deoxys_keysetup(DEOXYS_KEY subkeys, const uint8_t key[BLOCKLEN])
{
    memcpy(subkeys, key, BLOCKLEN);

    for (size_t i = 0; i < DEOXYS_ROUNDS; ++i) {
        tweakey_round(subkeys + (i+1)*BLOCKLEN, subkeys + i*BLOCKLEN);
    }

    for (size_t i = 0; i <= DEOXYS_ROUNDS; ++i) {
        const block rcon = {
            1,2,4,8,RCON[i],RCON[i],RCON[i],RCON[i],0,0,0,0, 0,0,0,0
        };
        vxor_block(subkeys + i*BLOCKLEN, subkeys + i*BLOCKLEN, rcon);
    }
}

// ---------------------------------------------------------------------

static void deoxys_encrypt(const DEOXYS_KEY key, 
                           const block input, 
                           block tweak, 
                           block output)
{
    block state;
    vxor_block(state, input, key);
    vxor_block(state, state, tweak);
    permute_tweak(tweak);

    for(size_t i = 1; i < DEOXYS_ROUNDS; ++i) {
        aes_encrypt_round_tweaked(state, state, tweak, key + i*BLOCKLEN);
        permute_tweak(tweak);
    }

    aes_encrypt_round_tweaked(state, output, tweak, key + DEOXYS_ROUNDS*BLOCKLEN);
}

// ---------------------------------------------------------------------
// Simpira
// ---------------------------------------------------------------------

static void simpira_set_const(block round_const, const size_t c, const size_t b)
{
    round_const[0] = c;
    round_const[4] = b;
}

// ---------------------------------------------------------------------

static void simpira_decrypt(const uint8_t k[BLOCKLEN], 
                            const uint8_t c[SIMPIRA_BLOCKLEN], 
                            uint8_t m[SIMPIRA_BLOCKLEN])
{
    uint8_t l[BLOCKLEN];
    uint8_t r[BLOCKLEN];
    uint8_t s[BLOCKLEN];
    uint8_t round_const[BLOCKLEN];
    memset(round_const, 0x00, BLOCKLEN);

    vxor_block(l, c, k);
    memcpy(r, c + BLOCKLEN, BLOCKLEN);

    simpira_set_const(round_const, SIMPIRA_ROUNDS, 2);
    aes_encrypt_round(l, s, round_const);
    aes_encrypt_round(s, r, r);

    for (int i = SIMPIRA_ROUNDS-1; i >= 1; i -= 2) {
        simpira_set_const(round_const, i, 2);
        aes_encrypt_round(r, s, round_const);
        aes_encrypt_round(s, l, l);

        simpira_set_const(round_const, i-1, 2);
        aes_encrypt_round(l, s, round_const);
        aes_encrypt_round(s, r, r);
    }

    vxor_block(m, l, k);
    memcpy(m + BLOCKLEN, r, BLOCKLEN);
}

// ---------------------------------------------------------------------

static void simpira_encrypt(const uint8_t k[BLOCKLEN], 
                            const uint8_t m[SIMPIRA_BLOCKLEN], 
                            uint8_t c[SIMPIRA_BLOCKLEN])
{
    uint8_t l[BLOCKLEN];
    uint8_t r[BLOCKLEN];
    uint8_t s[BLOCKLEN];
    uint8_t round_const[BLOCKLEN];
    memset(round_const, 0x00, BLOCKLEN);

    vxor_block(l, m, k);
    memcpy(r, m + BLOCKLEN, BLOCKLEN);

    for (int i = 1; i < SIMPIRA_ROUNDS; i += 2) {
        simpira_set_const(round_const, i, 2);
        aes_encrypt_round(l, s, round_const);
        aes_encrypt_round(s, r, r);

        simpira_set_const(round_const, i+1, 2);
        aes_encrypt_round(r, s, round_const);
        aes_encrypt_round(s, l, l);
    }

    simpira_set_const(round_const, SIMPIRA_ROUNDS, 2);
    aes_encrypt_round(l, s, round_const);
    aes_encrypt_round(s, r, r);

    vxor_block(c, l, k);
    memcpy(c + BLOCKLEN, r, BLOCKLEN);
}

// ---------------------------------------------------------------------
// CTRT
// ---------------------------------------------------------------------

static inline void set_domain_in_tweak(block x, const block mask)
{
    vand(x, x, DOMAIN_MASK);
    vor(x, x, mask);
}

// ---------------------------------------------------------------------

static inline void xor_with_counter(block out, 
                                    const block in, 
                                    const uint64_t counter)
{
    memcpy(out, in, 8);
    to_be_array(out+8, counter);
    vxor(out+8, out+8, in+8, 8);
}

// ---------------------------------------------------------------------

static inline void ctrt_mode(const DEOXYS_KEY key, 
                             const uint8_t iv[DEOXYS_IVLEN],
                             const uint8_t* in,
                             uint64_t len, 
                             uint8_t* out)
{
    uint64_t counter = 0L;

    block cipher_input;
    memcpy(cipher_input, iv, BLOCKLEN);

    block tweak;
    memcpy(tweak, iv+BLOCKLEN, BLOCKLEN);

    block current_tweak;
    block cipher_output;

    while(len >= BLOCKLEN) {
        xor_with_counter(current_tweak, tweak, counter);
        set_domain_in_tweak(current_tweak, DOMAIN_ENC);
        deoxys_encrypt(key, cipher_input, current_tweak, cipher_output);
        vxor_block(out, cipher_output, in);
        in += BLOCKLEN;
        out += BLOCKLEN;
        len -= BLOCKLEN;
        counter++;
    }
    
    if (len > 0) {
        xor_with_counter(current_tweak, tweak, counter);
        set_domain_in_tweak(current_tweak, DOMAIN_ENC);
        deoxys_encrypt(key, cipher_input, current_tweak, cipher_output);
        vxor(out, cipher_output, in, len);
    }
}

// ---------------------------------------------------------------------
// Encoding/Decoding
// ---------------------------------------------------------------------

/**
 * Assumes that m_left provides SIMPIRA_BLOCKLEN - TAGLEN of space.
 * Will only SET m_right to the start byte inside m.
 */
static void encode(const uint8_t* m, 
                   const size_t mlen, 
                   uint8_t* m_left, 
                   uint8_t** m_right, 
                   size_t* m_left_len, 
                   size_t* m_right_len)
{
    const size_t m_left_start = SIMPIRA_BLOCKLEN - TAGLEN;

    // M_L
    memset(m_left, 0x00, TAGLEN);
    memcpy(m_left + TAGLEN, m, m_left_start);
    *m_left_len = SIMPIRA_BLOCKLEN;

    // M_R
    *m_right = (uint8_t*)(m + m_left_start);
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
static int decode(const uint8_t* m_left,
                  const size_t c_right_len,
                  uint8_t* message, 
                  size_t* mlen)
{
    const size_t m_left_start = SIMPIRA_BLOCKLEN - TAGLEN;
    memcpy(message, m_left + TAGLEN, m_left_start);
    *mlen = m_left_start + c_right_len;

    uint8_t expected_redundancy[TAGLEN];
    zeroize(expected_redundancy, TAGLEN);

    const int is_invalid = compare(m_left, expected_redundancy, TAGLEN);
    const uint8_t mask = is_invalid ? 0x00 : 0xFF;
    mask_plaintext(message, *mlen, mask);
    return is_invalid;
}

// ---------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------

static void encrypt_long(dct_context_t* ctx, 
                         uint8_t* c, size_t* clen, 
                         const uint8_t* h, const size_t hlen,
                         const uint8_t* m, const size_t mlen)
{
    // (M_L, M_R) = Encode(M)
    uint8_t m_left[SIMPIRA_BLOCKLEN];
    uint8_t* m_right;
    size_t m_left_len, m_right_len;
    encode(m, mlen, m_left, &m_right, &m_left_len, &m_right_len);

    #ifdef DEBUG
    puts("ENCRYPT");
    printf("|M_L|: %zu, |M_R|: %zu\n", m_left_len, m_right_len);
    print_hex_var("M_L", m_left, m_left_len);
    print_hex_var("M_R", m_right, m_right_len);
    #endif

    // X = BRW(A, M_R)
    uint8_t x[BRW_HASHLEN];
    brwhash(&ctx->hash_ctx, h, hlen, m_right, m_right_len, x);

    // Y = M_L xor X
    uint8_t y[SIMPIRA_BLOCKLEN];
    vxor(y, x, m_left, SIMPIRA_BLOCKLEN);

    #ifdef DEBUG
    print_hex_var("X", x, BRW_HASHLEN);
    print_hex_var("Y", y, SIMPIRA_BLOCKLEN);
    #endif

    uint8_t* c_left = c;
    uint8_t* c_right = c + SIMPIRA_BLOCKLEN;
    
    // C_L = Simpira(simpira_key, Y)
    simpira_encrypt(ctx->simpira_key, y, c_left);

    // C_R = CTRT(DEOXYS_KEY, C_L, M_R)
    ctrt_mode(ctx->expanded_key, c_left, m_right, m_right_len, c_right);
    
    // Set clen
    *clen = SIMPIRA_BLOCKLEN + m_right_len;

    #ifdef DEBUG
    print_hex_var("C_L", c_left, SIMPIRA_BLOCKLEN);
    print_hex_var("C_R", c_right, m_right_len);
    printf("|C|: %zu\n", *clen);
    #endif
}

// ---------------------------------------------------------------------

static int decrypt_long(dct_context_t* ctx, 
                        uint8_t* m, size_t* mlen, 
                        const uint8_t* h, const size_t hlen,
                        const uint8_t* c, const size_t clen)
{
    // (C_L, C_R) = C
    uint8_t* c_left = (uint8_t*)c;
    uint8_t* c_right = (uint8_t*)(c + SIMPIRA_BLOCKLEN);
    const size_t c_right_len = clen - SIMPIRA_BLOCKLEN;

    #ifdef DEBUG
    puts("DECRYPT");
    printf("|C|: %zu, |C_R|: %zu\n", clen, c_right_len);
    print_hex_var("C_L", c_left, SIMPIRA_BLOCKLEN);
    print_hex_var("C_R", c_right, c_right_len);
    #endif

    // M_R = CTRT^{-1}(DEOXYS_KEY, C_L)
    const size_t m_left_start = SIMPIRA_BLOCKLEN - TAGLEN;
    uint8_t* m_right = m + m_left_start;
    ctrt_mode(ctx->expanded_key, c_left, c_right, c_right_len, m_right);

    // X = BRW(A, M_R)
    uint8_t x[BRW_HASHLEN];
    brwhash(&ctx->hash_ctx, h, hlen, m_right, c_right_len, x);

    // Y = Simpira^{-1}(simpira_key, C_L)
    uint8_t y[SIMPIRA_BLOCKLEN];
    simpira_decrypt(ctx->simpira_key, c_left, y);

    #ifdef DEBUG
    print_hex_var("X", x, BRW_HASHLEN);
    print_hex_var("Y", y, SIMPIRA_BLOCKLEN);
    #endif

    // M_L = X xor Y
    vxor(x, x, y, SIMPIRA_BLOCKLEN);

    #ifdef DEBUG
    print_hex_var("M_L", x, SIMPIRA_BLOCKLEN);
    print_hex_var("M_R", m_right, c_right_len);
    #endif

    // Decode(M_L)
    return decode(x, c_right_len, m, mlen);
}

// ---------------------------------------------------------------------
// API
// ---------------------------------------------------------------------

void keysetup(dct_context_t* ctx, const unsigned char key[KEYLEN])
{
    deoxys_keysetup(ctx->expanded_key, key);

    block ctr;
    uint8_t output[4*BLOCKLEN];
    zeroize_block(ctr);

    block tweak;

    for (size_t i = 0; i < 4; ++i) {
        ctr[BLOCKLEN-1] = i;
        memcpy(tweak, KEYGEN_TWEAK, BLOCKLEN);
        deoxys_encrypt(ctx->expanded_key, ctr, tweak, 
            output + (i * BLOCKLEN));
    }

    // Use (K1,K2) for hashing, K3 for Deoxys, and K4 for Simpira
    brwhash_keysetup(&ctx->hash_ctx, output); 
    deoxys_keysetup(ctx->expanded_key, output + 2*BLOCKLEN);
    memcpy(ctx->simpira_key, output + 3*BLOCKLEN, BLOCKLEN);
}

// ---------------------------------------------------------------------

void encrypt_final(dct_context_t* ctx, 
                   unsigned char* c, size_t* clen, 
                   const unsigned char* h, const size_t hlen,
                   const unsigned char* m, const size_t mlen)
{
    if (mlen < MIN_MESSAGE_LEN) {
        // TODO: encrypt_small(ctx, c, clen, h, hlen, m, mlen);
    } else {
        encrypt_long(ctx, c, clen, h, hlen, m, mlen);
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
        // TODO: return decrypt_small(ctx, m, mlen, h, hlen, c, clen);
    } else {
        return decrypt_long(ctx, m, mlen, h, hlen, c, clen);
    }
}
