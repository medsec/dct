#include "api.h"
#include "dct.h"

// ---------------------------------------------------------------------
// SUPERCOP API
// ---------------------------------------------------------------------

void crypto_aead_encrypt(unsigned char *c, size_t *clen,
                         const unsigned char *h, size_t hlen,
                         const unsigned char *m, size_t mlen,
                         const unsigned char *nonce,
                         const unsigned char *key)
{
    dct_context_t ctx;
    keysetup(&ctx, key);
    (void)nonce;
    encrypt_final(&ctx, c, clen, h, hlen, m, mlen);
}

// ---------------------------------------------------------------------

int crypto_aead_decrypt(unsigned char *m, size_t *mlen,
                        const unsigned char *h, size_t hlen,
                        const unsigned char *c, size_t clen,
                        const unsigned char *nonce,
                        const unsigned char *key)
{   
    dct_context_t ctx;
    keysetup(&ctx, key);
    (void)nonce;
    return decrypt_final(&ctx, m, mlen, h, hlen, c, clen);
}
