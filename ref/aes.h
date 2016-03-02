#pragma once

// ---------------------------------------------------------------------

#include <stdint.h>

// ---------------------------------------------------------------------

void aes_encrypt_round(const uint8_t in[16], 
                       uint8_t out[16], 
                       const uint8_t subkey[16]);

// ---------------------------------------------------------------------

void aes_encrypt_last_round(const uint8_t in[16], 
                            uint8_t out[16], 
                            const uint8_t subkey[16]);

// ---------------------------------------------------------------------

void aes_encrypt_round_tweaked(const uint8_t in[16], 
                               uint8_t out[16], 
                               const uint8_t tweak[16], 
                               const uint8_t subkey[16]);

// ---------------------------------------------------------------------

void aes_encrypt_last_round_tweaked(const uint8_t in[16], 
                                    uint8_t out[16], 
                                    const uint8_t tweak[16], 
                                    const uint8_t subkey[16]);
