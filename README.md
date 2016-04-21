# Deterministic Counter in Tweak (DCT)

DCT is a beyond-birthday-bound (BBB) secure Deterministic Authenticted
Encryption (DAE) scheme inspired by the Counter-in-Tweak encryption scheme by
Peyrin and Seurin.

DCT combines a fast almost-XOR-universal family of hash with a single call to a
2n-bit block cipher, and a BBB-secure encryption scheme. First, we describe our
construction generically with three independent keys, one for each component.
Next, we present an efficient instantiation which 
1. requires only a single key, 
2. provides software efficiency by encrypting at less than two cycles per
byte on current x64 processors, and 
3. produces only the minimal t-bit stretch for t bit authenticity. 

At the moment, we leave open two minor aspects for future work: our current
generic construction is defined for messages of at least 2n-t bits, and the
verification algorithm requires the inverse of the used 2n-bit block cipher and
the encryption scheme.

## Academic Paper: 
Christian Forler, Eik List, Stefan Lucks, and Jakob Wenzel: Efficient Beyond-
Birthday-Bound-Secure Deterministic Authenticated Encryption with Minimal
Stretch. IACR Cryptology ePrint Archive, 2016:407, 2016.

## Instance:
Our instantiation uses
- The efficient Bernstein-Rabin-Winograd (BRW) BRW-256 as 2n-bit hash function.
- Simpira for 2n-bit blocks as 2n-bit cipher.
- The Counter-in-Tweak (CTRT) encryption scheme, instantiated with the Deoxys-
  BC-128-128 tweakable block cipher.

## Content:
- C reference implementation
- C optimized implementation (requires the AES-NI, pclmulqdq, and SSE4.1
  instructions)

## Make Targets:
- ref-tests: Reference implementation tests
- sse-tests: Optimized implementation tests
- sse-bench: Optimized implementation benchmarks
- all
- clean

## Dependencies:
- clang/gcc
- make

## References:
[Ber07] Daniel J. Bernstein. Polynomial evaluation and message authentication.
        http://cr.yp.to/papers, permanent ID: b1ef3f2d385a926123e1517392e20f8c, 
        2, 2007.
[GM16]  Shay Gueron and Nicky Mouha. Simpira v2: A Family of Efficient 
        Permutations Using the AES Round Function. IACR Cryptology ePrint 
        Archive, 2016:122, 2016.
[JNP14] Jeremy Jean, Ivica Nikolic, and Thomas Peyrin. Deoxys v1.3, 2015. 
        Second-round submission to the CAESAR competition, 
        http://competitions.cr.yp.to/caesar-submissions.html. 
[PS15]  Thomas Peyrin and Yannick Seurin. Counter-in-Tweak: Authenticated 
        Encryption Modes for Tweakable Block Ciphers. IACR Cryptology ePrint 
        Archive, 2015:1049, 2015.
