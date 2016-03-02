# export ASAN_SYMBOLIZER_PATH=/usr/bin/llvm-symbolizer-3.5
# export ASAN_OPTIONS=symbolize=1

CC = clang
CFLAGS = -W -Wall -Wextra -std=c99 -mavx2 -maes -mpclmul -march=native
REFLAGS = -W -Wall -Wextra -std=c99 -march=native

SSE = sse
REF = ref
UTILS = utils

OBJECTS = $(SSE)/brw256.o $(SSE)/dct.o 
REF_OBJECTS = $(REF)/aes.o $(REF)/brw256.o $(REF)/dct.o 

ifdef DEBUG
CFLAGS += -g3 -DDEBUG -fsanitize=undefined -fsanitize=address -fsanitize=alignment -ftrapv -fno-omit-frame-pointer -fno-optimize-sibling-calls
REFLAGS += -g3 -DDEBUG -fsanitize=undefined -fsanitize=address -fsanitize=alignment -ftrapv -fno-omit-frame-pointer -fno-optimize-sibling-calls
else
CFLAGS += -O3
REFLAGS += -O3
endif

.PHONY: clean sse-tests ref-tests benchmark all

all: ref-tests sse-tests sse-bench

$(SSE)/%.o: $(SSE)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(REF)/%.o: $(REF)/%.c
	$(CC) $(REFLAGS) -c $< -o $@

$(UTILS)/%.o: $(UTILS)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

ref-tests: $(REF_OBJECTS)
	$(CC) $(REFLAGS) -I $(REF) $(REF_OBJECTS) $(UTILS)/tests.c -o $@

sse-bench: $(OBJECTS)
	$(CC) $(CFLAGS) -I $(SSE) $(OBJECTS) $(UTILS)/benchmark.c -o $@

sse-tests: $(OBJECTS)
	$(CC) $(CFLAGS) -I $(SSE) $(OBJECTS) $(UTILS)/tests.c -o $@

generatetests: $(REF_OBJECTS)
	$(CC) $(REFLAGS) -I$(REF) $(UTILS)/generatetests.c $(UTILS)/encrypt.c $^  -o $@

clean:
	rm -f $(REF)/*.o $(SSE)/*.o $(UTILS)/*.o bench core architectures \
		implementors ref-tests sse-bench sse-tests

