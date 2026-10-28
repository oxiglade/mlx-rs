#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    volatile unsigned char *pointer = malloc(64);
    if (!pointer) return 2;
    pointer[0] = 42;
    free((void *)pointer);
    fprintf(stderr, "guard-malloc-control:%s\n", argv[1]);
    fflush(stderr);
    if (!strcmp(argv[1], "use_after_free")) return pointer[0];
    if (!strcmp(argv[1], "double_free")) free((void *)pointer);
    return 0;
}
