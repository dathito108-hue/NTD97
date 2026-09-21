#define main llama2c_reference_main
#include "llama2c-run.c"
#undef main

static void print_tokens(Tokenizer * tokenizer, const char * text) {
    size_t capacity = strlen(text) + 3;
    int * tokens = malloc(capacity * sizeof(int));
    if (tokens == NULL) {
        fprintf(stderr, "token allocation failed\n");
        exit(EXIT_FAILURE);
    }

    int count = 0;
    encode(tokenizer, (char *) text, 1, 0, tokens, &count);
    for (int i = 0; i < count; i++) {
        if (i > 0) {
            putchar(',');
        }
        printf("%d", tokens[i]);
    }
    putchar('\n');
    free(tokens);
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <tok512.bin>\n", argv[0]);
        return EXIT_FAILURE;
    }

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, argv[1], 512);

    const char * prompts[] = {
        "",
        "hello",
        "hello!",
        "Once upon a time",
        "Lily's ball.",
        " red ball",
        "one  two",
    };
    const size_t prompt_count = sizeof(prompts) / sizeof(prompts[0]);
    for (size_t i = 0; i < prompt_count; i++) {
        print_tokens(&tokenizer, prompts[i]);
    }

    free_tokenizer(&tokenizer);
    return EXIT_SUCCESS;
}
