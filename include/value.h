#ifndef VALUE_H
#define VALUE_H

#include <stddef.h>

typedef struct Value Value;

typedef void (*BackwardFn)(Value *);

struct Value
{
    double data;
    double grad;

    Value **prev;
    size_t prev_count;

    BackwardFn backward;
    int visited;
};

Value *value_create(double data);
void value_free(Value *v);
void value_print(Value *v);

#endif
