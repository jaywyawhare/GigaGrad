#ifndef VALUE_H
#define VALUE_H

#include <stddef.h>

typedef struct Value Value;

typedef void (*BackwardFn)(Value *self);

struct Value
{
    double data;
    double grad;

    Value **prev;
    size_t prev_count;

    BackwardFn backward;
    void *backward_ctx;
    int visited;
};

Value *value_create(double data);
void value_free(Value *v);

Value *value_add(Value *a, Value *b);
Value *value_sub(Value *a, Value *b);
Value *value_mul(Value *a, Value *b);
Value *value_div(Value *a, Value *b);
Value *value_relu(Value *x);
Value *value_pow(Value *base, double exponent);

void value_zero_grad(Value *v);
void value_backward(Value *v);

void value_print(Value *v);

const char *value_get_op_symbol(Value *v);
void value_get_label(Value *v, char *buf, size_t size);
void value_get_forward_label(Value *v, char *buf, size_t size);

#endif
