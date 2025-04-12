#include "../include/value.h"

#include <stdlib.h>
#include <stdio.h>

Value *value_create(double data)
{
    Value *v = malloc(sizeof(Value));
    v->data = data;
    v->grad = 0.0;
    v->prev = NULL;
    v->prev_count = 0;
    v->backward = NULL;
    v->visited = 0;
    return v;
}

void value_free(Value *v)
{
    if (v->prev)
    {
        free(v->prev);
    }
    free(v);
}

void value_print(Value *v)
{
    printf("Value(data=%.4f, grad=%.4f)\n", v->data, v->grad);
}
