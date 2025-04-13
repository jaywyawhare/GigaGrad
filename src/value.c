#include "../include/value.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

void backward_add(Value *self)
{
    self->prev[0]->grad += self->grad;
    self->prev[1]->grad += self->grad;
}

void backward_sub(Value *self)
{
    self->prev[0]->grad += self->grad;
    self->prev[1]->grad -= self->grad;
}

void backward_mul(Value *self)
{
    self->prev[0]->grad += self->prev[1]->data * self->grad;
    self->prev[1]->grad += self->prev[0]->data * self->grad;
}

void backward_div(Value *self)
{
    self->prev[0]->grad += self->grad / self->prev[1]->data;
    self->prev[1]->grad -= (self->grad * self->prev[0]->data) / (self->prev[1]->data * self->prev[1]->data);
}

void backward_pow(Value *self)
{
    double exponent = *(double *)self->backward_ctx;
    self->prev[0]->grad += exponent * pow(self->prev[0]->data, exponent - 1) * self->grad;
}

void backward_relu(Value *self)
{
    self->prev[0]->grad += (self->prev[0]->data > 0 ? self->grad : 0);
}

Value *value_create(double data)
{
    Value *v = malloc(sizeof(Value));
    if (!v)
        return NULL;
    v->data = data;
    v->grad = 0.0;
    v->prev = NULL;
    v->prev_count = 0;
    v->backward = NULL;
    v->backward_ctx = NULL;
    v->visited = 0;
    return v;
}

void value_free(Value *v)
{
    if (!v)
        return;
    if (v->prev)
    {
        free(v->prev);
    }
    if (v->backward_ctx)
    {
        free(v->backward_ctx);
    }
    free(v);
}

void value_print(Value *v)
{
    printf("Value(data=%.4f, grad=%.4f)\n", v->data, v->grad);
}

Value *value_add(Value *a, Value *b)
{
    Value *out = value_create(a->data + b->data);
    if (!out)
        return NULL;

    out->prev = malloc(2 * sizeof(Value *));
    if (!out->prev)
    {
        value_free(out);
        return NULL;
    }

    out->prev[0] = a;
    out->prev[1] = b;
    out->prev_count = 2;
    out->backward = backward_add;
    return out;
}

Value *value_sub(Value *a, Value *b)
{
    Value *out = value_create(a->data - b->data);
    if (!out)
        return NULL;

    out->prev = malloc(2 * sizeof(Value *));
    if (!out->prev)
    {
        value_free(out);
        return NULL;
    }

    out->prev[0] = a;
    out->prev[1] = b;
    out->prev_count = 2;
    out->backward = backward_sub;
    return out;
}

Value *value_mul(Value *a, Value *b)
{
    Value *out = value_create(a->data * b->data);
    if (!out)
        return NULL;

    out->prev = malloc(2 * sizeof(Value *));
    if (!out->prev)
    {
        value_free(out);
        return NULL;
    }

    out->prev[0] = a;
    out->prev[1] = b;
    out->prev_count = 2;
    out->backward = backward_mul;
    return out;
}

Value *value_div(Value *a, Value *b)
{
    Value *out = value_create(a->data / b->data);
    if (!out)
        return NULL;

    out->prev = malloc(2 * sizeof(Value *));
    if (!out->prev)
    {
        value_free(out);
        return NULL;
    }

    out->prev[0] = a;
    out->prev[1] = b;
    out->prev_count = 2;
    out->backward = backward_div;
    return out;
}

Value *value_relu(Value *x)
{
    Value *out = value_create(x->data > 0 ? x->data : 0);
    if (!out)
        return NULL;

    out->prev = malloc(sizeof(Value *));
    if (!out->prev)
    {
        value_free(out);
        return NULL;
    }

    out->prev[0] = x;
    out->prev_count = 1;
    out->backward = backward_relu;
    return out;
}

Value *value_pow(Value *base, double exponent)
{
    Value *out = value_create(pow(base->data, exponent));
    if (!out)
        return NULL;

    out->prev = malloc(sizeof(Value *));
    if (!out->prev)
    {
        value_free(out);
        return NULL;
    }

    double *exp_ptr = malloc(sizeof(double));
    if (!exp_ptr)
    {
        value_free(out);
        return NULL;
    }
    *exp_ptr = exponent;

    out->prev[0] = base;
    out->prev_count = 1;
    out->backward = backward_pow;
    out->backward_ctx = exp_ptr;
    return out;
}

const char *value_get_op_symbol(Value *v)
{
    if (!v->backward)
        return "";
    if (v->backward == backward_add)
        return "+";
    if (v->backward == backward_mul)
        return "*";
    if (v->backward == backward_sub)
        return "-";
    if (v->backward == backward_div)
        return "/";
    if (v->backward == backward_relu)
        return "ReLU";
    if (v->backward == backward_pow)
        return "^";
    return "?";
}

void value_get_label(Value *v, char *buf, size_t size)
{
    const char *op = value_get_op_symbol(v);
    if (op[0])
    {
        snprintf(buf, size, "{%s | data %.4f | grad %.4f}",
                 op, v->data, v->grad);
    }
    else
    {
        snprintf(buf, size, "{data %.4f | grad %.4f}",
                 v->data, v->grad);
    }
}

void value_get_forward_label(Value *v, char *buf, size_t size)
{
    const char *op = value_get_op_symbol(v);
    if (op[0])
    {
        snprintf(buf, size, "{%s | data %.4f}",
                 op, v->data);
    }
    else
    {
        snprintf(buf, size, "{data %.4f}",
                 v->data);
    }
}