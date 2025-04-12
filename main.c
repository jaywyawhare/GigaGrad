#include "include/value.h"
#include "include/engine.h"
#include <stdio.h>

int main()
{
    Value *a = value_create(2.0);
    Value *b = value_create(-3.0);
    Value *c = value_create(10.0);

    Value *x = value_mul(a, b);
    Value *y = value_add(x, c);
    Value *z = value_pow(y, 2.0);

    printf("\nForward Pass:\n");
    printf("a = %.2f\n", a->data);
    printf("b = %.2f\n", b->data);
    printf("c = %.2f\n", c->data);
    printf("x = a * b = %.2f\n", x->data);
    printf("y = x + c = %.2f\n", y->data);
    printf("z = y^2 = %.2f\n", z->data);

    printf("\nBefore Backpropagation:\n");
    printf("Initial gradients:\n");
    printf("a (grad): %.4f\n", a->grad);
    printf("b (grad): %.4f\n", b->grad);
    printf("c (grad): %.4f\n", c->grad);

    value_backward(z);

    printf("\nAfter Backpropagation:\n");
    printf("Final gradients:\n");
    printf("z = (a*b + c)^2, where a=%.1f, b=%.1f, c=%.1f\n", a->data, b->data, c->data);
    printf("∂z/∂a = %.4f\n", a->grad);
    printf("∂z/∂b = %.4f\n", b->grad);
    printf("∂z/∂c = %.4f\n", c->grad);

    value_free(z);
    value_free(y);
    value_free(x);
    value_free(c);
    value_free(b);
    value_free(a);

    return 0;
}
