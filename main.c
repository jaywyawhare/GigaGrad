#include "include/value.h"
#include <stdio.h>

int main()
{
    Value *a = value_create(4.0);
    Value *b = value_create(2.0);
    Value *c = value_create(-1.0);

    printf("Initial values:\n");
    printf("a: ");
    value_print(a);
    printf("b: ");
    value_print(b);
    printf("c: ");
    value_print(c);

    Value *sum = value_add(a, b);   
    Value *diff = value_sub(a, b);  
    Value *prod = value_mul(a, b);  
    Value *quot = value_div(a, b);  
    Value *pow = value_pow(a, 2.0); 
    Value *relu_pos = value_relu(a);
    Value *relu_neg = value_relu(c);

    printf("\nOperations results:\n");
    printf("a + b = ");
    value_print(sum);
    printf("a - b = ");
    value_print(diff);
    printf("a * b = ");
    value_print(prod);
    printf("a / b = ");
    value_print(quot);
    printf("a^2 = ");
    value_print(pow);
    printf("relu(a) = ");
    value_print(relu_pos);
    printf("relu(c) = ");
    value_print(relu_neg);

    value_free(sum);
    value_free(diff);
    value_free(prod);
    value_free(quot);
    value_free(pow);
    value_free(relu_pos);
    value_free(relu_neg);

    value_free(c);
    value_free(b);
    value_free(a);

    return 0;
}
