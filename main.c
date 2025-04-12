#include "include/value.h"
#include <stdio.h>

int main()
{
    Value *v = value_create(69.420);

    value_print(v);
    value_free(v);

    return 0;
}
