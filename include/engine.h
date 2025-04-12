#ifndef ENGINE_H
#define ENGINE_H

#include "value.h"

void value_zero_grad(Value *v);
void value_backward(Value *v);
void value_set_grad(Value *v, double grad);

#endif
