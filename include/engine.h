#ifndef ENGINE_H
#define ENGINE_H

#include "value.h"

void value_zero_grad(Value *v);
void value_backward(Value *v);
void value_set_grad(Value *v, double grad);

void value_print_forward_graph(Value *v, const char *filename);
void value_print_backward_graph(Value *v, const char *filename);
void value_print_full_graph(Value *v, const char *filename);

#endif
