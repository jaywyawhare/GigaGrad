#include "include/value.h"
#include "include/engine.h"
#include <stdio.h>

int main()
{
    Value *a = value_create(3.0);
    Value *b = value_create(4.0);
    Value *n3 = value_create(3.0);

    Value *c = value_add(a, b);
    Value *d = value_mul(c, c);
    Value *e = value_add(d, a);
    Value *f = value_add(e, n3);

    printf("Forward Pass Results:\n");
    printf("a = %.4f\n", a->data);
    printf("b = %.4f\n", b->data);
    printf("c = (a + b) = %.4f\n", c->data);
    printf("d = (c * c) = %.4f\n", d->data);
    printf("e = (d + a) = %.4f\n", e->data);
    printf("f = (e + 3) = %.4f\n", f->data);

    value_backward(f);

    printf("\nBackward Pass Results (Gradients):\n");
    printf("∂f/∂a = %.4f\n", a->grad);
    printf("∂f/∂b = %.4f\n", b->grad);
    printf("∂f/∂c = %.4f\n", c->grad);
    printf("∂f/∂d = %.4f\n", d->grad);
    printf("∂f/∂e = %.4f\n", e->grad);
    printf("∂f/∂f = %.4f\n", f->grad);

    value_print_forward_graph(f, "forward_graph.dot");
    value_print_backward_graph(f, "backward_graph.dot");
    value_print_full_graph(f, "full_graph.dot");

    printf("\nTo visualize graphs, run:\n");
    printf("dot -Tsvg output/filename.dot -o output/filename.svg\n");

    value_free(f);
    value_free(e);
    value_free(d);
    value_free(c);
    value_free(b);
    value_free(a);
    value_free(n3);

    return 0;
}
