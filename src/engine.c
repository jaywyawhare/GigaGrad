#include "../include/engine.h"
#include <stdlib.h>

static void build_topo(Value *v, Value **topo, size_t *idx)
{
    if (v->visited)
        return;
    v->visited = 1;

    for (size_t i = 0; i < v->prev_count; i++)
    {
        if (v->prev[i])
        {
            build_topo(v->prev[i], topo, idx);
        }
    }
    topo[*idx] = v;
    (*idx)++;
}

static void count_nodes(Value *node, size_t *count)
{
    if (node && !node->visited)
    {
        node->visited = 1;
        (*count)++;
        for (size_t i = 0; i < node->prev_count; i++)
        {
            count_nodes(node->prev[i], count);
        }
    }
}

static void reset_visited(Value *node)
{
    if (node && node->visited)
    {
        node->visited = 0;
        for (size_t i = 0; i < node->prev_count; i++)
        {
            reset_visited(node->prev[i]);
        }
    }
}

void value_zero_grad(Value *v)
{
    v->grad = 0.0;
    v->visited = 0;
}

void value_backward(Value *v)
{
    size_t node_count = 0;
    Value **topo = NULL;
    size_t topo_idx = 0;

    count_nodes(v, &node_count);
    reset_visited(v);

    topo = malloc(node_count * sizeof(Value *));
    if (!topo)
        return;

    build_topo(v, topo, &topo_idx);

    if (v->grad == 0)
    {
        v->grad = 1.0;
    }

    for (int i = topo_idx - 1; i >= 0; i--)
    {
        if (topo[i]->backward)
        {
            topo[i]->backward(topo[i]);
        }
    }

    reset_visited(v);
    free(topo);
}

void value_set_grad(Value *v, double grad)
{
    v->grad = grad;
}
