#include "../include/engine.h"
#include <stdlib.h>
#include <stdio.h>
#include <sys/stat.h>

static void ensure_output_dir()
{
    struct stat st = {0};
    if (stat("output", &st) == -1)
    {
        mkdir("output", 0700);
    }
}

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

void value_print_forward_graph(Value *v, const char *filename)
{
    ensure_output_dir();
    char filepath[256];
    snprintf(filepath, sizeof(filepath), "output/%s", filename);

    FILE *f = fopen(filepath, "w");
    if (!f)
        return;

    fprintf(f, "digraph G {\n");
    fprintf(f, "    node [shape=record];\n\n");

    reset_visited(v);
    Value **nodes = malloc(1000 * sizeof(Value *));
    size_t idx = 0;
    build_topo(v, nodes, &idx);

    for (size_t i = 0; i < idx; i++)
    {
        char label[100];
        value_get_forward_label(nodes[i], label, sizeof(label));
        fprintf(f, "    n%p [label=\"%s\"];\n", (void *)nodes[i], label);
    }

    reset_visited(v);
    for (size_t i = 0; i < idx; i++)
    {
        Value *node = nodes[i];
        for (size_t j = 0; j < node->prev_count; j++)
        {
            if (node->prev[j])
            {
                fprintf(f, "    n%p -> n%p;\n",
                        (void *)node->prev[j], (void *)node);
            }
        }
    }

    fprintf(f, "}\n");
    fclose(f);
    free(nodes);
}

void value_print_backward_graph(Value *v, const char *filename)
{
    ensure_output_dir();
    char filepath[256];
    snprintf(filepath, sizeof(filepath), "output/%s", filename);

    FILE *f = fopen(filepath, "w");
    if (!f)
        return;

    fprintf(f, "digraph G {\n");
    fprintf(f, "    rankdir=LR;\n");
    fprintf(f, "    node [shape=record, fontsize=10];\n\n");

    reset_visited(v);
    Value **nodes = malloc(1000 * sizeof(Value *));
    size_t idx = 0;
    build_topo(v, nodes, &idx);

    for (size_t i = 0; i < idx; i++)    
    {
        char label[100];
        value_get_label(nodes[i], label, sizeof(label));
        fprintf(f, "    n%p [label=\"%s\"];\n", (void *)nodes[i], label);
    }

    reset_visited(v);
    fprintf(f, "\n    edge [color=red, style=dashed];\n");

    for (size_t i = 0; i < idx; i++)
    {
        Value *node = nodes[i];
        for (size_t j = 0; j < node->prev_count; j++)
        {
            if (node->prev[j])
            {
                fprintf(f, "    n%p -> n%p;\n",
                        (void *)node, (void *)node->prev[j]);
            }
        }
    }

    fprintf(f, "}\n");
    fclose(f);
    free(nodes);
}

void value_print_full_graph(Value *v, const char *filename)
{
    ensure_output_dir();
    char filepath[256];
    snprintf(filepath, sizeof(filepath), "output/%s", filename);

    FILE *f = fopen(filepath, "w");
    if (!f)
        return;

    fprintf(f, "digraph G {\n");
    fprintf(f, "    rankdir=LR;\n");
    fprintf(f, "    node [shape=record, fontsize=10];\n\n");

    reset_visited(v);
    Value **nodes = malloc(1000 * sizeof(Value *));
    size_t idx = 0;
    build_topo(v, nodes, &idx);

    for (size_t i = 0; i < idx; i++)
    {
        char label[100];
        value_get_label(nodes[i], label, sizeof(label));
        fprintf(f, "    n%p [label=\"%s\"];\n", (void *)nodes[i], label);
    }

    reset_visited(v);
    fprintf(f, "\n    // Forward edges\n");
    for (size_t i = 0; i < idx; i++)
    {
        Value *node = nodes[i];
        for (size_t j = 0; j < node->prev_count; j++)
        {
            if (node->prev[j])
            {
                fprintf(f, "    n%p -> n%p;\n",
                        (void *)node->prev[j], (void *)node);
            }
        }
    }

    fprintf(f, "\n    // Backward edges\n");
    fprintf(f, "    edge [color=red, style=dashed, constraint=false];\n");
    for (size_t i = 0; i < idx; i++)
    {
        Value *node = nodes[i];
        for (size_t j = 0; j < node->prev_count; j++)
        {
            if (node->prev[j])
            {
                fprintf(f, "    n%p -> n%p;\n",
                        (void *)node, (void *)node->prev[j]);
            }
        }
    }

    fprintf(f, "}\n");
    fclose(f);
    free(nodes);
}
