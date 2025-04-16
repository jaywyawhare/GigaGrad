#include "../include/engine.h"
#include <stdlib.h>
#include <stdio.h>
#include <sys/stat.h>

static void ensure_output_dir()
{
    struct stat st = {0};
    if (stat("output", &st) == -1)
    {
        if (mkdir("output", 0700) == -1)
        {
            perror("Failed to create output directory");
            exit(1);
        }
    }
}

static void build_topo(Value *v, Value **topo, size_t *idx)
{
    if (!v || v->visited || !topo || !idx)
        return;

    v->visited = 1;

    if (v->prev)
    {
        for (size_t i = 0; i < v->prev_count; i++)
        {
            if (v->prev[i])
                build_topo(v->prev[i], topo, idx);
        }
    }

    topo[*idx] = v;
    (*idx)++;
}

static void count_nodes(Value *node, size_t *count)
{
    if (!node || !count)
        return;

    if (node->visited)
        return;

    node->visited = 1;
    (*count)++;

    if (node->prev)
    {
        for (size_t i = 0; i < node->prev_count; i++)
        {
            if (node->prev[i])
            {
                count_nodes(node->prev[i], count);
            }
        }
    }
}

static void reset_visited(Value *node)
{
    if (!node || !node->visited)
        return;

    node->visited = 0;

    if (node->prev)
    {
        for (size_t i = 0; i < node->prev_count; i++)
        {
            if (node->prev[i])
            {
                reset_visited(node->prev[i]);
            }
        }
    }
}

static void write_graph_header(FILE *f, const char *title)
{
    fprintf(f, "digraph G {\n");
    fprintf(f, "    rankdir=LR;\n");
    fprintf(f, "    bgcolor=\"#ffffff\";\n");
    fprintf(f, "    title=\"%s\";\n", title);
    fprintf(f, "    node [shape=record, style=filled, fillcolor=\"#f8e8e8\", fontsize=10];\n");
    fprintf(f, "    edge [color=\"#2c3e50\"];\n\n");
}

void value_zero_grad(Value *v)
{
    if (!v)
        return;
    v->grad = 0.0;
    v->visited = 0;
}

void value_backward(Value *v)
{
    if (!v)
        return;

    size_t node_count = 0;
    Value **topo = NULL;
    size_t topo_idx = 0;

    reset_visited(v);
    count_nodes(v, &node_count);
    reset_visited(v);

    if (node_count == 0)
        return;

    topo = malloc(node_count * sizeof(Value *));
    if (!topo)
        return;

    build_topo(v, topo, &topo_idx);
    if (topo_idx != node_count)
    {
        free(topo);
        return;
    }

    for (size_t i = 0; i < topo_idx; i++)
        topo[i]->grad = 0.0;

    v->grad = 1.0;

    for (int i = (int)topo_idx - 1; i >= 0; i--)
    {
        if (topo[i] && topo[i]->backward)
            topo[i]->backward(topo[i]);
    }

    reset_visited(v);
    free(topo);
}

void value_set_grad(Value *v, double grad)
{
    if (v)
        v->grad = grad;
}

void value_print_forward_graph(Value *v, const char *filename)
{
    if (!v || !filename)
        return;

    ensure_output_dir();
    char filepath[256];
    snprintf(filepath, sizeof(filepath), "output/%s", filename);

    FILE *f = fopen(filepath, "w");
    if (!f)
        return;

    write_graph_header(f, "Forward Computation Graph");

    fprintf(f, "    compound=true;\n");
    fprintf(f, "    splines=ortho;\n");
    fprintf(f, "    nodesep=0.5;\n");
    fprintf(f, "    ranksep=0.7;\n\n");

    reset_visited(v);
    size_t node_count = 0;
    count_nodes(v, &node_count);
    reset_visited(v);

    Value **nodes = malloc(node_count * sizeof(Value *));
    if (!nodes)
    {
        fclose(f);
        return;
    }

    size_t idx = 0;
    build_topo(v, nodes, &idx);

    if (idx != node_count)
    {
        free(nodes);
        fclose(f);
        return;
    }

    fprintf(f, "    subgraph cluster_0 {\n");
    fprintf(f, "        style=filled;\n");
    fprintf(f, "        fillcolor=\"#f8e8e8\";\n");
    fprintf(f, "        color=\"#2c3e50\";\n");
    fprintf(f, "        label=\"Forward Pass\";\n");
    fprintf(f, "        fontsize=12;\n");

    fprintf(f, "        { rank=same; ");
    for (size_t i = 0; i < idx; i++)
    {
        if (!nodes[i]->prev)
        {
            fprintf(f, "n%p; ", (void *)nodes[i]);
        }
    }
    fprintf(f, "}\n");

    for (size_t i = 0; i < idx; i++)
    {
        char label[100];
        value_get_forward_label(nodes[i], label, sizeof(label));
        fprintf(f, "        n%p [label=\"%s\"];\n", (void *)nodes[i], label);
    }

    fprintf(f, "    }\n");

    reset_visited(v);
    for (size_t i = 0; i < idx; i++)
    {
        Value *node = nodes[i];
        for (size_t j = 0; j < node->prev_count; j++)
        {
            if (node->prev[j])
            {
                fprintf(f, "    n%p -> n%p [weight=2];\n",
                        (void *)node->prev[j], (void *)node);
            }
        }
    }

    fprintf(f, "}\n");
    fclose(f);
    free(nodes);
    reset_visited(v);
}

void value_print_backward_graph(Value *v, const char *filename)
{
    if (!v || !filename)
        return;

    ensure_output_dir();
    char filepath[256];
    snprintf(filepath, sizeof(filepath), "output/%s", filename);

    FILE *f = fopen(filepath, "w");
    if (!f)
        return;

    write_graph_header(f, "Backward Computation Graph");

    fprintf(f, "    compound=true;\n");
    fprintf(f, "    splines=ortho;\n");
    fprintf(f, "    nodesep=0.5;\n");
    fprintf(f, "    ranksep=0.7;\n\n");

    reset_visited(v);
    size_t node_count = 0;
    count_nodes(v, &node_count);
    reset_visited(v);

    Value **nodes = malloc(node_count * sizeof(Value *));
    if (!nodes)
    {
        fclose(f);
        return;
    }

    size_t idx = 0;
    build_topo(v, nodes, &idx);

    if (idx != node_count)
    {
        free(nodes);
        fclose(f);
        return;
    }

    fprintf(f, "    subgraph cluster_0 {\n");
    fprintf(f, "        style=filled;\n");
    fprintf(f, "        fillcolor=\"#f8e8e8\";\n");
    fprintf(f, "        color=\"#2c3e50\";\n");
    fprintf(f, "        label=\"Backward Pass\";\n");
    fprintf(f, "        fontsize=12;\n");

    fprintf(f, "        { rank=same; ");
    for (size_t i = idx - 1; i < idx; i--)
    {
        if (nodes[i]->prev_count == 0)
        {
            fprintf(f, "n%p; ", (void *)nodes[i]);
        }
    }
    fprintf(f, "}\n");

    for (size_t i = 0; i < idx; i++)
    {
        char label[100];
        value_get_label(nodes[i], label, sizeof(label));
        fprintf(f, "        n%p [label=\"%s\"];\n", (void *)nodes[i], label);
    }

    fprintf(f, "    }\n");

    fprintf(f, "\n    edge [color=\"#e74c3c\", style=dashed];\n");

    for (size_t i = 0; i < idx; i++)
    {
        Value *node = nodes[i];
        for (size_t j = 0; j < node->prev_count; j++)
        {
            if (node->prev[j])
            {
                fprintf(f, "    n%p -> n%p [weight=2];\n",
                        (void *)node, (void *)node->prev[j]);
            }
        }
    }

    fprintf(f, "}\n");
    fclose(f);
    free(nodes);
    reset_visited(v);
}

void value_print_full_graph(Value *v, const char *filename)
{
    if (!v || !filename)
        return;

    ensure_output_dir();
    char filepath[256];
    snprintf(filepath, sizeof(filepath), "output/%s", filename);

    FILE *f = fopen(filepath, "w");
    if (!f)
        return;

    write_graph_header(f, "Complete Computation Graph");

    fprintf(f, "    compound=true;\n");
    fprintf(f, "    splines=ortho;\n");
    fprintf(f, "    nodesep=0.5;\n");
    fprintf(f, "    ranksep=0.7;\n\n");

    size_t node_count = 0;
    count_nodes(v, &node_count);
    reset_visited(v);

    Value **nodes = malloc(node_count * sizeof(Value *));
    if (!nodes)
    {
        fclose(f);
        return;
    }

    size_t idx = 0;
    build_topo(v, nodes, &idx);

    if (idx != node_count)
    {
        free(nodes);
        fclose(f);
        return;
    }

    fprintf(f, "    subgraph cluster_forward {\n");
    fprintf(f, "        style=filled;\n");
    fprintf(f, "        fillcolor=\"#f8e8e8\";\n");
    fprintf(f, "        color=\"#2c3e50\";\n");
    fprintf(f, "        label=\"Computation Graph\";\n");
    fprintf(f, "        fontsize=12;\n");

    for (size_t i = 0; i < idx; i++)
    {
        if (nodes[i])
        {
            char label[100];
            value_get_label(nodes[i], label, sizeof(label));
            fprintf(f, "        n%p [label=\"%s\"];\n", (void *)nodes[i], label);
        }
    }
    fprintf(f, "    }\n\n");

    reset_visited(v);

    fprintf(f, "\n    // Forward edges\n");
    fprintf(f, "    edge [color=\"#2c3e50\", style=solid];\n");
    for (size_t i = 0; i < idx; i++)
    {
        Value *node = nodes[i];
        if (node && node->prev)
        {
            for (size_t j = 0; j < node->prev_count; j++)
            {
                if (node->prev[j])
                {
                    fprintf(f, "    n%p -> n%p;\n",
                            (void *)node->prev[j], (void *)node);
                }
            }
        }
    }

    fprintf(f, "\n    // Backward edges\n");
    fprintf(f, "    edge [color=\"#e74c3c\", style=dashed];\n");
    for (size_t i = 0; i < idx; i++)
    {
        Value *node = nodes[i];
        if (node && node->prev)
        {
            for (size_t j = 0; j < node->prev_count; j++)
            {
                if (node->prev[j])
                {
                    fprintf(f, "    n%p -> n%p [color=\"#e74c3c\", style=dashed];\n",
                            (void *)node, (void *)node->prev[j]);
                }
            }
        }
    }

    fprintf(f, "}\n");
    fclose(f);
    free(nodes);
    reset_visited(v);
}
