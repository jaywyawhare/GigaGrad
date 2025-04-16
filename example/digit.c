#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <time.h>

#include "../include/value.h"
#include "../include/engine.h"

#define INPUT_SIZE 1024
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.001
#define EPOCHS 10
#define BATCH_SIZE 32
#define M_PI 3.14159265358979323846

double he_init(int fan_in)
{
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    double normal = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return normal * sqrt(2.0 / fan_in);
}

typedef struct
{
    float *image;
    int label;
} Dataset;

typedef struct
{
    Dataset *train;
    Dataset *test;
    int train_count;
    int test_count;
} SplitDataset;

Dataset *create_dataset(const char *filename, int *out_sample_count);
void free_dataset(Dataset *data, int sample_count);
int argmax(double *array, int length);
SplitDataset split_dataset(Dataset *data, int total_count);
void evaluate_model(Value **weights, Value **biases, Dataset *data, int count);
Value *cross_entropy_loss(Value **softmax_outputs, int label, int n);

void value_free_safe(Value **value_ptr)
{
    if (value_ptr && *value_ptr)
    {
        Value *v = *value_ptr;
        if (v->prev)
        {
            v->prev_count = 0;
            free(v->prev);
            v->prev = NULL;
        }
        if (v->backward_ctx)
        {
            free(v->backward_ctx);
            v->backward_ctx = NULL;
        }
        free(v);
        *value_ptr = NULL;
    }
}

Dataset *create_dataset(const char *filename, int *out_sample_count)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        perror("Error opening file");
        return NULL;
    }

    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    int line_count = 0;

    while ((read = getline(&line, &len, file)) != -1)
    {
        line_count++;
    }

    if (line_count <= 1)
    {
        free(line);
        fclose(file);
        return NULL;
    }

    rewind(file);
    getline(&line, &len, file);
    int sample_count = line_count - 1;
    Dataset *data = malloc(sample_count * sizeof(Dataset));
    if (!data)
    {
        perror("Memory allocation failed");
        free(line);
        fclose(file);
        return NULL;
    }

    int current_sample = 0;
    while ((read = getline(&line, &len, file)) != -1 && current_sample < sample_count)
    {
        line[strcspn(line, "\n")] = 0;
        int col_count = 0;

        char *temp = strdup(line);
        char *token = strtok(temp, ",");
        while (token)
        {
            col_count++;
            token = strtok(NULL, ",");
        }
        free(temp);

        if (col_count < 2)
            continue;

        int feature_count = col_count - 1;
        data[current_sample].image = malloc(feature_count * sizeof(float));
        if (!data[current_sample].image)
        {
            perror("Memory allocation failed for image");
            free_dataset(data, current_sample);
            free(line);
            fclose(file);
            return NULL;
        }

        token = strtok(line, ",");
        for (int i = 0; i < feature_count && token != NULL; i++)
        {
            if (i == 0)
            {
                data[current_sample].label = atoi(token);
            }
            else
            {
                data[current_sample].image[i - 1] = strtof(token, NULL);
            }
            token = strtok(NULL, ",");
        }

        current_sample++;
    }

    free(line);
    fclose(file);

    *out_sample_count = current_sample;
    return data;
}

void free_dataset(Dataset *data, int sample_count)
{
    for (int i = 0; i < sample_count; i++)
    {
        free(data[i].image);
    }
    free(data);
}

int argmax(double *array, int length)
{
    int max_index = 0;
    for (int i = 1; i < length; i++)
    {
        if (array[i] > array[max_index])
        {
            max_index = i;
        }
    }
    return max_index;
}

SplitDataset split_dataset(Dataset *data, int total_count)
{
    int train_target_per_class[OUTPUT_SIZE] = {0};
    int train_counts[OUTPUT_SIZE] = {0};
    int label_counts[OUTPUT_SIZE] = {0};

    for (int i = 0; i < total_count; i++)
    {
        label_counts[data[i].label]++;
    }

    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        train_target_per_class[i] = label_counts[i] / 1.5;
    }

    Dataset *train = malloc(sizeof(Dataset) * total_count);
    Dataset *test = malloc(sizeof(Dataset) * total_count);
    if (!train || !test)
    {
        perror("Memory allocation failed for split datasets");
        free(train);
        free(test);
        exit(EXIT_FAILURE);
    }

    int train_idx = 0;
    int test_idx = 0;

    for (int i = 0; i < total_count; i++)
    {
        int label = data[i].label;
        Dataset *target_dataset;
        int *target_idx;

        if (train_counts[label] < train_target_per_class[label])
        {
            target_dataset = train;
            target_idx = &train_idx;
            train_counts[label]++;
        }
        else
        {
            target_dataset = test;
            target_idx = &test_idx;
        }

        int feature_count = INPUT_SIZE;
        target_dataset[*target_idx].image = malloc(sizeof(float) * feature_count);
        if (!target_dataset[*target_idx].image)
        {
            perror("Memory allocation failed for dataset image");
            free_dataset(train, train_idx);
            free_dataset(test, test_idx);
            exit(EXIT_FAILURE);
        }
        memcpy(target_dataset[*target_idx].image, data[i].image, sizeof(float) * feature_count);
        target_dataset[*target_idx].label = label;
        (*target_idx)++;
    }

    return (SplitDataset){train, test, train_idx, test_idx};
}

void evaluate_model(Value **weights, Value **biases,
                    Dataset *data, int count)
{
    int correct = 0;
    double total_loss = 0.0;

    for (int i = 0; i < count; i++)
    {
        Value *input[INPUT_SIZE];
        for (int j = 0; j < INPUT_SIZE; j++)
            input[j] = value_create(data[i].image[j]);

        Value *out[OUTPUT_SIZE];
        for (int j = 0; j < OUTPUT_SIZE; j++)
        {
            Value *sum = value_create(0.0);
            for (int k = 0; k < INPUT_SIZE; k++)
            {
                Value *prod = value_mul(input[k], weights[k + j * INPUT_SIZE]);
                Value *new_sum = value_add(sum, prod);
                value_free_safe(&sum);
                value_free_safe(&prod);
                sum = new_sum;
            }
            out[j] = value_add(sum, biases[j]);
            value_free_safe(&sum);
        }

        Value **softmax_output = value_softmax(out, OUTPUT_SIZE);
        Value *loss = cross_entropy_loss(softmax_output, data[i].label, OUTPUT_SIZE);
        total_loss += loss->data;

        double probs[OUTPUT_SIZE];
        for (int j = 0; j < OUTPUT_SIZE; j++)
        {
            probs[j] = softmax_output[j]->data;
        }
        int predicted = argmax(probs, OUTPUT_SIZE);
        if (predicted == data[i].label)
            correct++;

        for (int j = 0; j < INPUT_SIZE; j++)
            value_free_safe(&input[j]);
        for (int j = 0; j < OUTPUT_SIZE; j++)
            value_free_safe(&softmax_output[j]);
        free(softmax_output);
        for (int j = 0; j < OUTPUT_SIZE; j++)
            value_free_safe(&out[j]);
        value_free_safe(&loss);
    }

    printf("Evaluation: Loss: %.4f | Accuracy: %.2f%%\n",
           total_loss / count, (double)correct / count * 100.0);
}

void backward_cross_entropy(Value *self)
{
    int n = self->prev_count;
    int label = *(int *)self->backward_ctx;

    for (int i = 0; i < n; i++)
    {
        double indicator = (i == label) ? 1.0 : 0.0;
        self->prev[i]->grad = self->grad * (self->prev[i]->data - indicator);
    }
}

Value *cross_entropy_loss(Value **softmax_outputs, int label, int n)
{
    const double epsilon = 1e-7;
    Value *neg_log_prob = value_create(-log(softmax_outputs[label]->data + epsilon));
    if (!neg_log_prob)
    {
        return NULL;
    }

    neg_log_prob->prev = malloc(n * sizeof(Value *));
    if (!neg_log_prob->prev)
    {
        value_free(neg_log_prob);
        return NULL;
    }

    int *label_ptr = malloc(sizeof(int));
    if (!label_ptr)
    {
        free(neg_log_prob->prev);
        value_free(neg_log_prob);
        return NULL;
    }

    *label_ptr = label;

    for (int i = 0; i < n; i++)
    {
        neg_log_prob->prev[i] = softmax_outputs[i];
    }
    neg_log_prob->prev_count = n;
    neg_log_prob->backward = backward_cross_entropy;
    neg_log_prob->backward_ctx = label_ptr;

    return neg_log_prob;
}

int main()
{
    const char *file_path = "data.csv";
    int sample_count;
    Dataset *full_data = create_dataset(file_path, &sample_count);
    if (!full_data)
    {
        printf("Failed to create dataset\n");
        return 1;
    }

    SplitDataset split = split_dataset(full_data, sample_count);
    if (split.train_count == 0 || split.test_count == 0)
    {
        printf("Insufficient data for training or testing\n");
        free_dataset(full_data, sample_count);
        return 1;
    }

    printf("Split dataset: %d training samples, %d test samples\n",
           split.train_count, split.test_count);

    Value *weights[INPUT_SIZE * OUTPUT_SIZE];
    Value *biases[OUTPUT_SIZE];

    for (int i = 0; i < INPUT_SIZE * OUTPUT_SIZE; i++)
        weights[i] = value_create(he_init(INPUT_SIZE));
    for (int i = 0; i < OUTPUT_SIZE; i++)
        biases[i] = value_create(0.0);

    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        double epoch_loss = 0.0;
        int correct = 0;

        for (int batch_start = 0; batch_start < split.train_count; batch_start += BATCH_SIZE)
        {
            int batch_end = batch_start + BATCH_SIZE;
            if (batch_end > split.train_count)
                batch_end = split.train_count;

            for (int sample_index = batch_start; sample_index < batch_end; sample_index++)
            {

                for (int i = 0; i < INPUT_SIZE * OUTPUT_SIZE; i++)
                    value_zero_grad(weights[i]);
                for (int i = 0; i < OUTPUT_SIZE; i++)
                    value_zero_grad(biases[i]);

                Value *input[INPUT_SIZE];
                for (int i = 0; i < INPUT_SIZE; i++)
                    input[i] = value_create(split.train[sample_index].image[i]);

                Value *out[OUTPUT_SIZE];
                for (int i = 0; i < OUTPUT_SIZE; i++)
                {
                    Value *sum = value_create(0.0);
                    for (int j = 0; j < INPUT_SIZE; j++)
                    {
                        Value *prod = value_mul(input[j], weights[j + i * INPUT_SIZE]);
                        Value *new_sum = value_add(sum, prod);
                        value_free_safe(&sum);
                        value_free_safe(&prod);
                        sum = new_sum;
                    }
                    out[i] = value_add(sum, biases[i]);
                    value_free_safe(&sum);
                }

                Value **softmax_output = value_softmax(out, OUTPUT_SIZE);
                Value *loss = cross_entropy_loss(softmax_output, split.train[sample_index].label, OUTPUT_SIZE);

                if (epoch == 0 && sample_index == 0)
                {
                    value_print_forward_graph(loss, "digit_forward.dot");

                    value_backward(loss);

                    value_print_backward_graph(loss, "digit_backward.dot");
                    value_print_full_graph(loss, "digit_full.dot");

                    printf("\nComputation graphs generated for neural network:\n");
                    printf("Forward graph: output/digit_forward.dot\n");
                    printf("Backward graph: output/digit_backward.dot\n");
                    printf("Full graph: output/digit_full.dot\n");
                    printf("\nTo visualize, run:\n");
                    printf("dot -Tsvg output/digit_*.dot -o digit_*.svg\n\n");
                }
                else
                {
                    value_backward(loss);
                }

                epoch_loss += loss->data;

                double probs[OUTPUT_SIZE];
                for (int i = 0; i < OUTPUT_SIZE; i++)
                {
                    probs[i] = softmax_output[i]->data;
                }
                int predicted = argmax(probs, OUTPUT_SIZE);
                if (predicted == split.train[sample_index].label)
                    correct++;

                double clip_threshold = 1.0;
                for (int i = 0; i < INPUT_SIZE * OUTPUT_SIZE; i++)
                {
                    double grad = weights[i]->grad;
                    if (grad > clip_threshold)
                        grad = clip_threshold;
                    if (grad < -clip_threshold)
                        grad = -clip_threshold;
                    weights[i]->data -= LEARNING_RATE * grad;
                }
                for (int i = 0; i < OUTPUT_SIZE; i++)
                {
                    double grad = biases[i]->grad;
                    if (grad > clip_threshold)
                        grad = clip_threshold;
                    if (grad < -clip_threshold)
                        grad = -clip_threshold;
                    biases[i]->data -= LEARNING_RATE * grad;
                }

                for (int i = 0; i < INPUT_SIZE; i++)
                    value_free_safe(&input[i]);
                for (int i = 0; i < OUTPUT_SIZE; i++)
                    value_free_safe(&softmax_output[i]);
                free(softmax_output);
                for (int i = 0; i < OUTPUT_SIZE; i++)
                    value_free_safe(&out[i]);
                value_free_safe(&loss);
            }
        }

        printf("Epoch %d | Train Loss: %.4f | Train Accuracy: %.2f%%\n", epoch + 1,
               epoch_loss / split.train_count, (double)correct / split.train_count * 100.0);
    }

    evaluate_model(weights, biases,
                   split.test, split.test_count);

    for (int i = 0; i < INPUT_SIZE * OUTPUT_SIZE; i++)
        value_free_safe(&weights[i]);
    for (int i = 0; i < OUTPUT_SIZE; i++)
        value_free_safe(&biases[i]);

    free_dataset(split.train, split.train_count);
    free_dataset(split.test, split.test_count);
    free_dataset(full_data, sample_count);

    return 0;
}