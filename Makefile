CC = gcc
CFLAGS = -Wall -Wextra -Iinclude 
LDFLAGS = -lm -ljpeg

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
EXAMPLE_DIR = example

SRCS = main.c $(SRC_DIR)/value.c $(SRC_DIR)/engine.c
OBJS = $(SRCS:%.c=$(OBJ_DIR)/%.o)
TARGET = $(BIN_DIR)/gigagrad

EXAMPLE_SRCS = $(EXAMPLE_DIR)/digit.c $(SRC_DIR)/value.c $(SRC_DIR)/engine.c
EXAMPLE_OBJS = $(EXAMPLE_SRCS:%.c=$(OBJ_DIR)/%.o)
EXAMPLE_TARGET = $(BIN_DIR)/digit

.PHONY: all clean examples

all: $(TARGET) examples

examples: $(EXAMPLE_TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(EXAMPLE_TARGET): $(EXAMPLE_OBJS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/$(SRC_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/$(EXAMPLE_DIR)/%.o: $(EXAMPLE_DIR)/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)