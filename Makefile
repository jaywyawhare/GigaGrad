CC = gcc
CFLAGS = -Wall -Wextra -Iinclude
LDFLAGS = 
SRC = src/value.c main.c
OBJ = $(SRC:%.c=obj/%.o)
TARGET = bin/gigagrad

all: $(TARGET)

bin:
	mkdir -p bin

obj:
	mkdir -p obj $(patsubst %/,%,$(dir $(OBJ)))

$(TARGET): $(OBJ) | bin
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

obj/%.o: %.c | obj
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf obj bin

.PHONY: all clean bin obj
