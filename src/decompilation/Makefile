# Makefile for recursively compiling C source files
# 

# Compiler and flags
CC := clang
CFLAGS := -g -O0

ROOT_DIR := /home/xiachunwei/Dataset/decompilation-dataset/AnghaBench-Sample

# Directories
SRC_DIR := $(ROOT_DIR)
OBJ_DIR := $(ROOT_DIR)-obj-g-O0
ASS_DIR := $(ROOT_DIR)-assembly-g-O0
LL_DIR := $(ROOT_DIR)-ll-g-O0


# Source files
SRCS := $(shell find $(SRC_DIR) -name '*.c')
OBJS := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))
ASSEMBLYS := $(patsubst $(SRC_DIR)/%.c,$(ASS_DIR)/%.s,$(SRCS))
LLS := $(patsubst $(SRC_DIR)/%.c,$(LL_DIR)/%.ll,$(SRCS))

# Targets
all: $(ASSEMBLYS)

# .PRECIOUS: %.o
.SECONDARY: $(OBJS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@


$(ASS_DIR)/%.s: $(OBJ_DIR)/%.o
	@mkdir -p $(dir $@)
	llvm-objdump -d -S --no-show-raw-insn $< > $@

.PHONY: all
