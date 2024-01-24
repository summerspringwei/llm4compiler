# Makefile for recursively compiling C source files
# 

# Compiler and flags
CC := clang-15
CFLAGS := -g -O2
DIS := llvm-objdump-15
# CFLAGS := -g -O2

ROOT_DIR := /data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench

# Directories
SRC_DIR := $(ROOT_DIR)
OBJ_DIR := $(ROOT_DIR)-obj-g-O2
ASS_DIR := $(ROOT_DIR)-assembly-g-O2
LL_DIR := $(ROOT_DIR)-ll-g-O2

DIS_FLAGS := -d --no-show-raw-insn

# Source files
SRCS := $(shell find $(SRC_DIR) -name '*.c')
OBJS := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))
ASSEMBLYS := $(patsubst $(SRC_DIR)/%.c,$(ASS_DIR)/%.s,$(SRCS))
LLS := $(patsubst $(SRC_DIR)/%.c,$(LL_DIR)/%.ll,$(SRCS))

TMP := $(shell mkdir -p $(OBJ_DIR))
TMP := $(shell mkdir -p $(ASS_DIR))
TMP := $(shell mkdir -p $(LL_DIR))

$(info $$LLS is [${LLS}])

# Targets
all: $(ASSEMBLYS) $(LLS)

# .PRECIOUS: %.o
.SECONDARY: $(OBJS) $(LLS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@


$(LL_DIR)/%.ll: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(info $$@ is [$@])
	$(CC) $(CFLAGS) -c $< -mllvm -print-after=verify 2> $@


$(ASS_DIR)/%.s: $(OBJ_DIR)/%.o
	@mkdir -p $(dir $@)
	$(DIS) $(DIS_FLAGS)  $< > $@

.PHONY: all
# clean:
# 	rm $(ASSEMBLYS) $(LLS)
