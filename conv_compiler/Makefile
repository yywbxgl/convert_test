.PHONY : all clean
EXE = conv-split
SRC = $(wildcard *.c)
OBJ = $(SRC:%.c=%.o)
DEP = $(SRC:%.c=%.d)
CC = gcc
CFLAGS = -Wall -I include -O2 -MMD
LDFLAGS = -s
all: $(EXE)
$(EXE) : $(OBJ)
	$(CC) $(LDFLAGS) $^ -o $@
clean:
	$(RM) $(wildcard *.o *.d $(EXE))
-include $(DEP)
