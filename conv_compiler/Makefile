.PHONY : all clean
EXE = conv-split
SRC = $(wildcard *.c)
CC = gcc
CFLAGS = -I include -O2 -MMD
LDFLAGS = 
all: $(EXE)
$(EXE) : $(SRC:%.c=%.o)
	$(CC) $(LDFLAGS) $^ -o $@
clean:
	$(RM) $(wildcard *.o *.d $(EXE))
-include $(SRC:%.c=%.d)
