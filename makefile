.ONESHELL:# Use one shell per recipie and not one per line
.DELETE_ON_ERROR:# Deletes target file if recipie fails
#MAKEFLAGS += --warn-undefined-variables # NOT working for some reason
#MAKEFLAGS+=--no-builtin-rules# Removes the magic predefined rules.

ifeq ($(origin .RECIPEPREFIX), undefined)
  $(error This Make does not support .RECIPEPREFIX. Please use GNU Make 4.0 or later)
endif

TARGET_EXEC = main
BUILD_DIR = .build
SRC_DIRS = src

CCC=g++
RM=rm
#Linker flags
LDFLAGS=
# MMD Generate .d files as part of compilation.
# MP Generate phony targets for all dependencys. Solves som problems if files are delted
# MT Target name for the make-rule,
# MF Filename for the to-be generated dependency file
DEPFLAGS=-MMD -MP -MT $@
# Allows for multiple src directorys
SRCS = $(shell find $(SRC_DIRS) -name "*.cc" -not -name "tests.cc" -not -name "main.cc")
# will result in uggly .cc.o extension?
OBJS = $(SRCS:%=$(BUILD_DIR)/%.o)
DEPS = $(OBJS:.o=.d)

# Adds the multiple src directorys to the compiler options
INC_DIRS := $(shell find $(SRC_DIRS) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

BUILD_DIRS := $(INC_DIRS:%=$(BUILD_DIR)/%)

# std=c++17 use C++17 standard
# -g, generate information to be used in for example valgrind
# -Wall, inlcude all warnings
# -pedantic, gives more warnings
CFLAGS= $(INC_FLAGS) -std=c++17 -g -Wall -Wextra -pedantic -fmax-errors=5
#.PHONY targets, declares that target is not associated with file and will always
# be run if it is a dependency
.PHONY: clean


# Compiles main target from all .o files.
$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS)
	@echo "hello $@"
	@echo "objs $(OBJS)"
	$(CCC) $(CFLAGS) $(OBJS) $(SRC_DIRS)/main.cc -o $@ $(LDFLAGS)

run: $(BUILD_DIR)/$(TARGET_EXEC)
	./$(BUILD_DIR)/$(TARGET_EXEC)

$(BUILD_DIR)/%.cc.o: %.cc | $(BUILD_DIR)
	@echo "objs $(INC_DIRS)"
	$(CCC) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIRS)


tests: $(OBJS) $(SRC_DIRS)/tests.cc | $(BUILD_DIR)
	$(CCC) $(CFLAGS) -o tests $(SRC_DIRS)/tests.cc $(OBJS)
	./tests

# Creates o-files from cc-files. $* becomes what % is in the pattern. Make will
# automatically add all dependencys from in here from the -dfiles in the include
# statement. The .d files are regenerated with this command as well so that the
# dependency files always reflect the last compiled version. And since the
# comilation process only was dependent on the files listed in the .d file, it
# should be impossible to modify files in any way and have it not be correctly
# recompiled when any of the dependent files change.

# This is here to do nothing if the .d file does not exist.
$(DEPS): # Not neded?

clean:
	$(RM) -r $(BUILD_DIR)
# Appends dependencys to all existing rules. Will be the dependencys at the last
# compilation time, but will be utdated immediatetlly after compilation is done.
# This is not a problem since it does only looks at the dependency files to
# figure out if enough has changed inorder to require recompilation of a certain
# file.
-include $(DEPS)
