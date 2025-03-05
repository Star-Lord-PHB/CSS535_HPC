BUILD_MODE = debug 
BLOCK_COUNT = 1024
THREAD_COUNT = 1024
CORRECTNESS_CHECK = false
PRINTING = false
JSON = false 

# compilers 
CPP_COMPILOR = cl
C_COMPILOR = cl
NV_COMPILOR = nvcc

# project structure 
INCLUDE_PATH = include
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# CBLAS_INCLUDE_PATH = /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers
# CBLAS_LIB_LDFLAG = -lblas
INCLUDE_FLAG = /I$(INCLUDE_PATH)
MSVC_INCLUDE_FLAG = /I"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.38.33130\include" /I"C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\um" /I"C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\ucrt"
CUDA_INCLUDE_FLAG = /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\include"

MSVC_LIB_LDFLAGS = /LIBPATH:"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.38.33130\lib\x64" /LIBPATH:"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64" /LIBPATH:"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\ucrt\x64"
CUDA_LIB_LDFLAGS = /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64" cudart.lib

COMPILOR_FLAGS += $(INCLUDE_FLAG) $(CUDA_INCLUDE_FLAG) $(MSVC_INCLUDE_FLAG) /std:c++17 /nologo /W3 /DTHREAD_COUNT=$(THREAD_COUNT) /DBLOCK_COUNT=$(BLOCK_COUNT) /DJSON=$(JSON) /DCORRECTNESS_CHECK=$(CORRECTNESS_CHECK) /DPRINTING=$(PRINTING)
NVCCFLAGS += -std=c++17 -I$(INCLUDE_PATH) -DTHREAD_COUNT=$(THREAD_COUNT) -DBLOCK_COUNT=$(BLOCK_COUNT) -DJSON=$(JSON) -DCORRECTNESS_CHECK=$(CORRECTNESS_CHECK) -DPRINTING=$(PRINTING)
LDFLAGS = $(CBLAS_LIB_LDFLAG) $(MSVC_LIB_LDFLAGS) $(CUDA_LIB_LDFLAGS)

ifeq ($(BUILD_MODE), debug)
	COMPILOR_FLAGS += /g /Od /Wall 
	NVCCFLAGS += -g -G
else
	COMPILOR_FLAGS += /Ox
	NVCCFLAGS += -O2
endif


C_SRCS = $(wildcard $(SRC_DIR)/*.c)
CPP_SRCS = $(wildcard $(SRC_DIR)/*.cpp)
CUDA_SRCS = $(wildcard $(SRC_DIR)/*.cu)

OBJS = $(C_SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.obj) $(CPP_SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.obj) $(CUDA_SRCS:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.obj)
DEPS = $(OBJS:.obj=.d)
EXEC = $(BIN_DIR)/program.exe

$(shell mkdir $(OBJ_DIR) $(BIN_DIR) >nul 2>nul)

# Main target
all: $(EXEC)

# Linking
$(EXEC): $(OBJS)
	$(CPP_COMPILOR) $(OBJS) /link /out:$(EXEC) $(LDFLAGS) 

# Compilation
$(OBJ_DIR)/%.obj: $(SRC_DIR)/%.c
	$(C_COMPILOR) $(COMPILOR_FLAGS) /MP /c $< /Fo$@ 

$(OBJ_DIR)/%.obj: $(SRC_DIR)/%.cpp
	$(CPP_COMPILOR) $(COMPILOR_FLAGS) /MP /c $< /Fo$@ 

$(OBJ_DIR)/%.obj: $(SRC_DIR)/%.cu
	$(NV_COMPILOR) $(NVCCFLAGS) -M $< -MF $(basename $@).d 
	$(NV_COMPILOR) $(NVCCFLAGS) -c $< -o $@ 

-include $(DEPS)

build: all 

run: $(EXEC)
	./$(EXEC)

# Clean
clean:
	del /Q /F $(OBJ_DIR)\* $(BIN_DIR)\*

run: all 
	./$(EXEC)

.PHONY: all clean	