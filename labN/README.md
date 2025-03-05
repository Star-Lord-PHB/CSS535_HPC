## Compilation

```sh
make clean build BLOCK_COUNT=<block_count> THREAD_COUNT=<thread_count>
# replace <block_count> and <thread_count> with actual number
```

Example: 
```sh
make clean build THREAD_COUNT=1024 BLOCK_COUNT=1024
```

Note: Make sure you have GNUWin32 `make` installed 

If `make` is not installed, here is an alternative command: 

```sh
nvcc -Iinclude .\src\Main.cu -DDBLOCK_COUNT=<block_count> -DTHREAD_COUNT=<thread_count>
# replace <block_count> and <thread_count> with actual number
```

The executable is `./bin/program.exe`

## Specs

GPU: 

* Model: 2080 Super
* Architecture: Turing 
* CUDA Cores: 3072
* Base Clock Speed: 1.65GHz
* Memory: 8 GB GDDR6 256-bit
* CUDA Capability: 7.5
* CUDA Version: V12.6.77

System: 

* Windows 10.0.22621.2506
* Compiler: cl 19.38.33135, nvcc