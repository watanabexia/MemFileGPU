#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>

//Debug
#include <stdlib.h>
#include <stdio.h>

__device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results,
                             int input_size) {

    // Debug
    for (int i = 0; i < input_size; i++) {

        //Debug
        //printf("Writing %d-th entry...\n", i);

        vm_write(vm, i, input[i]);
  }

    for (int i = input_size - 1; i >= input_size - 32769; i--) {

        //Debug
        //printf("Reading %d-th entry...\n", i);

        int value = vm_read(vm, i);
  }

    //Debug
    //int head = vm->invert_page_table[4000];
    //printf("head = %d\n", head);
    //printf("tail page_number = %d\n", vm->invert_page_table[2048]);
    //printf("head page_number = %d\n", vm->invert_page_table[head-1]);

  vm_snapshot(vm, results, 0, input_size);
}
