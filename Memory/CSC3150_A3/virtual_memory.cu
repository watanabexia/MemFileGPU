#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>

//Debug
#include <stdlib.h>
#include <stdio.h>

__device__ void init_invert_page_table(VirtualMemory *vm) {

  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    vm->invert_page_table[i] = 0x80000000; // invalid := MSB is 1
    vm->invert_page_table[i + vm->PAGE_ENTRIES] = i;
  }

  vm->invert_page_table[4000] = 2048; // LRU Head Index
}

__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *invert_page_table, int *pagefault_num_ptr,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES) {
  // init variables
  vm->buffer = buffer;
  vm->storage = storage;
  vm->invert_page_table = invert_page_table;
  vm->pagefault_num_ptr = pagefault_num_ptr;

  // init constants
  vm->PAGESIZE = PAGESIZE;
  vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
  vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
  vm->STORAGE_SIZE = STORAGE_SIZE;
  vm->PAGE_ENTRIES = PAGE_ENTRIES;

  // before first vm_write or vm_read
  init_invert_page_table(vm);
}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {
  /* Complate vm_read function to read single element from data buffer */
    int PAGESIZE = vm->PAGESIZE;
    int PAGE_ENTRIES = vm->PAGE_ENTRIES;
    int page_number = addr / PAGESIZE;
    int page_offset = addr % PAGESIZE;
    bool page_hit = false;
    int frame_number = -1;
    bool empty_hit = false;
    int empty_frame_number = -1;
    int tail = 2048;
    int LRU_page_number = vm->invert_page_table[tail];
    int LRU_frame_number = -1;
    uchar value = -1;

    for (int i = PAGE_ENTRIES; i < 2*PAGE_ENTRIES; i++) {
        int j = i - PAGE_ENTRIES;
        int frame_status = vm->invert_page_table[j];

        if (empty_hit == false) {
            if (frame_status == 0x80000000) {
                empty_hit = true;
                empty_frame_number = j;
            }
        }

        if (vm->invert_page_table[i] == LRU_page_number) {
            if (frame_status == 0x00000000) {
                LRU_frame_number = j;
            }
        }

        if (vm->invert_page_table[i] == page_number) {
            if (frame_status == 0x00000000) {
                page_hit = true;
                frame_number = j;
                break;
            }
        }
    }

    if (page_hit) {

        //Debug
        //printf("page_hit\n");

        int phy_addr = frame_number * PAGESIZE + page_offset;
        value = vm->buffer[phy_addr];

        int head = vm->invert_page_table[4000];
        bool LRU_hit = false;
        int LRUIndex = -1;
        for (int i = head - 1; i >= 2048; i--) {
            if (vm->invert_page_table[i] == page_number) {
                LRU_hit = true;
                LRUIndex = i;
                break;
            }
        }

        if (LRU_hit) {
            for (int i = LRUIndex; i < head - 1; i++) {
                vm->invert_page_table[i] = vm->invert_page_table[i+1];
            }
            vm->invert_page_table[head - 1] = page_number;
        }
    }
    else {

        //Debug
        //printf("PAGE_MISS\n");

        *(vm->pagefault_num_ptr) += 1;

        if (empty_hit) {
            int vis_addr_page = page_number * PAGESIZE;
            int phy_addr_frame = empty_frame_number * PAGESIZE;
            for (int i = 0; i < PAGESIZE; i++) {
                vm->buffer[phy_addr_frame + i] = vm->storage[vis_addr_page + i];
            }
            int phy_addr = empty_frame_number * PAGESIZE + page_offset;
            value = vm->buffer[phy_addr];

            vm->invert_page_table[empty_frame_number] = 0x00000000;
            vm->invert_page_table[empty_frame_number + PAGE_ENTRIES] = page_number;

            int head = vm->invert_page_table[4000];
            vm->invert_page_table[head] = page_number;
            vm->invert_page_table[4000] += 1;
        }
        else {
            int vis_addr_page = LRU_page_number * PAGESIZE;
            int phy_addr_frame = LRU_frame_number * PAGESIZE;
            for (int i = 0; i < PAGESIZE; i++) {
                vm->storage[vis_addr_page + i] = vm->buffer[phy_addr_frame + i];
            }
            vis_addr_page = page_number * PAGESIZE;
            for (int i = 0; i < PAGESIZE; i++) {
                vm->buffer[phy_addr_frame + i] = vm->storage[vis_addr_page + i];
            }
            int phy_addr = LRU_frame_number * PAGESIZE + page_offset;
            value = vm->buffer[phy_addr];

            vm->invert_page_table[LRU_frame_number + PAGE_ENTRIES] = page_number;

            int head = vm->invert_page_table[4000];

            for (int i = tail; i < head - 1; i++) {
                vm->invert_page_table[i] = vm->invert_page_table[i + 1];
            }
            vm->invert_page_table[head - 1] = page_number;
        }
    }
  return value; //TODO
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
  /* Complete vm_write function to write value into data buffer */
    int PAGESIZE = vm->PAGESIZE;
    int PAGE_ENTRIES = vm->PAGE_ENTRIES;
    int page_number = addr / PAGESIZE;
    int page_offset = addr % PAGESIZE;
    bool page_hit = false;
    int frame_number = -1;
    bool empty_hit = false;
    int empty_frame_number = -1;
    int tail = 2048;
    int LRU_page_number = vm->invert_page_table[tail];
    int LRU_frame_number = -1;

    for (int i = PAGE_ENTRIES; i < 2*PAGE_ENTRIES; i++) {
        int j = i - PAGE_ENTRIES;
        int frame_status = vm->invert_page_table[j];

        if (empty_hit == false) {
            if (frame_status == 0x80000000) {
                empty_hit = true;
                empty_frame_number = j;
            }
        }

        if (vm->invert_page_table[i] == LRU_page_number) {
            if (frame_status == 0x00000000) {
                LRU_frame_number = j;
            }
        }

        if (vm->invert_page_table[i] == page_number) {
            if (frame_status == 0x00000000) {
                page_hit = true;
                frame_number = j;
                break;
            }
        }
    }

    if (page_hit) {

        //Debug
        //printf("page_hit\n");

        int phy_addr = frame_number * PAGESIZE + page_offset;
        vm->buffer[phy_addr] = value;

        int head = vm->invert_page_table[4000];
        bool LRU_hit = false;
        int LRUIndex = -1;
        for (int i = head - 1; i >= 2048; i--) {
            if (vm->invert_page_table[i] == page_number) {
                LRU_hit = true;
                LRUIndex = i;
                break;
            }
        }

        if (LRU_hit) {
            for (int i = LRUIndex; i < head - 1; i++) {
                vm->invert_page_table[i] = vm->invert_page_table[i+1];
            }
            vm->invert_page_table[head - 1] = page_number;
        }
    }
    else {

        //Debug
        //printf("PAGE_MISS\n");

        *(vm->pagefault_num_ptr) += 1;

        if (empty_hit) {
            int vis_addr_page = page_number * PAGESIZE;
            int phy_addr_frame = empty_frame_number * PAGESIZE;
            for (int i = 0; i < PAGESIZE; i++) {
                vm->buffer[phy_addr_frame + i] = vm->storage[vis_addr_page + i];
            }
            int phy_addr = empty_frame_number * PAGESIZE + page_offset;
            vm->buffer[phy_addr] = value;

            vm->invert_page_table[empty_frame_number] = 0x00000000;
            vm->invert_page_table[empty_frame_number + PAGE_ENTRIES] = page_number;

            int head = vm->invert_page_table[4000];
            vm->invert_page_table[head] = page_number;
            vm->invert_page_table[4000] += 1;
        }
        else {
            int vis_addr_page = LRU_page_number * PAGESIZE;
            int phy_addr_frame = LRU_frame_number * PAGESIZE;
            for (int i = 0; i < PAGESIZE; i++) {
                vm->storage[vis_addr_page + i] = vm->buffer[phy_addr_frame + i];
            }
            vis_addr_page = page_number * PAGESIZE;
            for (int i = 0; i < PAGESIZE; i++) {
                vm->buffer[phy_addr_frame + i] = vm->storage[vis_addr_page + i];
            }
            int phy_addr = LRU_frame_number * PAGESIZE + page_offset;
            vm->buffer[phy_addr] = value;

            vm->invert_page_table[LRU_frame_number + PAGE_ENTRIES] = page_number;

            int head = vm->invert_page_table[4000];

            for (int i = tail; i < head - 1; i++) {
                vm->invert_page_table[i] = vm->invert_page_table[i + 1];
            }
            vm->invert_page_table[head - 1] = page_number;
        }
    }

}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size) {
  /* Complete snapshot function togther with vm_read to load elements from data
   * to result buffer */
    for (int i = offset; i < input_size; i++) {
        results[i] = vm_read(vm, i);
    }
}

