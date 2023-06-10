# MemFileGPU
*This repository was part of the assignment 3 & 4 in CSC3150 (Winter 2021) in CUHKsz.*
 Memory, file system simulation on GPU devices

# Memory
## Program Design
For vm_write() and vm_read(), we can know the page number and the off- set of the desired memory when getting virtual memory address addr using addr/PAGESIZE and addr%PAGESIZE.
First, search for the desired page in the invert_page_table using linear search.
(a) When we cannot find the desired page and the physical memory is not full, load the page from the secondary storage to the unused frame.
(b) When we cannot find the desired page and the physical memory is full, replace the LRU page with the loaded desired page from the secondary storage to the unused frame.
Write to/read from the desired memory.
Update the LRU table according to the changes we made. Noted that the LRU algorithm implemented in the program is very simple.
I use `invert_page_table[2*PAGE_ENTRIES:3*PAGE_ENTRIES-1]` as a list to store all the available page number in the inquiry order. An index head = `invert_page_table[4000]` starting with the value `2*PAGE_ENTRIES` always points at the head of the list. The LRU page number would always be `invert_page_table[2*PAGE_ENTRIES]`. Every time a page is requested, it would be added to/moved to the head of the list.
For vm_snapshot(), just use the function vm_read() to read the data stored in between the virtual address offset and offset+input_size to the result buffer.
## Program Execution
1. Use visual studio in a valid version on a CUDA supported machine to compile and run CSC3150_A3.sln. Noted that I did not implement any high level data structures or algorithm, so the running time on my machine is around 3-4 minutes.
## Page Fault Number Explanation
In the first section of user_program.cu, it writes all the data in input buffer to the physical memory in order. This means that one page fault would occur in every 32 bytes of written data. The total number of page fault that would occur in this section would be input_size/32 = 4096.
In the second section, it reads the data in between virtual address `[input_size-32769:input_size-1]` in a reversed order. That’s a total of 32769 bytes of data. However, since we have loaded the last PAGESIZE*PAGE_ENTRIES = 32768 bytes of data in the physical memory, the page fault would only occur when we try to read the data at address input_size-32769. So the total number of page fault that would occur in this section is 1.
In the third section, it reads all the data from secondary storage to the result buffer in order (in my implementation of vm_snapshot() using linear read). The page fault condition in this section is the same as the first section. The total number of page fault that would occur in this section would be input_size/32 = 4096.
In conclusion, the final pagefault number would be 4096+1+4096=8193.
# File System
## Program Design
1. Before implementing the file functions, first initialize the Super Block and FCB. Set all the bits in Super Block to 1(which represent free blocks using Bit- Map), and set the modified time of every FCB entry to 1024(NOT EXISTED).
2. In my implementation, for each FCB entry, (32 bytes in total)
0:3 bytes store modified time and availability as u32. If the value is 1024, it means the file does not exist. If the value is between 0-1023, the value represent the modified time of the file. The smaller the value, the more recent the file has been modified.
4:7 bytes store file size (in bytes) as u32. 8:11 bytes store the file head address as u32.
12:31 bytes store the file name as char[20].
3. For fs_open(), first do a linear search among all FCB entries with the file
name s.
- If the file exists, return the corresponding FCB address as u32.
- If the file does not exist, use the first empty PCB entry to create a new file with zero byte size.
4. Forfs_read(),Update the modified time information of all files. For other file which has a smaller modified time, add 1 to their modified time. For the current file, the modified time is 0.
- Copy the data to the output buffer. 
5. Forfs_write(),
- Update the modified time information like fs_read().
- If the new size requires the same amount of blocks, simply copy the
information from input buffer to the file location.
- Otherwise,
  - Compress the file contents to eliminate the orginal blocks the file was using.
  - Update the file location information in FCB of other files with the same offset (the size of the blocks that were previously used by the file).
  - Place the new file data at the end of the file contents.
  - Update the Super Block information with the new size of used blocks. 6. Forfs_gsys(RM),
- Set the corresponding modified time to be 1024.
- Compress FCB to eliminate the removed FCB entry.
6. For fs_gsys(LS_D/LS_S), sort the file name with respect to file size/modified time with bubble sort algorithm.
## Problems Encountered
### Handle U32/char Data with Uchar Pointer
In the template, we are given an uchar array uchar* volume to locate the infor- mation in the volume. But for the implementation of FCB, we need to store and retrieve some numerical information to and from the volume. For simplicity, I han- dle this problem with type conversion of pointers. To handle u32 data, I make a u32 pointer with (u32*)(fs->volume[ADDRESS]) and simply store/retrieve the data with dereference operand *. Similiar technique is also applicable on char data.
### Data Compation
For simplicity, I implement a simple compation stragety. During the fs_write() process, if the file has redundant/insuﬀicient blocks, I will compact the file blocks and place the new data at the end of all blocks.
### Sort by Created Time
In my implementation of FCB, I do not have extra space for creation time of files. However, in fs_gsys(LS_S), we need to print the file information in the order of creation time. To do so, every time I remove a file, I will do FCB compaction so that the newly created file would always be created at the end of FCB, thus maintaining the file creation order.
## Program Execution
1. Use visual studio in a valid version on a CUDA supported machine to compile and run CSC3150_A4.sln.
