#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__device__ __managed__ u32 gtime = 0;


__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
  // init variables
  fs->volume = volume;

  // init constants
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
  fs->FCB_SIZE = FCB_SIZE;
  fs->FCB_ENTRIES = FCB_ENTRIES;
  fs->STORAGE_SIZE = VOLUME_SIZE;
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  fs->MAX_FILE_NUM = MAX_FILE_NUM;
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;

  // init SUPER BLOCK (ADDR # 0-4095) (BLOCK # 0-127)
  for (int i = SUPER_BASE_ADDRESS; i < (SUPER_BASE_ADDRESS + fs->SUPERBLOCK_SIZE); i++) {
	  fs->volume[i] = 0b11111111; // 1 <==> free block
  }

  // init FCB (ADDR # 4096-36863) (BLOCK # 128-1151)
  // 1024 files (maximum) * 32bytes = 32KB
  // inside each FCB ENTRY:
	// 0-3: modified time				 (u32) (range: 0-1023) smallest <==> most recent, 1024 <==> not exist.
	// 4-7: file size					 (u32) (range: 0-1048576)
	// 8-11: file head addr				 (u32) (range: 36864-1085439) (block range: 1152-33919)
	// 12-31: file name					 (uchar[20]) ends with '\0'

  for (int i = 0; i < (fs->MAX_FILE_NUM); i++) {
	  * (u32*) (&(fs->volume[FCB_BASE_ADDRESS + (i * 32)])) = 1024;
  }

  // init Contents of the files (ADDR # 36864-1085439) (BLOCK # 1152-33919)
}


__device__ bool strcmp(char* s1, char* s2) {
	for (int i = 0; i < 20; i++) {
		if (s1[i] != s2[i]) {
			return false;
		}
		else {
			if (s1[i] == '\0') {
				return true;
			}
		}
	}
}


__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
	/* Implement open operation here */

	//Debug
	//printf("Opening file...\n");

	// Search for the file with the file name
	bool first_ava_found = false;
	u32 first_ava_FCB_ENTRY_ADDR = 0;

	for (int i = 0; i < (fs->MAX_FILE_NUM); i++) {
		u32 FCB_ENTRY_ADDR = FCB_BASE_ADDRESS + (i * 32);
		u32 Modi_t = *(u32*)(&(fs->volume[FCB_ENTRY_ADDR]));
		if (Modi_t != 1024) {
			int j = FCB_ENTRY_ADDR + 12;
			char f_name_ch = *(char*)(&(fs->volume[j]));
			int k = 0;
			char f_name[20];
			while (f_name_ch != '\0') {
				f_name[k] = f_name_ch;
				k++;
				j++;
				f_name_ch = *(char*)(&(fs->volume[j]));
			}
			f_name[k] = f_name_ch;

			if (strcmp(f_name, s) == true) {
				return FCB_ENTRY_ADDR;
			}
		}
		else {
			if (first_ava_found == false) {
				first_ava_FCB_ENTRY_ADDR = FCB_ENTRY_ADDR;
				first_ava_found = true;
			}
		}
	}

	if (first_ava_found) { // NOT EXIST and EMPTY available

		//Debug
		//printf("Creating new file at %lu...\n", first_ava_FCB_ENTRY_ADDR);

		// update Modi_t
		for (int i = 0; i < (fs->MAX_FILE_NUM); i++) {
			int FCB_ENTRY_ADDR = FCB_BASE_ADDRESS + (i * (fs->FCB_SIZE));
			u32 Modi_t = *(u32*)(&(fs->volume[FCB_ENTRY_ADDR]));
			if (Modi_t != 1024) {

				//Debug
				//printf("Adding Modi_t at %lu\n", FCB_ENTRY_ADDR);

				*(u32*)(&(fs->volume[FCB_ENTRY_ADDR])) += 1;
			}
		}

		*(u32*)(&(fs->volume[first_ava_FCB_ENTRY_ADDR])) = 0;   // modified time
		*(u32*)(&(fs->volume[first_ava_FCB_ENTRY_ADDR+4])) = 0; // size

		// file name
		int k = 0;
		while (s[k] != '\0') {
			*(char*)(&(fs->volume[first_ava_FCB_ENTRY_ADDR + 12 + k])) = s[k];
			k++;
		}
		*(char*)(&(fs->volume[first_ava_FCB_ENTRY_ADDR + 12 + k])) = s[k];

		return first_ava_FCB_ENTRY_ADDR;
	}
}


__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{

	//Debug
	//printf("Reading data from %lu...\n", fp);

	/* Implement read operation here */
	u32 FILE_Modi_t = *(u32*)(&(fs->volume[fp]));
	// update Modi_t
	for (int i = 0; i < (fs->MAX_FILE_NUM); i++) {
		int FCB_ENTRY_ADDR = FCB_BASE_ADDRESS + (i * (fs->FCB_SIZE));
		u32 Modi_t = *(u32*)(&(fs->volume[FCB_ENTRY_ADDR]));
		if (Modi_t < FILE_Modi_t) {
			*(u32*)(&(fs->volume[FCB_ENTRY_ADDR])) += 1;
		}
	}
	*(u32*)(&(fs->volume[fp])) = 0;

	u32 FILE_BASE_ADDR = *(u32*)(&(fs->volume[fp + 8]));
	for (int i = 0; i < size; i++) {
		output[i] = fs->volume[FILE_BASE_ADDR + i];
	}
}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
	//Debug
	//printf("Writing data to %lu...\n", fp);

	/* Implement write operation here */
	u32 FILE_CUUR_SIZE = *(u32*)(&(fs->volume[fp + 4]));

	u32 FILE_BLOK_SIZE = size / (fs->STORAGE_BLOCK_SIZE);
	if (size % (fs->STORAGE_BLOCK_SIZE) != 0) {
		FILE_BLOK_SIZE += 1;
	}

	if (FILE_CUUR_SIZE == 0) {

		//Debug
		//printf("This is an empty file. Writing to the end...\n");

		for (int i = 0; i < (fs->SUPERBLOCK_SIZE); i++) { // search for the first empty block
			uchar Bit_Map_8Blocks = fs->volume[i];

			if (Bit_Map_8Blocks != 0b00000000) {
				u32 FILE_BLOK_ADDR = i * 8;
				while (Bit_Map_8Blocks < 0b10000000) {
					FILE_BLOK_ADDR += 1;
					Bit_Map_8Blocks <<= 1;
				}

				//Debug
				//printf("The first empty block location is %lu\n", FILE_BLOK_ADDR);

				// REconstruct superblock
				int TOTAL_BLOK_SIZE = FILE_BLOK_ADDR + FILE_BLOK_SIZE;
				int j = 0;
				while (TOTAL_BLOK_SIZE >= 8) {
					fs->volume[j] = 0b00000000;
					TOTAL_BLOK_SIZE /= 8;
					j++;
				}
				if (TOTAL_BLOK_SIZE > 0) {
					fs->volume[j] = 0b01111111;
					TOTAL_BLOK_SIZE -= 1;
					while (TOTAL_BLOK_SIZE > 0) {
						(fs->volume[j]) >>= 1;
						TOTAL_BLOK_SIZE -= 1;
					}
				}
				u32 FILE_ADDR = (fs->FILE_BASE_ADDRESS) + (FILE_BLOK_ADDR * 32);
				*(u32*)(&(fs->volume[fp + 4])) = size;
				*(u32*)(&(fs->volume[fp + 8])) = FILE_ADDR;
				break;
			}
		}
	}
	else {

		int FILE_CUUR_BLOK_SIZE = FILE_CUUR_SIZE / (fs->STORAGE_BLOCK_SIZE);
		if (FILE_CUUR_SIZE % (fs->STORAGE_BLOCK_SIZE) != 0) {
			FILE_CUUR_BLOK_SIZE += 1;
		}

		u32 FILE_CUUR_ADDR = *(u32*)(&(fs->volume[fp + 8]));

		int FILE_CUUR_BLOK_ADDR = (FILE_CUUR_ADDR - (fs->FILE_BASE_ADDRESS)) / 32;

		if (FILE_CUUR_BLOK_SIZE != FILE_BLOK_SIZE) {

			//Debug
			//printf("The new file block size is not consistant.\n");

			int NEXT_BLOK_ADDR = FILE_CUUR_BLOK_ADDR + FILE_CUUR_BLOK_SIZE;
			int NEXT_ADDR = (fs->FILE_BASE_ADDRESS) + NEXT_BLOK_ADDR * 32;

			// search for the first empty block
			for (int i = 0; i < (fs->SUPERBLOCK_SIZE); i++) { 
				uchar Bit_Map_8Blocks = fs->volume[i];
				if (Bit_Map_8Blocks != 0b00000000) {
					u32 FILE_BLOK_ADDR = i * 8;
					while (Bit_Map_8Blocks < 0b10000000) {
						FILE_BLOK_ADDR += 1;
						Bit_Map_8Blocks <<= 1;
					}

					//Debug
					//printf("The first empty block location is %lu\n", FILE_BLOK_ADDR);
					
					// Move back contents
					//Debug
					//printf("Moving Contents...\n");
					int END_FILE_ADDR = ((fs->FILE_BASE_ADDRESS) + FILE_BLOK_ADDR*32) - 1;

					//Debug
					//printf("NEXT_ADDR %d\n", NEXT_ADDR);
					//printf("END_FILE_ADDR %d\n", END_FILE_ADDR);

					for (int j = 0; j < END_FILE_ADDR - NEXT_ADDR + 1; j++) {
						fs->volume[FILE_CUUR_ADDR + j] = fs->volume[NEXT_ADDR + j];
					}

					// Update file addr
					//Debug
					//printf("Update file ADDR\n");
					int offset = FILE_CUUR_BLOK_SIZE * 32;
					for (int j = 0; j < (fs->MAX_FILE_NUM); j++) {
						int FCB_ENTRY_ADDR = FCB_BASE_ADDRESS + (j * (fs->FCB_SIZE));
						u32 file_addr = *(u32*)(&(fs->volume[FCB_ENTRY_ADDR + 8]));
						if (file_addr > FILE_CUUR_ADDR) {
							*(u32*)(&(fs->volume[FCB_ENTRY_ADDR + 8])) -= offset;
						}
					}

					// REconstruct superblock
					for (int j = SUPER_BASE_ADDRESS; j < (SUPER_BASE_ADDRESS + fs->SUPERBLOCK_SIZE); j++) {
						fs->volume[j] = 0b11111111;
					}
					int TOTAL_BLOK_SIZE = FILE_BLOK_ADDR - FILE_CUUR_BLOK_SIZE + FILE_BLOK_SIZE;
					int j = 0;
					while (TOTAL_BLOK_SIZE >= 8) {
						fs->volume[j] = 0b00000000;
						TOTAL_BLOK_SIZE /= 8;
						j++;
					}
					if (TOTAL_BLOK_SIZE > 0) {
						fs->volume[j] = 0b01111111;
						TOTAL_BLOK_SIZE -= 1;
						while (TOTAL_BLOK_SIZE > 0) {
							(fs->volume[j]) >>= 1;
							TOTAL_BLOK_SIZE -= 1;
						}
					}

					FILE_BLOK_ADDR = FILE_BLOK_ADDR - FILE_CUUR_BLOK_SIZE;
					u32 FILE_ADDR = (fs->FILE_BASE_ADDRESS) + (FILE_BLOK_ADDR * 32);
					*(u32*)(&(fs->volume[fp + 4])) = size;
					*(u32*)(&(fs->volume[fp + 8])) = FILE_ADDR;
					break;
				}
			}
		}
		else {
			*(u32*)(&(fs->volume[fp + 4])) = size;
		}
	}

	u32 FILE_Modi_t = *(u32*)(&(fs->volume[fp]));
	// update Modi_t
	for (int i = 0; i < (fs->MAX_FILE_NUM); i++) {
		int FCB_ENTRY_ADDR = FCB_BASE_ADDRESS + (i * (fs->FCB_SIZE));
		u32 Modi_t = *(u32*)(&(fs->volume[FCB_ENTRY_ADDR]));
		if (Modi_t < FILE_Modi_t) {
			*(u32*)(&(fs->volume[FCB_ENTRY_ADDR])) += 1;
		}
	}
	*(u32*)(&(fs->volume[fp])) = 0;

	u32 FILE_BASE_ADDR = *(u32*)(&(fs->volume[fp + 8]));
	for (int i = 0; i < size; i++) {
		fs->volume[FILE_BASE_ADDR + i] = input[i];
	}
}
__device__ void fs_gsys(FileSystem *fs, int op)
{
	//Debug
	//printf("Sorting files...\n");

	/* Implement LS_D and LS_S operation here */
	int file_num = 1024;
	for (int i = 0; i < (fs->MAX_FILE_NUM); i++) {
		u32 FCB_ENTRY_ADDR = FCB_BASE_ADDRESS + (i * 32);
		u32 Modi_t = *(u32*)(&(fs->volume[FCB_ENTRY_ADDR]));
		if (Modi_t == 1024) {
			file_num = i;
			break;
		}
	}

	int no_list[1024];
	for (int i = 0; i < file_num; i++) {
		no_list[i] = i;
	}

	if (op == LS_D) {
		printf("===sort by modified time===\n");
		u32 modi_list[1024];
		for (int i = 0; i < file_num; i++) {
			u32 FCB_ENTRY_ADDR = FCB_BASE_ADDRESS + (i * 32);
			u32 Modi_t = *(u32*)(&(fs->volume[FCB_ENTRY_ADDR]));
			modi_list[i] = Modi_t;
		}

		for (int i = 0; i < file_num; i++) {
			for (int j = 0; j < file_num - i - 1; j++) {
				if (modi_list[j] > modi_list[j + 1]) {
					int interlude = modi_list[j];
					modi_list[j] = modi_list[j + 1];
					modi_list[j + 1] = interlude;
					interlude = no_list[j];
					no_list[j] = no_list[j + 1];
					no_list[j + 1] = interlude;
				}
			}
		}

		for (int i = 0; i < file_num; i++) {
			u32 FCB_ENTRY_ADDR = FCB_BASE_ADDRESS + (no_list[i] * 32);
			int k = FCB_ENTRY_ADDR + 12;
			char* cptr = (char*)(&(fs->volume[k]));
			while (*cptr != '\0') {
				printf("%c", *cptr);
				cptr++;
			}
			printf("\n");
		}
	}
	else {
		if (op == LS_S) {

			//Debug
			//printf("file_num %d\n", file_num);

			printf("===sort by file size===\n");
			u32 size_list[1024];
			for (int i = 0; i < file_num; i++) {
				u32 FCB_ENTRY_ADDR = FCB_BASE_ADDRESS + (i * 32);
				u32 size = *(u32*)(&(fs->volume[FCB_ENTRY_ADDR + 4]));
				size_list[i] = size;
			}

			for (int i = 0; i < file_num; i++) {
				for (int j = 0; j < file_num - i - 1; j++) {
					if (size_list[j] < size_list[j + 1]) {
						int interlude = size_list[j];
						size_list[j] = size_list[j + 1];
						size_list[j + 1] = interlude;
						interlude = no_list[j];
						no_list[j] = no_list[j + 1];
						no_list[j + 1] = interlude;
					}
				}
			}

			for (int i = 0; i < file_num; i++) {
				u32 FCB_ENTRY_ADDR = FCB_BASE_ADDRESS + (no_list[i] * 32);
				int k = FCB_ENTRY_ADDR + 12;
				char* cptr = (char*)(&(fs->volume[k]));
				while (*cptr != '\0') {
					printf("%c", *cptr);
					cptr++;
				}
				printf(" %d\n", size_list[i]);
			}

			//Debug
			//printf("flie_num: %d\n", file_num);
		}
	}
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	/* Implement rm operation here */
	if (op == RM) {

		//Debug
		//printf("Removing file...\n");

		// Search for the file with the file name
		for (int i = 0; i < (fs->MAX_FILE_NUM); i++) {
			u32 FCB_ENTRY_ADDR = FCB_BASE_ADDRESS + (i * 32);
			u32 Modi_t = *(u32*)(&(fs->volume[FCB_ENTRY_ADDR]));
			if (Modi_t != 1024) {
				int j = FCB_ENTRY_ADDR + 12;
				char* cptr = (char*)(&(fs->volume[j]));
				if (strcmp(cptr, s) == true) {

					//Debug
					//printf("Objective file founded.\n");

					u32 fp = FCB_ENTRY_ADDR;
					u32 FILE_Modi_t = *(u32*)(&(fs->volume[fp]));
					u32 FILE_CUUR_SIZE = *(u32*)(&(fs->volume[fp + 4]));
					u32 FILE_CUUR_ADDR = *(u32*)(&(fs->volume[fp + 8]));

					// update Modi_t
					for (int j = 0; j < (fs->MAX_FILE_NUM); j++) {
						int FCB_ENTRY_ADDR = FCB_BASE_ADDRESS + (j * (fs->FCB_SIZE));
						u32 Modi_t = *(u32*)(&(fs->volume[FCB_ENTRY_ADDR]));
						if ((Modi_t > FILE_Modi_t) && (Modi_t != 1024)) {
							*(u32*)(&(fs->volume[FCB_ENTRY_ADDR])) -= 1;
						}
					}
					*(u32*)(&(fs->volume[fp])) = 1024;

					int FILE_CUUR_BLOK_SIZE = FILE_CUUR_SIZE / (fs->STORAGE_BLOCK_SIZE);
					if (FILE_CUUR_SIZE % (fs->STORAGE_BLOCK_SIZE) != 0) {
						FILE_CUUR_BLOK_SIZE += 1;
					}

					int FILE_CUUR_BLOK_ADDR = (FILE_CUUR_ADDR - (fs->FILE_BASE_ADDRESS)) / 32;

					int NEXT_BLOK_ADDR = FILE_CUUR_BLOK_ADDR + FILE_CUUR_BLOK_SIZE;
					int NEXT_ADDR = (fs->FILE_BASE_ADDRESS) + NEXT_BLOK_ADDR * 32;

					// search for the first empty block
					for (int k = 0; k < (fs->SUPERBLOCK_SIZE); k++) {
						uchar Bit_Map_8Blocks = fs->volume[k];
						if (Bit_Map_8Blocks != 0b00000000) {
							u32 FILE_BLOK_ADDR = k * 8;
							while (Bit_Map_8Blocks < 0b10000000) {
								FILE_BLOK_ADDR += 1;
								Bit_Map_8Blocks <<= 1;
							}

							// Move back contents
							int END_FILE_ADDR = ((fs->FILE_BASE_ADDRESS) + FILE_BLOK_ADDR * 32) - 1;
							for (int j = 0; j < END_FILE_ADDR - NEXT_ADDR + 1; j++) {
								fs->volume[FILE_CUUR_ADDR + j] = fs->volume[NEXT_ADDR + j];
							}

							// Update file addr
							int offset = FILE_CUUR_BLOK_SIZE * 32;
							for (int j = 0; j < (fs->MAX_FILE_NUM); j++) {
								int FCB_ENTRY_ADDR = FCB_BASE_ADDRESS + (j * (fs->FCB_SIZE));
								u32 file_addr = *(u32*)(&(fs->volume[FCB_ENTRY_ADDR + 8]));
								if (file_addr > FILE_CUUR_ADDR) {
									*(u32*)(&(fs->volume[FCB_ENTRY_ADDR + 8])) -= offset;
								}
							}

							// REconstruct superblock
							for (int j = SUPER_BASE_ADDRESS; j < (SUPER_BASE_ADDRESS + fs->SUPERBLOCK_SIZE); j++) {
								fs->volume[j] = 0b11111111;
							}
							int TOTAL_BLOK_SIZE = FILE_BLOK_ADDR - FILE_CUUR_BLOK_SIZE;
							int j = 0;
							while (TOTAL_BLOK_SIZE >= 8) {
								fs->volume[j] = 0b00000000;
								TOTAL_BLOK_SIZE /= 8;
								j++;
							}
							if (TOTAL_BLOK_SIZE > 0) {
								fs->volume[j] = 0b01111111;
								TOTAL_BLOK_SIZE -= 1;
								while (TOTAL_BLOK_SIZE > 0) {
									(fs->volume[j]) >>= 1;
									TOTAL_BLOK_SIZE -= 1;
								}
							}
							break;
						}
					}

					//Debug
					//printf("HERE.\n");

					// eliminate RMed PCB
					for (int j = i; j < (fs->MAX_FILE_NUM) - 1; j++) {
						u32 x_FCB_ENTRY_ADDR = FCB_BASE_ADDRESS + (j * 32);
						u32 n_FCB_ENTRY_ADDR = FCB_BASE_ADDRESS + ((j + 1) * 32);
						u32 x_Modi_t = *(u32*)(&(fs->volume[x_FCB_ENTRY_ADDR]));
						if ((x_Modi_t != 1024) || (j == i)) {
							*(u32*)(&(fs->volume[x_FCB_ENTRY_ADDR])) = *(u32*)(&(fs->volume[n_FCB_ENTRY_ADDR]));
							*(u32*)(&(fs->volume[x_FCB_ENTRY_ADDR + 4])) = *(u32*)(&(fs->volume[n_FCB_ENTRY_ADDR + 4]));
							*(u32*)(&(fs->volume[x_FCB_ENTRY_ADDR + 8])) = *(u32*)(&(fs->volume[n_FCB_ENTRY_ADDR + 8]));
							char* xcptr = (char*)(&(fs->volume[x_FCB_ENTRY_ADDR + 12]));
							char* ncptr = (char*)(&(fs->volume[n_FCB_ENTRY_ADDR + 12]));
							int o = 0;
							while (*ncptr != '\0') {
								*xcptr = *ncptr;
								ncptr++;
								xcptr++;
							}
							*xcptr = *ncptr;
						}
						else {
							break;
						}
					}
					*(u32*)(&(fs->volume[FCB_BASE_ADDRESS + (((fs->MAX_FILE_NUM) - 1) * 32)])) = 1024;
					break;
				}
			}
		}
	}
}
