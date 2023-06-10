#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

//Debug
__device__ void watch_FCB(FileSystem *fs, u32 FCB_ADDR) {

	printf("\n");
	printf("### FILE INFO ###\n");
	// file name
	printf("Name: ");
	int k = 0;
	char* cptr = (char*)(&(fs->volume[FCB_ADDR + 12]));
	while (*cptr != '\0') {
		printf("%c", *cptr);
		cptr++;
	}
	printf("\n");

	u32 size = *(u32*)(&(fs->volume[FCB_ADDR + 4]));
	u32 location = *(u32*)(&(fs->volume[FCB_ADDR + 8]));
	u32 chunksize = size / 32;
	if (size % 32 != 0) {
		chunksize += 1;
	}

	printf("Modi_t %lu\n", *(u32*)(&(fs->volume[FCB_ADDR])));   // modified time
	printf("Size %lu\n", size); // size
	printf("Location %lu\n", location);
	printf("Block Location %lu\n", location / 32 - 1152);
	printf("Chunck Size %lu\n", chunksize);

	//contents
	printf("First 5 bytes: ");
	for (int i = 0; i < 5; i++) {
		printf("%c ", fs->volume[location + i]);
	}
	printf("\n");

	printf("#################\n");
	printf("\n");
	
}

__device__ void user_program(FileSystem *fs, uchar *input, uchar *output) {
	
	
	/////////////// Test Case 1  ///////////////

	////Debug
	////printf("TEST STARTED\n");


	//u32 fp = fs_open(fs, "t.txt\0", G_WRITE);
	//fs_write(fs, input, 64, fp);
	//fp = fs_open(fs, "b.txt\0", G_WRITE);
	//fs_write(fs, input + 32, 32, fp);

	////Debug
	////watch_FCB(fs, 4096);
	////watch_FCB(fs, 4128);

	//fp = fs_open(fs, "t.txt\0", G_WRITE);
	//fs_write(fs, input + 32, 32, fp);

	////Debug
	////watch_FCB(fs, 4096);
	////watch_FCB(fs, 4128);

	//fp = fs_open(fs, "t.txt\0", G_READ);
	//fs_read(fs, output, 32, fp);
	//fs_gsys(fs,LS_D);
	//fs_gsys(fs, LS_S);
	//fp = fs_open(fs, "b.txt\0", G_WRITE);
	//fs_write(fs, input + 64, 12, fp);

	////Debug
	////watch_FCB(fs, 4096);
	////watch_FCB(fs, 4128);

	//fs_gsys(fs, LS_S);
	//fs_gsys(fs, LS_D);
	//fs_gsys(fs, RM, "t.txt\0");

	////Debug
	////watch_FCB(fs, 4096);
	////watch_FCB(fs, 4128);

	//fs_gsys(fs, LS_S);

	////Debug
	////watch_FCB(fs, 4096);
	////watch_FCB(fs, 4128);
	////watch_FCB(fs, 4160);


	///////////////// Test Case 2  ///////////////
	//u32 fp = fs_open(fs, "t.txt\0", G_WRITE);
	//fs_write(fs,input, 64, fp);
	//fp = fs_open(fs,"b.txt\0", G_WRITE);
	//fs_write(fs,input + 32, 32, fp);
	//fp = fs_open(fs,"t.txt\0", G_WRITE);
	//fs_write(fs,input + 32, 32, fp);
	//fp = fs_open(fs,"t.txt\0", G_READ);
	//fs_read(fs,output, 32, fp);
	//fs_gsys(fs,LS_D);
	//fs_gsys(fs,LS_S);
	//fp = fs_open(fs,"b.txt\0", G_WRITE);
	//fs_write(fs,input + 64, 12, fp);
	//fs_gsys(fs,LS_S);
	//fs_gsys(fs,LS_D);
	//fs_gsys(fs,RM, "t.txt\0");
	//fs_gsys(fs,LS_S);
	//char fname[10][20];
	//for (int i = 0; i < 10; i++)
	//{
	//	fname[i][0] = i + 33;
	//	for (int j = 1; j < 19; j++)
	//		fname[i][j] = 64 + j;
	//	fname[i][19] = '\0';
	//}

	//for (int i = 0; i < 10; i++)
	//{
	//	fp = fs_open(fs,fname[i], G_WRITE);
	//	fs_write(fs,input + i, 24 + i, fp);
	//}

	//fs_gsys(fs,LS_S);

	//for (int i = 0; i < 5; i++)
	//	fs_gsys(fs,RM, fname[i]);

	//fs_gsys(fs,LS_D);
	

	
	/////////////// Test Case 3  ///////////////
	u32 fp = fs_open(fs, "t.txt\0", G_WRITE);
	fs_write(fs, input, 64, fp);
	fp = fs_open(fs, "b.txt\0", G_WRITE);
	fs_write(fs, input + 32, 32, fp);
	fp = fs_open(fs, "t.txt\0", G_WRITE);
	fs_write(fs, input + 32, 32, fp);
	fp = fs_open(fs, "t.txt\0", G_READ);
	fs_read(fs, output, 32, fp);
	fs_gsys(fs, LS_D);
	fs_gsys(fs, LS_S);
	fp = fs_open(fs, "b.txt\0", G_WRITE);
	fs_write(fs, input + 64, 12, fp);
	fs_gsys(fs, LS_S);
	fs_gsys(fs, LS_D);
	fs_gsys(fs, RM, "t.txt\0");
	fs_gsys(fs, LS_S);

	char fname[10][20];
	for (int i = 0; i < 10; i++)
	{
		fname[i][0] = i + 33;
		for (int j = 1; j < 19; j++)
			fname[i][j] = 64 + j;
		fname[i][19] = '\0';
	}

	for (int i = 0; i < 10; i++)
	{
		fp = fs_open(fs, fname[i], G_WRITE);
		fs_write(fs, input + i, 24 + i, fp);
	}

	fs_gsys(fs, LS_S);

	for (int i = 0; i < 5; i++)
		fs_gsys(fs, RM, fname[i]);

	fs_gsys(fs, LS_D);

	char fname2[1018][20];
	int p = 0;

	for (int k = 2; k < 15; k++)
		for (int i = 50; i <= 126; i++, p++)
		{
			fname2[p][0] = i;
			for (int j = 1; j < k; j++)
				fname2[p][j] = 64 + j;
			fname2[p][k] = '\0';
		}

	for (int i = 0; i < 1001; i++)
	{
		fp = fs_open(fs, fname2[i], G_WRITE);
		fs_write(fs, input + i, 24 + i, fp);
	}

	fs_gsys(fs, LS_S);
	fp = fs_open(fs, fname2[1000], G_READ);
	fs_read(fs, output + 1000, 1024, fp);

	char fname3[17][3];
	for (int i = 0; i < 17; i++)
	{
		fname3[i][0] = 97 + i;
		fname3[i][1] = 97 + i;
		fname3[i][2] = '\0';
		fp = fs_open(fs, fname3[i], G_WRITE);
		fs_write(fs, input + 1024 * i, 1024, fp);
	}

	fp = fs_open(fs, "EA\0", G_WRITE);
	fs_write(fs, input + 1024 * 100, 1024, fp);
	fs_gsys(fs, LS_S);
}
