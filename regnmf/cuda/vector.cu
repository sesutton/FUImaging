#include <stdio.h>
#include <stdlib.h>

#include "vector.h"

void read_vector_from_array(vector* A, int len, float* value){
	//create vector with all elements from 'value'
	//vector length is len
	//set A->vec_d to NULL

	A->len = len;
	//cudaMallocHost((void**)&(A->mat),sizeof(float)*N); //page-locked memory (faster but limited)
	A->vec = (float*) malloc(sizeof(float) * len);
	A->vec = value;

	if (A->vec_d != NULL)
		cudaFree(A->vec_d);
	A->vec_d = NULL;

	//printf("read float as [%ix%i]\n",A->dim[0],A->dim[1]);
}

void create_vector(vector* A, int len, float value) {
	//create vector with all elements equal to 'value'
	//vector length is len
	//set A->vec_d to NULL

	A->len = len;
	A->vec = (float*) malloc(sizeof(float) * len);
	for (int i = 0; i < len; i++)
		A->vec[i] = value;

	if (A->vec_d != NULL)
		cudaFree(A->vec_d);

	A->vec_d = NULL;
}

void create_vector_on_device(vector* A, int len, float value) {
	//create vector on device  with all elements equal to 'value'
	//vector length is len

	A->len = len;
	A->vec = NULL;


	cudaError_t err;
	err = cudaMalloc((void**) &(A->vec_d), sizeof(float) * len);
	//printf("device pointer: %p\n",A->mat_d);
	if (err != cudaSuccess) {
		fprintf(stderr,
				"create_vector_on_device: cudaMalloc: ErrorMemoryAllocation\n");
		exit(1);
	}

	float *temp = (float*) malloc(sizeof(float) * len);
	for (int i = 0; i < len; i++)
		temp[i] = value;
	cudaMemcpy(A->vec_d, temp, sizeof(float) * len, cudaMemcpyHostToDevice);

	free(temp);

}

void create_vector_on_both(vector* A, int len, float value) {
	//create vector on device  with all elements equal to 'value'
	//vector length is len

	A->len = len;
	cudaError_t err;

	err = cudaMalloc((void**) &(A->vec_d), sizeof(float) * len);
	if (err != cudaSuccess) {
		fprintf(stderr,
				"create_vector_on_both: cudaMalloc: ErrorMemoryAllocation\n");
		exit(1);
	}

	A->vec = (float*) malloc(sizeof(float) * len);
	for (int i = 0; i < len; i++)
		A->vec[i] = value;
	cudaMemcpy(A->vec_d, A->vec, sizeof(float) * len, cudaMemcpyHostToDevice);

}

void copy_vector_to_device(vector* A) {

	cudaError_t err;

	if (A->vec == NULL) {
		fprintf(stderr,
				"copy_vector_to_device: vector not allocated on host\n");
		exit(1);
	}
	if (A->vec_d == NULL) {
		err = cudaMalloc((void**) &(A->vec_d), sizeof(float) * A->len);
		if (err != cudaSuccess) {
			fprintf(stderr, "copy_vector_to_device: cudaMalloc: FAIL\n");
			exit(1);
		}
	}

	err = cudaMemcpy(A->vec_d, A->vec, sizeof(float) * A->len,
			cudaMemcpyHostToDevice);
	switch (err) {
	case cudaErrorInvalidValue:
		fprintf(stderr, "copy_vector_to_device: cudaMemcpy: InvalidValue\n");
		exit(1);
		break;
	case cudaErrorInvalidDevicePointer:
		fprintf(stderr,
				"copy_vector_to_device: cudaMemcpy: InvalidDevicePointer\n");
		exit(1);
		break;
	case cudaErrorInvalidMemcpyDirection:
		fprintf(stderr,
				"copy_vector_to_device: cudaMemcpy: InvalidMemcpyDirection\n");
		exit(1);
		break;
	}
}

void copy_vector_from_device(vector* A) {

	if (A->vec_d == NULL) {
		fprintf(stderr,
				"copy_vector_from_device: vector not allocated on device\n");
		exit(1);
	}
	if (A->vec == NULL)
		cudaMallocHost((void**) &(A->vec), sizeof(float) * A->len);
	//A->mat = (float*)malloc(sizeof(float)*N);

	cudaError_t err;
	err = cudaMemcpy(A->vec, A->vec_d, sizeof(float) * A->len,
			cudaMemcpyDeviceToHost);
	switch (err) {
	case cudaErrorInvalidValue:
		fprintf(stderr, "copy_vector_from_device: cudaMemcpy: InvalidValue\n");
		exit(1);
		break;
	case cudaErrorInvalidDevicePointer:
		fprintf(stderr,
				"copy_vector_from_device: cudaMemcpy: InvalidDevicePointer\n");
		exit(1);
		break;
	case cudaErrorInvalidMemcpyDirection:
		fprintf(stderr,
				"copy_vector_from_device: cudaMemcpy: InvalidMemcpyDirection\n");
		exit(1);
		break;
	}
}

void allocate_vector_on_device(vector* A) {
	cudaError_t err;

	if (A->vec == NULL) {
		fprintf(stderr,
				"allocate_vector_on_device: vector not allocated on host\n");
		exit(1);
	}
	if (A->vec_d == NULL) {
		err = cudaMalloc((void**) &(A->vec_d), sizeof(float) * A->len);
		if (err != cudaSuccess) {
			fprintf(stderr, "allocate_vector_on_device: cudaMalloc: FAIL\n");
			exit(1);
		}
	} else {
		fprintf(stderr,
				"allocate_vector_on_device: vector already allocated on device");
		exit(1);
	}
}

void free_vector_on_device(vector* A) {
	if (A->vec_d != NULL)
		cudaFree(A->vec_d);
	A->vec_d = NULL;
}

void destroy_vector(vector* A) {
	if (A->vec != NULL)
		cudaFreeHost(A->vec);
	A->vec = NULL;
	if (A->vec_d != NULL)
		cudaFree(A->vec_d);
	A->vec_d = NULL;

	A->len = 0;
}

void vector_dot_product(vector a, vector b, float* out) {
	int N = a.len;

	if(a.len != b.len){
		fprintf(stderr,"vector_dot_product: length of vectors don't match");
		exit(1);
	}

	*out = cublasSdot(N, a.vec_d, 1, b.vec_d, 1);
	if (cublasGetError() != CUBLAS_STATUS_SUCCESS) {
				fprintf(stderr, "vector_outer_product: NOT SUCCESS\n");
				exit(1);
	}

	//cublasStatus_t cublasSdot (cublasHandle_t handle, int n,
	//	                           const float           *x, int incx,
	//	                           const float           *y, int incy,
	//	                           float           *result)
}

void vector_sqrt(vector a, float b){
	b = 0;
	for (int i = 0; i < a.len; i++){
		b += sqrt(a.vec[i]);
	}
}

void element_div(vector* a, float denom) {
	for (int i = 0; i < a->len; i++)
		a->vec[i] /= denom;
}
