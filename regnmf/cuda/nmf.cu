#include<stdio.h>
#include <math.h>
#include<stdlib.h>
#include"matrix.h"
#include<time.h>
#include<sys/time.h>

// status printed and convergence check every ITER_CHECK iterations
#define ITER_CHECK 25
// max number of iterations
#define MAX_ITER 100
// set to zero to guarantee MAX_ITER iterations, 0.001 is a good value otherwise
#define CONVERGE_THRESH 1E-5

// number of timers used in profiling (don't change)
#define TIMERS 10
char *tname[] = { "total", "sgemm", "eps", "vecdiv", "vecmult", "sumrows",
		"sumcols", "coldiv", "rowdiv", "check" };

void update_div(matrix W, matrix H, matrix X, const float thresh,
		const int max_iter, double* t, int verbose);
double get_time();
unsigned nextpow2(unsigned x);

void init_factors(matrix *W0, matrix *H0, matrix X0, int m, int n, int k,
		bool init_convex);
void fit(matrix W0, matrix H0, matrix X0, const float thresh,
		const int max_iter, double *t, int verbose);

int main(int argc, char *argv[]) {

	//factor X into W*H
	matrix W, H, X;

	// read in matrix data:
	// X - matrix to factorize
	// W - initial W matrix
	// H - initial H matrix
	read_matrix(&X, "./regnmf/cuda/pythonXin.bin");

	fit(W, H, X, CONVERGE_THRESH, MAX_ITER, NULL, 0);

	read_matrix(&W, "./regnmf/cuda/pythonWin.bin");
	read_matrix(&H, "./regnmf/cuda/pythonHin.bin");

	//make sure no zero elements
	matrix_eps(X);
	matrix_eps(H);
	matrix_eps(W);

	int max_iter;
	if (argc > 1)
		max_iter = atoi(argv[1]);
	else
		max_iter = MAX_ITER;

	// iterative nmf minimization
	update_div(W, H, X, CONVERGE_THRESH, max_iter, NULL, 1);

	// write results matrices to binary files
	// (can be read with export_bin.m in Matlab)
	write_matrix(W, "../Wout.bin");
	write_matrix(H, "../Hout.bin");

	destroy_matrix(&W);
	destroy_matrix(&H);
	destroy_matrix(&X);
	return 0;

}

extern "C" {
int nmf(float *WP, float *HP, float *XP, int m, int n, int k) {

	//factor X into W*H
	matrix W, H, X;

	// read in matrix data:
	// X - matrix to factorize
	// W - initial W matrix
	// H - initial H matrix
	read_matrix_from_array(&W, m, k, WP);
	read_matrix_from_array(&H, k, n, HP);
	read_matrix_from_array(&X, m, n, XP);

	//make sure no zero elements
	matrix_eps(X);
	matrix_eps(H);
	matrix_eps(W);

	// iterative nmf minimization
	update_div(W, H, X, CONVERGE_THRESH, MAX_ITER, NULL, 0);

	return 0;
}
}

extern "C" {
int regHALS(float *WP, float *HP, float *XP, int m, int n, int k) {

	//factor X into W*H
	matrix W, H, X;

	// read in matrix data:
	// X - matrix to factorize
	read_matrix_from_array(&X, m, n, XP);
	fit(W, H, X, CONVERGE_THRESH, MAX_ITER, NULL, 0);

	//make sure no zero elements
	matrix_eps(X);
	matrix_eps(H);
	matrix_eps(W);

	// iterative nmf minimization
	fit(W, H, X, CONVERGE_THRESH, MAX_ITER, NULL, 0);

	return 0;
}
}

int start_time(double* t, int i) {
	if (t != NULL) {
		t[i] -= get_time();
		return 1;
	} else
		return 0;
}

int stop_time(double* t, int i) {
	if (t != NULL) {
		t[i] += get_time();
		return 1;
	} else
		return 0;
}

void update_div(matrix W0, matrix H0, matrix X0, const float thresh,
		const int max_iter, double *t, int verbose) {
	//run iterative multiplicative updates on W,H

	cublasInit();

	const int M = W0.dim[0];
	const int K = W0.dim[1];
	const int N = H0.dim[1];

	// pad matrix dimensions to multiples of:
	const int PAD_MULT = 32;

	int M_padded = M;
	if (M % PAD_MULT != 0)
		M_padded = M + (PAD_MULT - (M % PAD_MULT));

	int K_padded = K;
	if (K % PAD_MULT != 0)
		K_padded = K + (PAD_MULT - (K % PAD_MULT));

	int N_padded = N;
	if (N % PAD_MULT != 0)
		N_padded = N + (PAD_MULT - (N % PAD_MULT));

	//unpadded test
	//M_padded = M;
	//N_padded = N;
	//K_padded = K;

	// find reduction parameters
	int MN_params[4] = { 1, 1, 1, 1 }; //M*N size reduction (whole matrix)
	int N_params[4] = { 1, 1, 1, 1 }; //N size reductions (rows)
	int M_params[4] = { 1, 1, 1, 1 }; //M size reductions (cols)

	int rem;
	rem = nextpow2(N_padded / 128 + (!(N_padded % 128) ? 0 : 1));
	if (rem <= 128) {
		N_params[0] = 128;
		N_params[1] = rem;
	} else if (rem <= 512) {
		N_params[0] = rem;
		N_params[1] = 128;
	} else {
		fprintf(stderr, "reduction parameter error\n");
		exit(1);
	}

	rem = nextpow2(M_padded / 128 + (!(M_padded % 128) ? 0 : 1));
	if (rem <= 128) {
		M_params[0] = 128;
		M_params[1] = rem;
	} else if (rem <= 512) {
		M_params[0] = rem;
		M_params[1] = 128;
	} else {
		fprintf(stderr, "reduction parameter error\n");
		exit(1);
	}

	MN_params[0] = M_params[0];
	MN_params[1] = M_params[1];
	MN_params[2] = N_params[0];
	MN_params[3] = N_params[1];

	//printf("reduction parameters: ");
	//printf("%u,%u,%u,%u\n",MN_params[0],MN_params[1],MN_params[2],MN_params[3]);

	// block size in vector arithmetic operations
	const int BLOCK_SIZE = 1024;

	//copy host matrices to device memory
	copy_matrix_to_device(&W0);
	copy_matrix_to_device(&H0);
	copy_matrix_to_device(&X0);

	//matrix to hold W*H
	matrix WH0;
	create_matrix_on_device(&WH0, M, N, 0.0);

	int i;

	/*
	 double t_array[TIMERS];
	 if(t==NULL)
	 t = t_array;
	 */
	if (t != NULL) {
		for (i = 0; i < TIMERS; i++)
			t[i] = 0;
	}

	//float nancheck, zerocheck;
	// compute initial divergence and error
	float diff, div, change, prev_diff, prev_div;

	matrix_multiply_d(W0, H0
			, WH0);
	diff = matrix_difference_norm_d(compute, X0, WH0, MN_params);

	div = matrix_div_d(compute, X0, WH0, MN_params);
	if (verbose)
		printf("i: %4i, error: %6.4f, initial div: %8.4e\n", 0, diff, div);

	// free device memory for unpadded matrices
	free_matrix_on_device(&W0);
	free_matrix_on_device(&H0);
	free_matrix_on_device(&X0);
	free_matrix_on_device(&WH0);

	//initialize temp matrices -----------------------

	//matrix to hold X./(W*H+EPS)
	matrix Z;
	create_matrix_on_device(&Z, M_padded, N_padded, 0.0);

	//matrix to hold W'*Z
	matrix WtZ;
	create_matrix_on_device(&WtZ, K_padded, N_padded, 0.0);

	//matrix to hold Z*H'
	matrix ZHt;
	create_matrix_on_device(&ZHt, M_padded, K_padded, 0.0);

	//matrix to hold sum(W) [sum of cols of W]
	matrix sumW;
	create_matrix_on_device(&sumW, 1, K_padded, 0.0);

	//matrix to hold sum(H,2) [sum of rows of H]
	matrix sumH2;
	create_matrix_on_device(&sumH2, K_padded, 1, 0.0);

	//matrices to hold padded versions of matrices
	matrix W;
	create_matrix_on_device(&W, M_padded, K_padded, 0.0);

	matrix H;
	create_matrix_on_device(&H, K_padded, N_padded, 0.0);

	matrix X;
	create_matrix_on_device(&X, M_padded, N_padded, 0.0);

	// move host matrices to padded device memory
	copy_matrix_to_device_padded(W0, W);
	copy_matrix_to_device_padded(H0, H);
	copy_matrix_to_device_padded(X0, X);

	//t[0] -= get_time();
	start_time(t, 0);

	//matrix test1;

	for (i = 0; i < max_iter; i++) {

		//check for convergence, print status
		if (i % ITER_CHECK == 0 && i != 0) {
			//t[9] -= get_time();
			start_time(t, 9);
			matrix_multiply_d(W, H, Z);
			prev_diff = diff;
			diff = matrix_difference_norm_d(compute, X, Z, MN_params);
			change = (prev_diff - diff) / prev_diff;
			//t[9] += get_time();
			stop_time(t, 9);
			if (verbose)
				printf("i: %4i, error: %6.4f, %% change: %8.5f\n", i, diff,
						change);
			if (change < thresh) {
				printf("converged\n");
				break;
			}
		}

		/* matlab algorithm
		 Z = X./(W*H+eps); H = H.*(W'*Z)./(repmat(sum(W)',1,F));
		 Z = X./(W*H+eps);
		 W = W.*(Z*H')./(repmat(sum(H,2)',N,1));
		 */

		//
		// UPDATE H -----------------------------
		//

		//WH = W*H
		//t[1] -= get_time();
		start_time(t, 1);
		matrix_multiply_d(W, H, Z);
		//t[1] += get_time();
		stop_time(t, 1);

		//WH = WH+EPS
		//t[2] -= get_time();
		start_time(t, 2);
		matrix_eps_d(Z, BLOCK_SIZE);
		//t[2] += get_time();
		stop_time(t, 2);

		//Z = X./WH
		//t[3] -= get_time();
		start_time(t, 3);
		element_divide_d(X, Z, Z, BLOCK_SIZE);
		//t[3] += get_time();
		stop_time(t, 3);

		//sum cols of W into row vector
		//t[6] -= get_time();
		start_time(t, 6);
		sum_cols_d(compute, W, sumW, M_params);
		matrix_eps_d(sumW, 32);
		//t[6] += get_time();
		stop_time(t, 6);

		//convert sumW to col vector (transpose)
		sumW.dim[0] = sumW.dim[1];
		sumW.dim[1] = 1;

		//WtZ = W'*Z
		//t[1] -= get_time();
		start_time(t, 1);
		matrix_multiply_AtB_d(W, Z, WtZ);
		//t[1] += get_time();
		stop_time(t, 1);

		//WtZ = WtZ./(repmat(sum(W)',1,H.dim[1])
		//[element divide cols of WtZ by sumW']
		//t[7] -= get_time();
		start_time(t, 7);
		col_divide_d(WtZ, sumW, WtZ);
		//t[7] += get_time();
		stop_time(t, 7);

		//H = H.*WtZ
		//t[4] -= get_time();
		start_time(t, 4);
		element_multiply_d(H, WtZ, H, BLOCK_SIZE);
		//t[4] += get_time();
		stop_time(t, 4);

		//
		// UPDATE W ---------------------------
		//

		//WH = W*H
		//t[1] -= get_time();
		start_time(t, 1);
		matrix_multiply_d(W, H, Z);
		//t[1] += get_time();
		stop_time(t, 1);

		//WH = WH+EPS
		//t[2] -= get_time();
		start_time(t, 2);
		matrix_eps_d(Z, BLOCK_SIZE);
		//t[2] += get_time();
		stop_time(t, 2);

		//Z = X./WH
		//t[3] -= get_time();
		start_time(t, 3);
		element_divide_d(X, Z, Z, BLOCK_SIZE);
		//t[3] += get_time();
		stop_time(t, 3);

		//sum rows of H into col vector
		//t[5] -= get_time();
		start_time(t, 5);
		sum_rows_d(compute, H, sumH2, N_params);
		matrix_eps_d(sumH2, 32);
		//t[5] += get_time();
		stop_time(t, 5);

		//convert sumH2 to row vector (transpose)
		sumH2.dim[1] = sumH2.dim[0];
		sumH2.dim[0] = 1;

		//ZHt = Z*H'
		//t[1] -= get_time();
		start_time(t, 1);
		matrix_multiply_ABt_d(Z, H, ZHt);
		//t[1] += get_time();
		stop_time(t, 1);

		//ZHt = ZHt./(repmat(sum(H,2)',W.dim[0],1)
		//[element divide rows of ZHt by sumH2']
		//t[8] -= get_time();
		start_time(t, 8);
		row_divide_d(ZHt, sumH2, ZHt);
		//t[8] += get_time();
		stop_time(t, 8);

		//W = W.*ZHt
		//t[4] -= get_time();
		start_time(t, 4);
		element_multiply_d(W, ZHt, W, BLOCK_SIZE);
		//t[4] += get_time();
		stop_time(t, 4);

		// ------------------------------------

		//reset sumW to row vector
		sumW.dim[1] = sumW.dim[0];
		sumW.dim[0] = 1;
		//reset sumH2 to col vector
		sumH2.dim[0] = sumH2.dim[1];
		sumH2.dim[1] = 1;

		// ---------------------------------------

	}

	//t[0] += get_time();
	stop_time(t, 0);

	//reallocate unpadded device memory
	allocate_matrix_on_device(&W0);
	allocate_matrix_on_device(&H0);

	//copy padded matrix to unpadded matrices
	copy_from_padded(W0, W);
	copy_from_padded(H0, H);

	// free padded matrices
	destroy_matrix(&W);
	destroy_matrix(&H);
	destroy_matrix(&X);

	// free temp matrices
	destroy_matrix(&Z);
	destroy_matrix(&WtZ);
	destroy_matrix(&ZHt);
	destroy_matrix(&sumW);
	destroy_matrix(&sumH2);

	copy_matrix_to_device(&X0);
	create_matrix_on_device(&WH0, M, N, 0.0);

	// copy device results to host memory
	copy_matrix_from_device(&W0);
	copy_matrix_from_device(&H0);

	// evaluate final results
	matrix_multiply_d(W0, H0, WH0);
	prev_diff = diff;
	diff = matrix_difference_norm_d(compute, X0, WH0, MN_params);
	prev_div = div;
	div = matrix_div_d(compute, X0, WH0, MN_params);
	if (verbose) {
		change = (prev_diff - diff) / prev_diff;
		printf("max iterations reached\n");
		printf("i: %4i, error: %6.4f, %% change: %8.5f\n", i, diff, change);
		change = (prev_div - div) / prev_div;
		printf("\tfinal div: %8.4e, %% div change: %8.5f\n", div, change);

		printf("\n");
		if (t != NULL) {printf("cublas Init");
			for (i = 0; i < TIMERS; i++)
				printf("t[%i]: %8.3f (%6.2f %%) %s\n", i, t[i],
						t[i] / t[0] * 100, tname[i]);
		}
	}

	//clean up extra reduction memory
	matrix_difference_norm_d(cleanup, X0, WH0, MN_params);
	matrix_div_d(cleanup, X0, WH0, MN_params);
	sum_cols_d(cleanup, W, sumW, M_params);
	sum_rows_d(cleanup, H, sumH2, N_params);

	// free device memory for unpadded matrices
	free_matrix_on_device(&W0);
	free_matrix_on_device(&H0);
	free_matrix_on_device(&X0);

	// free temp matrices
	destroy_matrix(&WH0);

	cublasShutdown();

}

void convex_cone(matrix* W0, matrix* H0, matrix data, int latents, int* params,
		int BLOCK_SIZE) {
	int row = data.dim[0];
	int col = data.dim[1];

	create_matrix(W0, latents, row, 0);
	create_matrix(H0, latents, col, 0);

	for (int i = 0; i < latents; i++) {
		int best_col = most_interesting_column(data);

		vector timecourse;

		create_vector(&timecourse, col, 0);
		matrix_column(data, &timecourse, best_col);

		float norm = 0;

		copy_vector_to_device(&timecourse);
		vector_dot_product(timecourse, timecourse, &norm);
		element_div(&timecourse, sqrtf(norm));

		vector base;

		copy_matrix_to_device(&data);

		create_vector_on_both(&base, row, 0);

		matrix_vector_multiply_Atb(data, timecourse, &base);
		zero_check_d(compute, data, params);

		matrix outer;
		create_matrix_on_both(&outer, row, col, 0);

		vector_outer_product(timecourse, base, &outer);

		matrix newdata;
		create_matrix_on_both(&newdata, row, col, 0);

		element_subtract_d(data, outer, newdata, BLOCK_SIZE);

		copy_matrix_from_device(&newdata);
		replace_matrix(&data, newdata);

		copy_vector_from_device(&base);
		copy_vector_from_device(&timecourse);

		set_matrix_column(W0, base, i); //check if float* is storing row or column order
		set_matrix_column(H0, timecourse, i);

		free_matrix_on_device(&data);
		destroy_matrix(&newdata);
		destroy_vector(&base);
		destroy_vector(&timecourse);

	}

	copy_matrix_to_device(H0);
	matrix_transpose(W0);
	copy_matrix_to_device(W0);

}

void init_factors(matrix* W0, matrix* H0, matrix X0, int m, int n, int k,
		bool init_convex, int* params, int BLOCK_SIZE) {
	if (init_convex) {
		convex_cone(W0, H0, X0, k, params, BLOCK_SIZE);
	} else {
		create_uniform_rand_matrix(W0, m, k);
		create_matrix(H0, k, n, 0);
		copy_matrix_to_device(W0);
		copy_matrix_to_device(H0);
	}
}

void create_nn_matrix(matrix* a, int lda, int tda) {
	const int len = lda*tda;


	for (int i = 0; i < lda; i++) {
		for (int j = 0; j < tda; j++) {
			matrix temp;
			create_matrix(&temp, lda, tda, 0);

			if (i > 0)
				temp.mat[i - 1 + temp.dim[0] * j] = 1;
			if (i < lda - 1)
				temp.mat[i + 1 + temp.dim[0] * j] = 1;
			if (j > 0)
				temp.mat[i + temp.dim[0] * j - 1] = 1;
			if (j < tda - 1)
				temp.mat[i + temp.dim[0] * j + 1] = 1;



		float sum = 0;
		vector flat;
		create_vector(&flat, len, 0);
		int k = 0;
			for (int x = 0; x < lda; x++){
			for (int y = 0; y < tda; y++) {
				flat.vec[k] = 1*temp.mat[x + temp.dim[0] * y];
				sum += temp.mat[x + temp.dim[0] * y];
				k++;
			}
			}

		element_div(&flat, sum);
		set_matrix_column(a, flat, i);
		//destroy_vector(&flat);
		}
	}
}

vector project_residuals(matrix res, int oldind, vector to_base, matrix X){
 float sparse_param = 0.5;
 float smooth_param = 2;

 vector new_vec;
 create_vector_on_both(&new_vec, res.dim[0], 0);
 matrix_vector_multiply_Atb(res, to_base, &new_vec);

 return new_vec;
}

void update(matrix X0, matrix W0, matrix H0, int BLOCK_SIZE){
	matrix E, dot;
	create_matrix_on_both(&E, W0.dim[0], H0.dim[1], 0);
	create_matrix_on_both(&dot, W0.dim[0], H0.dim[1], 0);
	matrix_multiply_d(W0, H0, dot);
	element_subtract_d(X0, dot, E, BLOCK_SIZE);

	matrix_transpose(&H0);


	const float basenorm = 1;
	const float psi = 1E-12;


	for (int i = 0; i < W0.dim[1]; i++) {
	vector aj, xj;
	create_vector_on_both(&aj, W0.dim[0], 0);
	create_vector_on_both(&xj, H0.dim[0], 0);

	matrix_column(W0, &aj, i);
	matrix_column(H0, &xj, i);

	matrix ajxj, Rt;
	create_matrix_on_both(&ajxj, X0.dim[0], X0.dim[1], 0);
	create_matrix_on_both(&Rt, X0.dim[0], X0.dim[1], 0);
	vector_outer_product(aj, xj, &ajxj);
	element_addition_d(E, ajxj, Rt, BLOCK_SIZE);
	matrix_transpose(&Rt);
	xj = project_residuals(Rt, i, aj, H0);

	element_div(&xj, basenorm + psi);

	matrix_transpose(&Rt);
	matrix_transpose(&W0);

	aj = project_residuals(Rt, i, xj, W0);

	matrix_transpose(&W0);
	element_div(&aj, timenorm(aj) + psi);

	vector_outer_product(aj, xj, &ajxj);

	matrix newRt;
	create_matrix_on_both(&newRt, X0.dim[0], X0.dim[1], 0);
	element_subtract_d(Rt, ajxj, newRt, BLOCK_SIZE);

	set_matrix_column(&W0, aj, i);
	set_matrix_column(&H0, xj, i);

	replace_matrix(&E, newRt);

//	destroy_matrix(&ajxj);
//	destroy_matrix(&Rt);
//	destroy_matrix(&newRt);
//	destroy_vector(&aj);
//	destroy_vector(&xj);
	}

	matrix_transpose(&H0);

}

void fit(matrix W0, matrix H0, matrix X0, const float thresh,
		const int max_iter, double *t, int verbose) {
	cublasInit();
	const int M = X0.dim[0];
	const int K = 80;
	const int N = X0.dim[1];

	// pad matrix dimensions to multiples of:
	const int PAD_MULT = 32;

	int M_padded = M;
	if (M % PAD_MULT != 0)
		M_padded = M + (PAD_MULT - (M % PAD_MULT));

	int K_padded = K;
	if (K % PAD_MULT != 0)
		K_padded = K + (PAD_MULT - (K % PAD_MULT));

	int N_padded = N;
	if (N % PAD_MULT != 0)
		N_padded = N + (PAD_MULT - (N % PAD_MULT));

	//unpadded test
	//M_padded = M;
	//N_padded = N;
	//K_padded = K;

	// find reduction parameters
	int MN_params[4] = { 1, 1, 1, 1 }; //M*N size reduction (whole matrix)
	int N_params[4] = { 1, 1, 1, 1 }; //N size reductions (rows)
	int M_params[4] = { 1, 1, 1, 1 }; //M size reductions (cols)

	int rem;
	rem = nextpow2(N_padded / 128 + (!(N_padded % 128) ? 0 : 1));
	if (rem <= 128) {
		N_params[0] = 128;
		N_params[1] = rem;
	} else if (rem <= 512) {
		N_params[0] = rem;
		N_params[1] = 128;
	} else {
		fprintf(stderr, "reduction parameter error\n");
		exit(1);
	}

	rem = nextpow2(M_padded / 128 + (!(M_padded % 128) ? 0 : 1));
	if (rem <= 128) {
		M_params[0] = 128;
		M_params[1] = rem;
	} else if (rem <= 512) {
		M_params[0] = rem;
		M_params[1] = 128;
	} else {
		fprintf(stderr, "reduction parameter error\n");
		exit(1);
	}

	MN_params[0] = M_params[0];
	MN_params[1] = M_params[1];
	MN_params[2] = N_params[0];
	MN_params[3] = N_params[1];

	//printf("reduction parameters: ");
	//printf("%u,%u,%u,%u\n",MN_params[0],MN_params[1],MN_params[2],MN_params[3]);

	// block size in vector arithmetic operations
	const int BLOCK_SIZE = 1024;

	init_factors(&W0, &H0, X0, M, N, K, true, MN_params, BLOCK_SIZE);

	//Redo this w/ Matrix and Vectors duh!
	//const int SINIT = 2500;
	//int SHAPE = 50;
	//matrix S;
	//create_matrix(&S, SINIT, SINIT, 0);

	//create_nn_matrix(&S, SHAPE, SHAPE);

	int count = 0;
	float obj_old = 1e99;
	float nrm_Y = 0;

	nrm_Y = frobenius_norm(X0);


	const int maxcount = 50;
	const float eps = 1E-05;

	while (true){
		if(count >= maxcount){ break;}
		count ++;

		update(X0, W0, H0, BLOCK_SIZE);

		matrix dot, E;
		create_matrix_on_both(&dot, X0.dim[0], X0.dim[1], 0);
		create_matrix_on_both(&E, X0.dim[0], X0.dim[1], 0);

		matrix_multiply_d(W0, H0, dot);
		element_subtract_d(X0, dot, E, BLOCK_SIZE);

		float obj = frobenius_norm(E) / nrm_Y;
		float delta_obj = obj - obj_old;

		if(-eps < delta_obj <=0){
			break;
		}
		obj_old = obj;
	}


	cublasShutdown();
}

double get_time() {
	//output time in microseconds

	//the following line is required for function-wise timing to work,
	//but it slows down overall execution time.
	//comment out for faster execution
	cudaThreadSynchronize();

	struct timeval t;
	gettimeofday(&t, NULL);
	return (double) (t.tv_sec + t.tv_usec / 1E6);
}

unsigned nextpow2(unsigned x) {
	x = x - 1;
	x = x | (x >> 1);
	x = x | (x >> 2);
	x = x | (x >> 4);
	x = x | (x >> 8);
	x = x | (x >> 16);
	return x + 1;

}
