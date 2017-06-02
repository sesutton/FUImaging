
#include <cublas.h>
#include <math.h>


typedef struct{
    float* mat;  //pointer to host data
    float* mat_d; //pointer to device data
    int dim[2];   //dimensions: {rows,cols}
} matrix;

typedef enum{
    compute,
    cleanup
} action_t;

//creating, allocating, moving matrices
void read_matrix(matrix* A, char* file);
void read_matrix_from_float(matrix* A, int rows, int cols, float* value);
void write_matrix(matrix A, char* file);
void create_matrix(matrix* A, int rows, int cols, float value);
void create_uniform_rand_matrix(matrix* A, int rows, int cols);
void create_matrix_on_device(matrix* A, int rows, int cols, float value);
void create_matrix_on_both(matrix* A, int rows, int cols, float value);
void copy_matrix_to_device(matrix* A);
void copy_matrix_on_device(matrix A, matrix B);
void copy_matrix_from_device(matrix* A);
void copy_to_padded(matrix A, matrix Apad);
void copy_matrix_to_device_padded(matrix A, matrix Apad);
void copy_from_padded(matrix A, matrix Apad);
void copy_matrix_from_device_padded(matrix A, matrix Apad);
void allocate_matrix_on_device(matrix* A);
void free_matrix_on_device(matrix* A);
void destroy_matrix(matrix* A);

//matrix analysis
void print_matrix(matrix A);
float matrix_difference_norm_d(action_t action,  matrix a, matrix c, int* params);
float matrix_div_d(action_t action, matrix a, matrix b, int* params);
float nan_check_d(action_t action, matrix a, int* params);
float zero_check_d(action_t action, matrix a, int* params);
float zero_check(matrix a);

//sgemms
void matrix_multiply_d( matrix a, matrix b, matrix c );
void matrix_multiply_AtB_d( matrix a, matrix b, matrix c );
void matrix_multiply_ABt_d( matrix a, matrix b, matrix c );

//element operations
void element_multiply_d( matrix a, matrix b, matrix c, int block_size);
void element_divide_d( matrix a, matrix b, matrix c, int block_size);
void matrix_eps_d( matrix a, int block_size);
void matrix_eps(matrix a);

//row/col-wise
void row_divide_d( matrix a, matrix b, matrix c);
void col_divide_d( matrix a, matrix b, matrix c);
void sum_cols_d(action_t action, matrix a, matrix c, int* params);
void sum_rows_d(action_t action, matrix a, matrix c, int* params);

int most_interesting_column(matrix a);
void max_columns(float *column_maxs, matrix a);
void matrix_column(matrix a, float *column, int col_index);
float dot_product(float v[], float u[], int n);
void elementwise_div(float *vector, int n, float denominator);
void matrix_vector_multiply_Atb(matrix *a, float *b, matrix *c);
void allocate_vector_on_device(float **d_A, int N);
void copy_vector_to_device(float *A, int N, float **d_A);
void matrix_vector_multiply_Atb(matrix a, float *b, float *c);
