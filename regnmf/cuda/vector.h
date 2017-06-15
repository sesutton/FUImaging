#include <cublas.h>
#include <math.h>

typedef struct{
    float* vec;  //pointer to host data
    float* vec_d; //pointer to device data
    int len;   //length of vector
} vector;



//creating, allocating, moving vectors
void read_vector_from_array(vector* A, int len, float* value);
void create_vector(vector* A, int len, float value);
void create_vector_on_device(vector* A, int len, float value);
void create_vector_on_both(vector* A, int len, float value);
void copy_vector_to_device(vector* A);
void copy_vector_from_device(vector* A);
void allocate_vector_on_device(vector* A);
void free_vector_on_device(vector* A);
void destroy_vector(vector* A);

void vector_dot_product(vector a, vector b, float* out);

void element_div(vector* a, float denom);
void vector_sqrt(vector a, float b);
