#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

void generate_random(double *input, size_t size)
{
    for (size_t i = 0; i < size; i++) {
        input[i] = rand() / (double)(RAND_MAX);
    }
}


double serial_sum(double *x, size_t size)
{

    double sum_val = 0.0;

    for (size_t i = 0; i < size; i++) {
        sum_val += x[i];
    }

    return sum_val;
}

double omp_sum(double *x, size_t size)
{
    double sum_val = 0.0;

    #pragma omp parallel for reduction(+:sum_val)
    for (size_t i = 0; i < size; i++) {
        sum_val += x[i];
    }

    return sum_val;
}

double omp_critical_sum(double *x, size_t size)
{
    double sum_val = 0.0;

    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        #pragma omp critical
        sum_val += x[i];
    }
    return sum_val;
}

double omp_local_sum(double *x, size_t size)
{
    int MAX_THREADS = omp_get_max_threads();
    double* local_sum = calloc(MAX_THREADS, sizeof(double));

    #pragma omp parallel 
    {
        int id = omp_get_thread_num();
        local_sum[id] = 0;
    #pragma omp for
        for (size_t i = id; i < size; i++) {
            local_sum[id] += x[i];
        }
    }

    double full_sum = 0;
    for (int i = 0; i < MAX_THREADS; i++) {
        full_sum += local_sum[i];
    }

    free(local_sum);
    return full_sum;

}



double omp_local_sum_padding(double *x, size_t size){
    typedef struct {
        double sum;
        char padding[128];
    } padded_double;

    int MAX_THREADS = omp_get_max_threads();
    padded_double *local_sum = calloc(MAX_THREADS, sizeof(padded_double));

    #pragma omp parallel shared(local_sum)
    {
        int id = omp_get_thread_num();
        #pragma omp for
        for (size_t i = 0; i < size; i++) {
            local_sum[id].sum += x[i];
        }
    }

    double full_sum = 0;
    for (int i = 0; i < MAX_THREADS; i++) {
        full_sum += local_sum[i].sum;
    }

    free(local_sum);
    return full_sum;

}



void function_time(double (*func)(double*, size_t),double* array, size_t size, int iterations,
                   double* mean_time, double* std_derivation)
{
    double* times = calloc(iterations, sizeof(double));


    for (int i = 0; i < iterations; i++) {
        double start = omp_get_wtime();
        func(array, size);
        double end = omp_get_wtime();
        times[i] = end - start;
    }

    for (int i = 0; i < iterations; i++) {
        *mean_time += times[i];
    }
    *mean_time /= iterations;

    for (int i = 0; i < iterations; i++) {
        *std_derivation += pow(times[i] - *mean_time, 2);
    }
    *std_derivation /= iterations;
    *std_derivation = sqrt(*std_derivation);

    free(times);
}

void print_test(double (*func)(double*, size_t), double* values, size_t size, int iterations, char* name, int num_threads){
    double mean_time = 0.0;
    double std_derivation = 0.0;
    function_time(func, values, size, iterations, &mean_time, &std_derivation);
    /*if(serial_sum(values, size) != func(values, size)){
        printf("Error: serial sum and %s sum are not equal\n", name);
        printf("The difference is approximately %.3e\n", fabs(serial_sum(values, size) - omp_critical_sum(values, size)));
    }*/
    printf("%s performance with %d threads: mean time = %.3es, std_derivation = %.3e\n", name, num_threads, mean_time, std_derivation);
}

int main() {

    int size = 10000000;
    double *values = calloc(size, sizeof(double));
    generate_random(values, size);
    int iterations = 100;

    int max_threads = omp_get_max_threads();

    // Print the maximum number of threads
    printf("Maximum number of threads: %d\n", max_threads);


    //1. serial sum
    double mean_time = 0.0;
    double std_derivation = 0.0;
    function_time(serial_sum, values, size, iterations, &mean_time, &std_derivation);
    printf("Serial sum: mean time = %.3es, standard derivation = %.3e\n\n", mean_time, std_derivation);


    //2. omp sum
    omp_set_num_threads(32);
    print_test(omp_sum, values, size, iterations, "OMP", 32);

    //3. omp critical sum
    int num_threads_critical[] = {1, 2, 4, 8, 16, 20,24,28,32};
    for (int i = 0; i < 9; ++i) {
        omp_set_num_threads(num_threads_critical[i]);
        print_test(omp_critical_sum, values, size, iterations, "OMP critical", num_threads_critical[i]);
    }

    //4.omp local sum
    int num_threads_local[] = {1, 32, 64, 128};
    for (int i = 0; i < 4; ++i) {
        omp_set_num_threads(num_threads_local[i]);
        printf("Number of threads: %d\n", num_threads_critical[i]);
        print_test(omp_local_sum, values, size, iterations, "OMP local", num_threads_local[i]);
    }

    //5.omp local sum padding
        int num_threads_local_padding[] = {1, 32, 64, 128};
        for (int i = 0; i < 4; ++i) {
            omp_set_num_threads(num_threads_local_padding[i]);
            printf("Number of threads: %d\n", num_threads_critical[i]);
            print_test(omp_local_sum_padding, values, size, iterations, "OMP local padding", num_threads_local_padding[i]);
        }

    free(values);
    return 0;
}
