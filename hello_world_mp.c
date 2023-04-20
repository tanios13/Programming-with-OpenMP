#include "omp.h"
#include <stdio.h>

int main(){
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        printf("Hello World from Thread %d!\n", id);
    }
    return 0;
}