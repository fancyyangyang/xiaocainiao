#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define MATRIX_SIZE 5
#define KERNEL_SIZE 3

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Read input matrix and kernel
    float input[MATRIX_SIZE][MATRIX_SIZE];
    float kernel[KERNEL_SIZE][KERNEL_SIZE];
    if (rank == 0) {
        printf("Enter input matrix:\n");
        for (int i = 0; i < MATRIX_SIZE; i++) {
            for (int j = 0; j < MATRIX_SIZE; j++) {
                scanf("%f", &input[i][j]);
            }
        }
        printf("Enter convolution kernel:\n");
        for (int i = 0; i < KERNEL_SIZE; i++) {
            for (int j = 0; j < KERNEL_SIZE; j++) {
                scanf("%f", &kernel[i][j]);
            }
        }
    }

    // Broadcast input matrix and kernel to all processes
    MPI_Bcast(input, MATRIX_SIZE * MATRIX_SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(kernel, KERNEL_SIZE * KERNEL_SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Pad input matrix with zeros
    float padded_input[MATRIX_SIZE + KERNEL_SIZE - 1][MATRIX_SIZE + KERNEL_SIZE - 1];
    for (int i = 0; i < MATRIX_SIZE + KERNEL_SIZE - 1; i++) {
        for (int j = 0; j < MATRIX_SIZE + KERNEL_SIZE - 1; j++) {
            if (i < KERNEL_SIZE / 2 || i >= MATRIX_SIZE + KERNEL_SIZE / 2 ||
                j < KERNEL_SIZE / 2 || j >= MATRIX_SIZE + KERNEL_SIZE / 2) {
                padded_input[i][j] = 0.0;
            } else {
                padded_input[i][j] = input[i - KERNEL_SIZE / 2][j - KERNEL_SIZE / 2];
            }
        }
    }

    // Divide input matrix into blocks and distribute them among processes
    int block_size = MATRIX_SIZE / size;
    float block[block_size + KERNEL_SIZE - 1][MATRIX_SIZE + KERNEL_SIZE - 1];
    MPI_Scatter(padded_input, (block_size + KERNEL_SIZE - 1) * (MATRIX_SIZE + KERNEL_SIZE - 1),
                MPI_FLOAT, block, (block_size + KERNEL_SIZE - 1) * (MATRIX_SIZE + KERNEL_SIZE - 1),
                MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Initialize output matrix to zero
    float output[block_size][MATRIX_SIZE];
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            output[i][j] = 0.0;
        }
    }

    // Compute convolution of each block
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            float sum = 0.0;
            for (int k = 0; k < KERNEL_SIZE; k++) {
                for (int l = 0; l < KERNEL_SIZE; l++) {
                    sum += block[i + k][j + l] * kernel[k][l];
                }
            }
            output[i][j] = sum;
            printf("%f ", output[i][j]);
        }
    }

    // Gather results from all processes and write output matrix to file
    float result[MATRIX_SIZE][MATRIX_SIZE];
    MPI_Gather(output, block_size * MATRIX_SIZE, MPI_FLOAT, result, block_size * MATRIX_SIZE,
               MPI_FLOAT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Output matrix:\n");
        for (int i = 0; i < MATRIX_SIZE; i++) {
            for (int j = 0; j < MATRIX_SIZE; j++) {
                printf("%f ", result[i][j]);
            }
            printf("\n");
        }
    }

    MPI_Finalize();
    return 0;
}
