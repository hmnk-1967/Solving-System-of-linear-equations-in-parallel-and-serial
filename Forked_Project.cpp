/*This is the modified version of maitreyeepaliwal/Solving-System-of-linear-equations-in-parallel-and-serial/random.c and is being used under MIT License.
Link to the master project: https://github.com/maitreyeepaliwal/Solving-System-of-linear-equations-in-parallel-and-serial 
To use OpenMP on Visual Studio, refer to this link: https://stackoverflow.com/questions/4515276/openmp-is-not-creating-threads-in-visual-studio */
#include <iostream>
#include <omp.h>

using namespace std;

//Function Prototypes
void conjugategradient_p(double* A, double* b, double* x, int n);
void conjugategradient_s(double* A, double* b, double* x, int n);

int main()
{
    long n;
    //input matrix dimensions
    cout << "Enter matrix dimension for a square matrix: ";
    cin >> n;

    //defining requirements
    double* matrix = (double*) malloc(sizeof(double) * n * n);
    double* b = (double*) malloc(sizeof(double) * n);
    double* x = (double*) malloc(sizeof(double) * n);

    for (int i = 0; i < (n * n); i++)
    {
        double x = (rand() / (double)RAND_MAX);
        matrix[i] = x;
    }

    for (int i = 0; i < n; i++)
    {
        double x = (rand() / (double)RAND_MAX);
        b[i] = x;
    }

    // matrices
    double* matrix2 = (double*) malloc(sizeof(double) * n * n);
    double* b2 = (double*) malloc(sizeof(double) * n);
    double* x2 = (double*) malloc(sizeof(double) * n);

    double* matrix3 = (double*) malloc(sizeof(double) * n * n);
    double* b3 = (double*) malloc(sizeof(double) * n);
    double* x3 = (double*) malloc(sizeof(double) * n);

    double* matrix4 = (double*) malloc(sizeof(double) * n * n);
    double* b4 = (double*) malloc(sizeof(double) * n);
    double* x4 = (double*) malloc(sizeof(double) * n);

    double* matrix5 = (double*) malloc(sizeof(double) * n * n);
    double* b5 = (double*) malloc(sizeof(double) * n);
    double* x5 = (double*) malloc(sizeof(double) * n);

    double* matrix6 = (double*) malloc(sizeof(double) * n * n);
    double* b6 = (double*) malloc(sizeof(double) * n);
    double* x6 = (double*) malloc(sizeof(double) * n);


    for (int i = 0; i < n * n; i++)
    {
        matrix2[i] = matrix[i];
        matrix3[i] = matrix[i];
        matrix4[i] = matrix[i];
        matrix5[i] = matrix[i];
        matrix6[i] = matrix[i];
    }

    for (int i = 0; i < n; i++)
    {
        b2[i] = b[i];
        b3[i] = b[i];
        b4[i] = b[i];
        b5[i] = b[i];
        b6[i] = b[i];
    }


    cout << "\n\nA * x = b  \nTo find : x \n\n";

    //A Matrix
    cout << "Matrix A: \n";
    for (int i = 0; i < n * n; i++)
    {
        cout << matrix[i] << "    ";
        if ((i + 1) % n == 0)
        {
            cout << endl;
        }
    }

    //B Matrix
    cout << "\n\nMatrix b: \n";
    for (int i = 0; i < n; i++)
    {
        cout << b[i] << endl;
    }

    cout << "\n\nFINDING SOLUTIONS: \n\n 1. Conjugate Gradient: \n\t A. Serial Execution: \n";

    //Conjugate Gradient
    double tb = omp_get_wtime();
    conjugategradient_s(matrix3, b3, x3, n);
    tb = omp_get_wtime() - tb;
    cout << "\t B. Parallel Execution: \n";
    double tb1 = omp_get_wtime();
    conjugategradient_p(matrix4, b4, x4, n);
    tb1 = omp_get_wtime() - tb1;

    cout << "\n\t Time for serial execution: " << tb << " seconds\n\n";
    cout << "\n\t Time for parallel execution: " << tb1 << " seconds\n\n";

    free(matrix);
    free(b);
    free(x);
    free(matrix2);
    free(b2);
    free(x2);
    free(matrix3);
    free(b3);
    free(x3);
    free(matrix4);
    free(b4);
    free(x4);
    free(matrix5);
    free(b5);
    free(x5);
    system("pause");
    return 0;
}

//Function definitions
void conjugategradient_p(double* A, double* b, double* x, int n)
{
    int t;
    int max_iterations;
    cout << "\nEnter number of iterations: ";
    cin >> max_iterations;

    cout << "\nEnter number of threads: ";
    cin >> t;
    double* r = new double[n];
    double* p = new double[n];
    double* px = new double[n];

#pragma omp parallel for num_threads(t)
    for (int i = 0; i < n; i++)
    {
        x[i] = 0;
        p[i] = b[i];
        r[i] = b[i];
        px[i] = 0;
    }


    int q = max_iterations;

    double alpha = 0;

    while (q--)
    {
        double sum = 0;
#pragma omp parallel  for num_threads(t) reduction(+ : sum)
        for (int i = 0; i < n; i++)
        {
            sum = r[i] * r[i] + sum;
        }

        double* temp = new double[n];
#pragma omp parallel for num_threads(t)
        for (int i = 0; i < n; i++)
        {
            temp[i] = 0;
        }

        double num = 0;
#pragma omp parallel for num_threads(t)
        for (int i = 0; i < n; i++)
        {
            double tmpory = temp[i];
#pragma omp parallel for reduction(+ : tmpory)
            for (int j = 0; j < n; j++)
            {
                temp[i] = A[i * n + j] * p[j] + temp[i];
            }
        }
#pragma omp parallel for num_threads(t) reduction(+ : num)
        for (int j = 0; j < n; j++)
        {
            num = num + temp[j] * p[j];
        }

        alpha = sum / num;

#pragma omp parallel for num_threads(t)
        for (int i = 0; i < n; i++)
        {
            px[i] = x[i];
            x[i] = x[i] + alpha * p[i];
            r[i] = r[i] - alpha * temp[i];
        }

        double beta = 0;
#pragma omp parallel for num_threads(t) reduction(+ : beta)
        for (int i = 0; i < n; i++)
        {
            beta = beta + r[i] * r[i];
        }

        beta = beta / sum;

#pragma omp parallel for num_threads(t)
        for (int i = 0; i < n; i++)
        {
            p[i] = r[i] + beta * p[i];
        }

        int c = 0;
        for (int i = 0; i < n; i++)
        {
            if (r[i] < 0.000001)
                c++;
        }

        if (c == n)
            break;
    }

    for (int i = 0; i < n; i++)
        cout << "\t\t" << x[i] << endl;
}

void conjugategradient_s(double* A, double* b, double* x, int n)
{
    int max_iterations;
    cout << "\nEnter number of iterations: ";
    cin >> max_iterations;
    double* r = new double[n];
    double* p = new double[n];
    double* px = new double[n];
    for (int i = 0; i < n; i++)
    {
        x[i] = 0;
        p[i] = b[i];
        r[i] = b[i];
        px[i] = 0;
    }
    double alpha = 0;
    while (max_iterations--)
    {

        double sum = 0;
        for (int i = 0; i < n; i++)
        {
            sum = r[i] * r[i] + sum;
        }

        double* temp = new double[n];
        for (int i = 0; i < n; i++)
        {
            temp[i] = 0;
        }

        double num = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                temp[i] = A[i * n + j] * p[j] + temp[i];
            }
        }
        for (int j = 0; j < n; j++)
        {
            num = num + temp[j] * p[j];
        }

        alpha = sum / num;
        for (int i = 0; i < n; i++)
        {
            px[i] = x[i];
            x[i] = x[i] + alpha * p[i];
            r[i] = r[i] - alpha * temp[i];
        }
        double beta = 0;
        for (int i = 0; i < n; i++)
        {
            beta = beta + r[i] * r[i];
        }
        beta = beta / sum;
        for (int i = 0; i < n; i++)
        {
            p[i] = r[i] + beta * p[i];
        }
        int c = 0;
        for (int i = 0; i < n; i++)
        {
            if (r[i] < 0.000001)
                c++;
        }
        if (c == n)
            break;
    }
    for (int i = 0; i < n; i++)
        cout << "\t\t" << x[i] << endl;
}
