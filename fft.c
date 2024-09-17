// #define  _POSIX_C_SOURCE 1

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include <getopt.h>
#include <sys/time.h>
#include <assert.h>
#include <math.h>
#include <complex.h>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <immintrin.h>

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;

double cutoff = 500;
u64 seed = 0;
u64 size = 0;
char *filename = NULL;
int version = 1;

typedef struct complexArray_
{
    double *real;
    double *imag;
} complexArray;
/******************** pseudo-random function (SPECK-like) ********************/

#define ROR(x, r) ((x >> r) | (x << (64 - r)))
#define ROL(x, r) ((x << r) | (x >> (64 - r)))
#define R(x, y, k) (x = ROR(x, 8), x += y, x ^= k, y = ROL(y, 3), y ^= x)
u64 PRF(u64 seed, u64 IV, u64 i)
{
    u64 y = i;
    u64 x = 0xBaadCafeDeadBeefULL;
    u64 b = IV;
    u64 a = seed;
    R(x, y, b);
    for (int i = 0; i < 32; i++)
    {
        R(a, b, i);
        R(x, y, b);
    }
    return x + i;
}

/************************** Fast Fourier Transform ***************************/
/* This code assumes that n is a power of two !!!                            */
/*****************************************************************************/

double complex *compute_powers_omega(u64 N){
    /* Compute the powers of omega from 0 to n-1. 
    V1 and V2 are very very similar in terms of performance (and same result).
    */

    double complex *powers_omega = (double complex *)malloc(sizeof(double complex) * N);
    if (powers_omega == NULL)
    {
        err(1, "compute_powers_omega malloc error");
    }

    // VERSION 1.
    double complex omega_n = cexp(-2*I*M_PI / N);   /* n-th root of unity*/
    if (N < 1024){
        powers_omega[0] = 1;
        for (int i = 1; i < N; i++){
            powers_omega[i] = powers_omega[i-1] * omega_n;
        }
        return powers_omega;
    }

    #pragma omp parallel
    {
        int p = omp_get_num_threads();
        int my_rank = omp_get_thread_num();

        // Split equally the interval [0: N-1].
        int start = my_rank * (N/p);
        int end = (my_rank+1) * (N/p);

        // Get the first omega of each interval.
        if (my_rank == 0){
            powers_omega[start] = 1;
        } else {
            powers_omega[start] = cexp(-2*I*M_PI*start / N);
        }

        // Compute the next powers.
        for (int i = start+1; i < end; i++){
            powers_omega[i] = powers_omega[i-1] * omega_n;
        }
    }
    return powers_omega;

}

void compute_powers_omega_double(u64 N, double* real, double* imag){
    /* Compute the powers of omega from 0 to n-1. 
    V1 and V2 are very very similar in terms of performance (and same result).
    */
    // VERSION 1.
    double omega_nReal = cos(-2*M_PI / N); 
    double omega_nImag = sin(-2*M_PI / N);
    if (N < 1024){
        real[0] = 1;
        imag[0] = 0;
        for (int i = 1; i < N; i++){
            real[i] = (real[i-1] * omega_nReal) - (imag[i-1] * omega_nImag);
            imag[i] = (real[i-1] * omega_nImag) + (imag[i-1] * omega_nReal);
        }
        return;
    }

    #pragma omp parallel
    {
        int p = omp_get_num_threads();
        int my_rank = omp_get_thread_num();

        // Split equally the interval [0: N-1].
        int start = my_rank * (N/p);
        int end = (my_rank+1) * (N/p);


        // Get the first omega of each interval.
        if (my_rank == 0){
            real[start] = 1;
            imag[start] = 0;
        } else {
            real[start] = cos(-2*M_PI*start / N);
            imag[start] = sin(-2*M_PI*start / N);
        }

        // Compute the next powers.
        for (int i = start+1; i < end; i++){
            real[i] = (real[i-1] * omega_nReal) - (imag[i-1] * omega_nImag);
            imag[i] = (real[i-1] * omega_nImag) + (imag[i-1] * omega_nReal);
        }
    }
}

void compute_powers_omega_double2(u64 N, double* real, double* imag){
    // compute the powers of omega

    int div = 1;
    u64 num = N;
    while (num >2)
    {
        num /= 2;
        div ++;
    }

    #pragma omp parallel for
    for(int i = 0; i< div; i++){
        u64 N2 = 1;
        for(int j = 0; j <= i; j++){
            N2 *=2;
        }
        double omega_nReal = cos(-2*M_PI / N2); 
        double omega_nImag = sin(-2*M_PI / N2);
        int pos = N-N2;
        real[pos] = 1;
        imag[pos] = 0;
        pos++;
        for(int i = 1; i < N2/2; i++){
            real[pos] = (real[pos-1] * omega_nReal) - (imag[pos-1] * omega_nImag);
            imag[pos] = (real[pos-1] * omega_nImag) + (imag[pos-1] * omega_nReal);
            pos++;
        }
    }
}

void FFT_rec_small(u64 n, const double *XR, const double *XI, double *YR, double *YI, u64 stride)
{
    if (n == 1)
    {
        YR[0] = XR[0];
        YI[0] = XI[0];
        return;
    }

    double complex omega_n = cexp(-2 * I * M_PI / n); /* n-th root of unity*/
    double complex omega = 1;
    double omegaR = creal(omega);
    double omegaI = cimag(omega); /* twiddle factor */
    double omega_nR = creal(omega_n);
    double omega_nI = cimag(omega_n);


    FFT_rec_small(n / 2, XR, XI, YR, YI, 2 * stride);
    FFT_rec_small(n / 2, XR + stride, XI + stride, YR + n / 2, YI + n / 2, 2 * stride);

    for (u64 i = 0; i < n / 2; i++)
    {
        double pReal = YR[i];
        double pImg = YI[i];

        double qReal = (YR[i + n / 2] * omegaR) - (YI[i + n / 2] * omegaI);
        double qImg = (YR[i + n / 2] * omegaI) + (YI[i + n / 2] * omegaR);


        YR[i] = pReal + qReal;
        YI[i] = pImg + qImg;

        YR[i + n / 2] = pReal - qReal;
        YI[i + n / 2] = pImg - qImg;

        double tmpR = omegaR * omega_nR - omegaI * omega_nI;
        double tmpI = omegaR * omega_nI + omegaI * omega_nR;
        omegaR = tmpR;
        omegaI = tmpI;
    }
}

void FFT_rec_small_V2(u64 n, const double *XR, const double *XI, double *YR, double *YI, u64 stride, double *powers_omegaReal, double * powers_omegaImag)
{
    if (n == 1)
    {
        YR[0] = XR[0];
        YI[0] = XI[0];
        return;
    }
    FFT_rec_small_V2(n / 2, XR, XI, YR, YI, 2 * stride, powers_omegaReal, powers_omegaImag);
    FFT_rec_small_V2(n / 2, XR + stride, XI + stride, YR + n / 2, YI + n / 2, 2 * stride, powers_omegaReal, powers_omegaImag);

    // n/2 % 4 !=0
    if(n <=4){
        for (u64 i = 0; i < n / 2; i++)
        {

            double omegaR = powers_omegaReal[i * stride];
            double omegaI = powers_omegaImag[i * stride];

            // Equivalent to: p = Y[i].
            double pReal = YR[i];
            double pImg = YI[i];


            // Equivalent to: q = Y[n/2 + i].
            double qReal = (YR[n/2 + i] * omegaR) - (YI[n/2 + i] * omegaI);
            double qImg = (YR[n/2 + i] * omegaI) + (YI[n/2 + i] * omegaR);

            // Equivalent to: Y[i] = p + q and Y[n/2 + i] = p - q.
            YR[i] = pReal + qReal;
            YI[i] = pImg + qImg;
            YR[i + n / 2] = pReal - qReal;
            YI[i + n / 2] = pImg - qImg;
        }
    }else{
        // n/2 % 4 == 0
        for (u64 i = 0; i < n / 2; i +=4)
        {

            // Load data
            __m256d v_omegaR = _mm256_set_pd(powers_omegaReal[(i+3) * stride],powers_omegaReal[(i+2) * stride],powers_omegaReal[(i+1) * stride],powers_omegaReal[i * stride]);
            __m256d v_omegaI = _mm256_set_pd(powers_omegaImag[(i+3) * stride],powers_omegaImag[(i+2) * stride],powers_omegaImag[(i+1) * stride],powers_omegaImag[i * stride]);




            // load YR[i] and YI[i]
            __m256d v_pReal = _mm256_load_pd(&YR[i]);
            __m256d v_pImg = _mm256_load_pd(&YI[i]);

            // load YR[n/2 + i] and YI[n/2 + i]
            __m256d v_YRn2 = _mm256_load_pd(&YR[n/2 + i]);
            __m256d v_YIn2 = _mm256_load_pd(&YI[n/2 + i]);

            // Equivalent to: q = Y[n/2 + i].
            __m256d v_qReal = _mm256_sub_pd(_mm256_mul_pd(v_YRn2, v_omegaR), _mm256_mul_pd(v_YIn2, v_omegaI));
            __m256d v_qImg = _mm256_add_pd(_mm256_mul_pd(v_YRn2, v_omegaI), _mm256_mul_pd(v_YIn2, v_omegaR));

            // Equivalent to: Y[i] = p + q and Y[n/2 + i] = p - q
            __m256d v_YR = _mm256_add_pd(v_pReal, v_qReal);
            __m256d v_YI = _mm256_add_pd(v_pImg, v_qImg);

            _mm256_store_pd(&YR[i], v_YR);
            _mm256_store_pd(&YI[i], v_YI);


            __m256d v_YR_n = _mm256_sub_pd(v_pReal, v_qReal);
            __m256d v_YI_n = _mm256_sub_pd(v_pImg, v_qImg);

            _mm256_store_pd(&YR[i + n / 2], v_YR_n);
            _mm256_store_pd(&YI[i + n / 2], v_YI_n);
        }
    }
}

void FFT_rec_small_V3(u64 n, const double *XR, const double *XI, double *YR, double *YI, u64 stride, double *powers_omegaReal, double *powers_omegaImag)
{
    if (n == 1)
    {
        YR[0] = XR[0];
        YI[0] = XI[0];
        return;
    }
    FFT_rec_small_V3(n / 2, XR, XI, YR, YI, 2 * stride, powers_omegaReal + n / 2, powers_omegaImag + n / 2);
    FFT_rec_small_V3(n / 2, XR + stride, XI + stride, YR + n / 2, YI + n / 2, 2 * stride, powers_omegaReal + n / 2, powers_omegaImag + n / 2);

    // n/2 % 4 !=0
    if (n <= 4){
        for (u64 i = 0; i < n / 2; i++)
        {
            
            double omegaR = powers_omegaReal[i];
            double omegaI = powers_omegaImag[i];
            // Equivalent to: p = Y[i].
            double pReal = YR[i];
            double pImg = YI[i];


            // Equivalent to: q = Y[n/2 + i].
            double qReal = (YR[n/2 + i] * omegaR) - (YI[n/2 + i] * omegaI);
            double qImg = (YR[n/2 + i] * omegaI) + (YI[n/2 + i] * omegaR);


            // Equivalent to: Y[i] = p + q and Y[n/2 + i] = p - q.
            YR[i] = pReal + qReal;
            YI[i] = pImg + qImg;
            YR[i + n / 2] = pReal - qReal;
            YI[i + n / 2] = pImg - qImg;
        }
    }else{
        // n/2 % 4 == 0
        for (u64 i = 0; i < n / 2; i +=4)
        {



            // Load data
            __m256d v_omegaR = _mm256_load_pd(&powers_omegaReal[i]);
            __m256d v_omegaI = _mm256_load_pd(&powers_omegaImag[i]);

            // Equivalent to: p = Y[i].
            // double pReal = YR[i];
            // double pImg = YI[i];

            // load YR[i] and YI[i]
            __m256d v_pReal = _mm256_load_pd(&YR[i]);
            __m256d v_pImg = _mm256_load_pd(&YI[i]);

            // load YR[n/2 + i] and YI[n/2 + i]
            __m256d v_YRn2 = _mm256_load_pd(&YR[n/2 + i]);
            __m256d v_YIn2 = _mm256_load_pd(&YI[n/2 + i]);

            // Equivalent to: q = Y[n/2 + i].


            __m256d v_qReal = _mm256_sub_pd(_mm256_mul_pd(v_YRn2, v_omegaR), _mm256_mul_pd(v_YIn2, v_omegaI));
            __m256d v_qImg = _mm256_add_pd(_mm256_mul_pd(v_YRn2, v_omegaI), _mm256_mul_pd(v_YIn2, v_omegaR));

            // Equivalent to: Y[i] = p + q and Y[n/2 + i] = p - q.


            __m256d v_YR = _mm256_add_pd(v_pReal, v_qReal);
            __m256d v_YI = _mm256_add_pd(v_pImg, v_qImg);

            _mm256_store_pd(&YR[i], v_YR);
            _mm256_store_pd(&YI[i], v_YI);


            __m256d v_YR_n = _mm256_sub_pd(v_pReal, v_qReal);
            __m256d v_YI_n = _mm256_sub_pd(v_pImg, v_qImg);

            _mm256_store_pd(&YR[i + n / 2], v_YR_n);
            _mm256_store_pd(&YI[i + n / 2], v_YI_n);
        }
    }
}

void FFT_iter_1_V1(u64 n, const complexArray *X, complexArray *Y)
{
    /* FFT: Iterative with only 1 sum version and without the powers
    of omega in parameter.
    Parallelism: NO because of the dependance omega *= omega_n.
    Performance: A bit slower than FFT_rec but OK.
    */

    if ((n & (n - 1)) != 0)
        errx(1, "size is not a power of two (this code does not handle other cases)");

    double complex omega_n = cexp(-2 * I * M_PI / n); /* n-th root of unity*/
    double complex omega_k = 1;

    for (u64 k = 0; k < n; k++)
    {

        double complex sum = 0;
        double complex omega = omega_k;

        for (u64 j = 0; j < n; j++)
        {

            // Equivalent of: sum += X[j] * omega_n^{jk}.
            double xR = X->real[j];
            double xI = X->imag[j];

            if (k == 0 || j == 0)
            {
                sum += (xR + I * xI) * 1;
            }
            else
            {
                sum += (xR + I * xI) * omega;
            }

            // Equivalent of: omega *= omega_k.
            if (j != 0)
            {
                omega *= omega_k;
            }
        }

        Y->real[k] = creal(sum);
        Y->imag[k] = cimag(sum);

        // Equivalent of: omega_k *= omega_n.
        omega_k *= omega_n;
    }
}

void FFT_iter_1_V2(u64 n, const complexArray *X, complexArray *Y)
{
    /* FFT: Iterative with only 1 sum version and WITH the powers
    of omega.
    Parallelism: YES we removed the dependance omega *= omega_n.
    Performance: Better than V1 but slower than REC.
    */

    if ((n & (n - 1)) != 0)
        errx(1, "size is not a power of two (this code does not handle other cases)");

    double complex *powers_omega = compute_powers_omega(n);

    #pragma omp parallel for
    for (u64 k = 0; k < n; k++)
    {
        double complex sum = 0;


        for (u64 j = 0; j < n; j++)
        {

            double complex omega = powers_omega[j * k % n];
            double xR = X->real[j];
            double xI = X->imag[j];
            sum += (xR + I * xI) * omega;
        }

        Y->real[k] = creal(sum);
        Y->imag[k] = cimag(sum);
    }

    free(powers_omega);
}

void FFT_iter_2_V2(u64 n, const complexArray *X, complexArray *Y)
{
    /* FFT: Iterative with 2 sums version WITH the powers
    of omega.
    Parallelism: YES but not finished.
    Performance: ?.
    */
    if ((n & (n - 1)) != 0)
        errx(1, "size is not a power of two (this code does not handle other cases)");

    // We suppose that: n = N1 * N2.

    u64 N1 = (u64) sqrt(n);
    u64 N2 = (u64) sqrt(n);
    double* powers_omegaReal = (double *)aligned_alloc(32, n * sizeof(double));
    double* powers_omegaImag = (double *)aligned_alloc(32, n * sizeof(double));
    compute_powers_omega_double(n, powers_omegaReal, powers_omegaImag);

    double* powers_omega_N1Real = (double *)aligned_alloc(32, N1 * sizeof(double));
    double* powers_omega_N1Imag = (double *)aligned_alloc(32, N1 * sizeof(double));
    compute_powers_omega_double(N1, powers_omega_N1Real, powers_omega_N1Imag);

    #pragma omp parallel for
    for (int k1 = 0; k1 < N2; k1++){ // k1 < N2 or k1 < N1?
        // #pragma omp parallel for
        for (int k2 = 0; k2 < N1; k2++){

            double sumReal = 0;
            double sumImag = 0;

            for (int j2 = 0; j2 < N2; j2++){
                double sum2Real = 0;
                double sum2Imag = 0;

                // #pragma omp parallel for reduction (+:sum2)
                for (int j1 = 0; j1 < N1; j1++){


                    // sum += X[j1n2 + j2] * omega^{j1k1}
                    double xR = X->real[j1*N2 + j2];
                    double xI = X->imag[j1*N2 + j2];
                    
                    // sum2 += (xR + I*xI) * powers_omega_N1[j1*k1 % N1];
                    sum2Real += (xR * powers_omega_N1Real[j1*k1 % N1]) - (xI * powers_omega_N1Imag[j1*k1 % N1]);
                    sum2Imag += (xR * powers_omega_N1Imag[j1*k1 % N1]) + (xI * powers_omega_N1Real[j1*k1 % N1]);
                }

                // sum2 *= powers_omega[j2*k1];
                double tmp = (sum2Real * powers_omegaReal[j2*k1]) - (sum2Imag * powers_omegaImag[j2*k1]);
                sum2Imag = (sum2Real * powers_omegaImag[j2*k1]) + (sum2Imag * powers_omegaReal[j2*k1]);
                sum2Real = tmp;

                // sum += sum2 * powers_omega_N1[j2*k2 % N1];
                sumReal += (sum2Real * powers_omega_N1Real[j2*k2 % N2]) - (sum2Imag * powers_omega_N1Imag[j2*k2 % N2]);
                sumImag += (sum2Real * powers_omega_N1Imag[j2*k2 % N2]) + (sum2Imag * powers_omega_N1Real[j2*k2 % N2]);  
            }
            Y->real[k1 + k2*N1] = sumReal;
            Y->imag[k1 + k2*N1] = sumImag;
        }
    }

}

void FFT_rec_V1(u64 n, const double *XR, const double *XI, double *YR, double *YI, u64 stride)
{
    /* V1: Without powers_omega in parameter.
    Performance: Faster than V2.
    */
    if (n == 1)
    {
        YR[0] = XR[0];
        YI[0] = XI[0];
        return;
    }

    double complex omega_n = cexp(-2 * I * M_PI / n); /* n-th root of unity*/
    double complex omega = 1;                         /* twiddle factor */
    double omegaR = creal(omega);
    double omegaI = cimag(omega);
    double omega_nR = creal(omega_n);
    double omega_nI = cimag(omega_n);

    if (n < 2048)
    {
        FFT_rec_small(n / 2, XR, XI, YR, YI, 2 * stride);
        FFT_rec_small(n / 2, XR + stride, XI + stride, YR + n / 2, YI + n / 2, 2 * stride);
    }
    else
    {
        #pragma omp task
        FFT_rec_V1(n / 2, XR, XI, YR, YI, 2 * stride);
        #pragma omp task
        FFT_rec_V1(n / 2, XR + stride, XI + stride, YR + n / 2, YI + n / 2, 2 * stride);
        #pragma omp taskwait
    }

    for (u64 i = 0; i < n / 2; i++)
    {
        // Equivalent to: p = Y[i].
        double pReal = YR[i];
        double pImg = YI[i];
 

        // Equivalent to: q = Y[n/2 + i].
        double qReal = (YR[n/2 + i] * omegaR) - (YI[n/2 + i] * omegaI);
        double qImg = (YR[n/2 + i] * omegaI) + (YI[n/2 + i] * omegaR);


        // Equivalent to: Y[i] = p + q and Y[n/2 + i] = p - q.
        YR[i] = pReal + qReal;
        YI[i] = pImg + qImg;

        YR[n/2 + i] = pReal - qReal;
        YI[n/2 + i] = pImg - qImg;

        // Equivalent to: omega *= omega_n.
        double tmpR = omegaR * omega_nR - omegaI * omega_nI;
        double tmpI = omegaR * omega_nI + omegaI * omega_nR;
        omegaR = tmpR;
        omegaI = tmpI;
    }
}

void FFT_rec_V2(u64 n, const double *XR, const double *XI, double *YR, double *YI, u64 stride, double *powers_omegaReal, double * powers_omegaImag)
{
    /* V2: With powers_omega in parameter.
    Performance: Slower than V1.
    */
    if (n == 1)
    {
        YR[0] = XR[0];
        YI[0] = XI[0];
        return;
    }

    if (n < 2048)
    {
        FFT_rec_small_V2(n / 2, XR, XI, YR, YI, 2 * stride, powers_omegaReal, powers_omegaImag);
        FFT_rec_small_V2(n / 2, XR + stride, XI + stride, YR + n / 2, YI + n / 2, 2 * stride, powers_omegaReal, powers_omegaImag);
    }
    else
    {
        #pragma omp task
        FFT_rec_V2(n / 2, XR, XI, YR, YI, 2 * stride, powers_omegaReal, powers_omegaImag);
        #pragma omp task
        FFT_rec_V2(n / 2, XR + stride, XI + stride, YR + n / 2, YI + n / 2, 2 * stride, powers_omegaReal, powers_omegaImag);
        #pragma omp taskwait
    }

    // n/2 % 4 !=0
    if(n <=4){
        for (u64 i = 0; i < n / 2; i++)
        {

            double omegaR = powers_omegaReal[i * stride];
            double omegaI = powers_omegaImag[i * stride];

            // Equivalent to: p = Y[i].
            double pReal = YR[i];
            double pImg = YI[i];

            // Equivalent to: q = Y[n/2 + i].
            double qReal = (YR[n/2 + i] * omegaR) - (YI[n/2 + i] * omegaI);
            double qImg = (YR[n/2 + i] * omegaI) + (YI[n/2 + i] * omegaR);

            // Equivalent to: Y[i] = p + q and Y[n/2 + i] = p - q.
            YR[i] = pReal + qReal;
            YI[i] = pImg + qImg;
            YR[i + n / 2] = pReal - qReal;
            YI[i + n / 2] = pImg - qImg;
        }
    }else{
        // n/2 % 4 == 0
        for (u64 i = 0; i < n / 2; i +=4)
        {


            // Load data
            __m256d v_omegaR = _mm256_set_pd(powers_omegaReal[(i+3) * stride],powers_omegaReal[(i+2) * stride],powers_omegaReal[(i+1) * stride],powers_omegaReal[i * stride]);
            __m256d v_omegaI = _mm256_set_pd(powers_omegaImag[(i+3) * stride],powers_omegaImag[(i+2) * stride],powers_omegaImag[(i+1) * stride],powers_omegaImag[i * stride]);


            // Equivalent to: p = Y[i].
            // double pReal = YR[i];
            // double pImg = YI[i];

            // load YR[i] and YI[i]
            __m256d v_pReal = _mm256_load_pd(&YR[i]);
            __m256d v_pImg = _mm256_load_pd(&YI[i]);

            // load YR[n/2 + i] and YI[n/2 + i]
            __m256d v_YRn2 = _mm256_load_pd(&YR[n/2 + i]);
            __m256d v_YIn2 = _mm256_load_pd(&YI[n/2 + i]);

            // Equivalent to: q = Y[n/2 + i].

            __m256d v_qReal = _mm256_sub_pd(_mm256_mul_pd(v_YRn2, v_omegaR), _mm256_mul_pd(v_YIn2, v_omegaI));
            __m256d v_qImg = _mm256_add_pd(_mm256_mul_pd(v_YRn2, v_omegaI), _mm256_mul_pd(v_YIn2, v_omegaR));

            // Equivalent to: Y[i] = p + q and Y[n/2 + i] = p - q.


            __m256d v_YR = _mm256_add_pd(v_pReal, v_qReal);
            __m256d v_YI = _mm256_add_pd(v_pImg, v_qImg);

            _mm256_store_pd(&YR[i], v_YR);
            _mm256_store_pd(&YI[i], v_YI);


            __m256d v_YR_n = _mm256_sub_pd(v_pReal, v_qReal);
            __m256d v_YI_n = _mm256_sub_pd(v_pImg, v_qImg);

            _mm256_store_pd(&YR[i + n / 2], v_YR_n);
            _mm256_store_pd(&YI[i + n / 2], v_YI_n);


        }
    }
    
}

void FFT_rec_V3(u64 n, const double *XR, const double *XI, double *YR, double *YI, u64 stride, double *powers_omegaReal, double * powers_omegaImag)
{
    /* V2: With powers_omega in parameter.
    Performance: Slower than V1.
    */
    if (n == 1)
    {
        YR[0] = XR[0];
        YI[0] = XI[0];
        return;
    }

    if (n < 2048)
    {

        FFT_rec_small_V3(n / 2, XR, XI, YR, YI, 2 * stride, powers_omegaReal + n/2, powers_omegaImag + n/2);

        FFT_rec_small_V3(n / 2, XR + stride, XI + stride, YR + n / 2, YI + n / 2, 2 * stride, powers_omegaReal + n/2, powers_omegaImag + n/2);

    }
    else
    {
        #pragma omp task
        FFT_rec_V3(n / 2, XR, XI, YR, YI, 2 * stride, powers_omegaReal + n/2, powers_omegaImag + n/2);
        #pragma omp task
        FFT_rec_V3(n / 2, XR + stride, XI + stride, YR + n / 2, YI + n / 2, 2 * stride, powers_omegaReal + n/2, powers_omegaImag + n/2);
        #pragma omp taskwait
    }

    // n/2 % 4 !=0
    if(n <=4){
        #pragma omp parallel for
        for (u64 i = 0; i < n / 2; i++)
        {

            double omegaR = powers_omegaReal[i];
            double omegaI = powers_omegaImag[i];

            // Equivalent to: p = Y[i].
            double pReal = YR[i];
            double pImg = YI[i];

            // Equivalent to: q = Y[n/2 + i].
            double qReal = (YR[n/2 + i] * omegaR) - (YI[n/2 + i] * omegaI);
            double qImg = (YR[n/2 + i] * omegaI) + (YI[n/2 + i] * omegaR);

            // Equivalent to: Y[i] = p + q and Y[n/2 + i] = p - q.
            YR[i] = pReal + qReal;
            YI[i] = pImg + qImg;
            YR[i + n / 2] = pReal - qReal;
            YI[i + n / 2] = pImg - qImg;
        }
    }else{
        // n/2 % 4 == 0
        #pragma omp parallel for
        for (u64 i = 0; i < n / 2; i +=4)
        {


            // Load data
            __m256d v_omegaR = _mm256_load_pd(&powers_omegaReal[i]);
            __m256d v_omegaI = _mm256_load_pd(&powers_omegaImag[i]);

            // Equivalent to: p = Y[i].

            // load YR[i] and YI[i]
            __m256d v_pReal = _mm256_load_pd(&YR[i]);
            __m256d v_pImg = _mm256_load_pd(&YI[i]);

            // load YR[n/2 + i] and YI[n/2 + i]
            __m256d v_YRn2 = _mm256_load_pd(&YR[n/2 + i]);
            __m256d v_YIn2 = _mm256_load_pd(&YI[n/2 + i]);

            // Equivalent to: q = Y[n/2 + i].

            __m256d v_qReal = _mm256_sub_pd(_mm256_mul_pd(v_YRn2, v_omegaR), _mm256_mul_pd(v_YIn2, v_omegaI));
            __m256d v_qImg = _mm256_add_pd(_mm256_mul_pd(v_YRn2, v_omegaI), _mm256_mul_pd(v_YIn2, v_omegaR));

            // Equivalent to: Y[i] = p + q and Y[n/2 + i] = p - q.

            __m256d v_YR = _mm256_add_pd(v_pReal, v_qReal);
            __m256d v_YI = _mm256_add_pd(v_pImg, v_qImg);

            _mm256_store_pd(&YR[i], v_YR);
            _mm256_store_pd(&YI[i], v_YI);


            __m256d v_YR_n = _mm256_sub_pd(v_pReal, v_qReal);
            __m256d v_YI_n = _mm256_sub_pd(v_pImg, v_qImg);

            _mm256_store_pd(&YR[i + n / 2], v_YR_n);
            _mm256_store_pd(&YI[i + n / 2], v_YI_n);


        }
    }
    
}

void FFT_V1(u64 n, const complexArray *X, complexArray *Y)
{
    /* FFT: Recursive without the powers of omega in parameter.
    Performance: ?.
    */

    /* sanity check */
    if ((n & (n - 1)) != 0)
    {
        errx(1, "size is not a power of two (this code does not handle other cases)");
    }

    #pragma omp parallel
    {
        #pragma omp single
        FFT_rec_V1(n, X->real, X->imag, Y->real, Y->imag, 1); /* stride == 1 initially */
    }
}

void FFT_V2(u64 n, const complexArray *X, complexArray *Y)
{
    /* FFT: Recursive WITH the powers of omega in parameter.
    Performance: ?.
    */
    /* sanity check */
    if ((n & (n - 1)) != 0)
    {
        errx(1, "size is not a power of two (this code does not handle other cases)");
    }

    // Allocate memory alligned
    double* powers_omegaReal = (double *)aligned_alloc(32, n * sizeof(double));
    double* powers_omegaImag = (double *)aligned_alloc(32, n * sizeof(double));
    compute_powers_omega_double(n,powers_omegaReal, powers_omegaImag);

    #pragma omp parallel
    {
        #pragma omp single
        FFT_rec_V2(n, X->real, X->imag, Y->real, Y->imag, 1, powers_omegaReal, powers_omegaImag); /* stride == 1 initially */
    }

    free(powers_omegaReal);
    free(powers_omegaImag);
}

void FFT_V3(u64 n, const complexArray *X, complexArray *Y)
{
    /* FFT: Recursive WITH the powers of omega in parameter.
    Performance: ?.
    */
    /* sanity check */
    if ((n & (n - 1)) != 0)
    {
        errx(1, "size is not a power of two (this code does not handle other cases)");
    }

    // Allocate memory aligned
    double* powers_omegaReal = (double *)aligned_alloc(32, n * sizeof(double));
    double* powers_omegaImag = (double *)aligned_alloc(32, n * sizeof(double));
    compute_powers_omega_double2(n,powers_omegaReal, powers_omegaImag);

    #pragma omp parallel
    {
        #pragma omp single
        FFT_rec_V2(n, X->real, X->imag, Y->real, Y->imag, 1, powers_omegaReal, powers_omegaImag); /* stride == 1 initially */
    }

    free(powers_omegaReal);
    free(powers_omegaImag);
}

/* Computes the inverse Fourier transform, but destroys the input */
void iFFT_V1(u64 n, complexArray *X, complexArray *Y)
{

    __m256d valpha = _mm256_set1_pd(-1.0);
    #pragma omp parallel for
    for (u64 i = 0; i < n; i += 4)
    {
        __m256d vu = _mm256_load_pd(&X->imag[i]);
        __m256d vres = _mm256_mul_pd(valpha, vu);
        _mm256_store_pd(&X->imag[i], vres);
    }

    FFT_V1(n, X, Y);

    __m256d inv_n = _mm256_set1_pd(n);
    __m256d minus_one = _mm256_set1_pd(-1.0);

    #pragma omp parallel for
    for (u64 i = 0; i < n; i += 4)
    {
        __m256d real_vals = _mm256_load_pd(&Y->real[i]);
        __m256d imag_vals = _mm256_load_pd(&Y->imag[i]);

        real_vals = _mm256_div_pd(real_vals, inv_n);
        imag_vals = _mm256_div_pd(imag_vals, inv_n);
        imag_vals = _mm256_mul_pd(imag_vals, minus_one);

        _mm256_store_pd(&Y->real[i], real_vals);
        _mm256_store_pd(&Y->imag[i], imag_vals);
    }
}

void iFFT_V2(u64 n, complexArray *X, complexArray *Y)
{

    __m256d valpha = _mm256_set1_pd(-1.0);
    #pragma omp parallel for
    for (u64 i = 0; i < n; i += 4)
    {
        __m256d vu = _mm256_load_pd(&X->imag[i]);
        __m256d vres = _mm256_mul_pd(valpha, vu);
        _mm256_store_pd(&X->imag[i], vres);
    }

    FFT_V2(n, X, Y);

    __m256d inv_n = _mm256_set1_pd(n);
    __m256d minus_one = _mm256_set1_pd(-1.0);

    #pragma omp parallel for
    for (u64 i = 0; i < n; i += 4)
    {
        __m256d real_vals = _mm256_load_pd(&Y->real[i]);
        __m256d imag_vals = _mm256_load_pd(&Y->imag[i]);

        real_vals = _mm256_div_pd(real_vals, inv_n);
        imag_vals = _mm256_div_pd(imag_vals, inv_n);
        imag_vals = _mm256_mul_pd(imag_vals, minus_one);

        _mm256_store_pd(&Y->real[i], real_vals);
        _mm256_store_pd(&Y->imag[i], imag_vals);
    }
}

void iFFT_V3(u64 n, complexArray *X, complexArray *Y)
{

    __m256d valpha = _mm256_set1_pd(-1.0);
    #pragma omp parallel for
    for (u64 i = 0; i < n; i += 4)
    {
        __m256d vu = _mm256_load_pd(&X->imag[i]);
        __m256d vres = _mm256_mul_pd(valpha, vu);
        _mm256_store_pd(&X->imag[i], vres);
    }

    FFT_V3(n, X, Y);

    __m256d inv_n = _mm256_set1_pd(n);
    __m256d minus_one = _mm256_set1_pd(-1.0);

    #pragma omp parallel for
    for (u64 i = 0; i < n; i += 4)
    {
        __m256d real_vals = _mm256_load_pd(&Y->real[i]);
        __m256d imag_vals = _mm256_load_pd(&Y->imag[i]);

        real_vals = _mm256_div_pd(real_vals, inv_n);
        imag_vals = _mm256_div_pd(imag_vals, inv_n);
        imag_vals = _mm256_mul_pd(imag_vals, minus_one);

        _mm256_store_pd(&Y->real[i], real_vals);
        _mm256_store_pd(&Y->imag[i], imag_vals);
    }
}

void iFFT_iter_1_V1(u64 n, complexArray *X, complexArray *Y)
{

    __m256d valpha = _mm256_set1_pd(-1.0);
    #pragma omp parallel for
    for (u64 i = 0; i < n; i += 4)
    {
        __m256d vu = _mm256_load_pd(&X->imag[i]);
        __m256d vres = _mm256_mul_pd(valpha, vu);
        _mm256_store_pd(&X->imag[i], vres);
    }

    FFT_iter_1_V1(n, X, Y);

    __m256d inv_n = _mm256_set1_pd(n);
    __m256d minus_one = _mm256_set1_pd(-1.0);

    #pragma omp parallel for
    for (u64 i = 0; i < n; i += 4)
    {
        __m256d real_vals = _mm256_load_pd(&Y->real[i]);
        __m256d imag_vals = _mm256_load_pd(&Y->imag[i]);

        real_vals = _mm256_div_pd(real_vals, inv_n);
        imag_vals = _mm256_div_pd(imag_vals, inv_n);
        imag_vals = _mm256_mul_pd(imag_vals, minus_one);

        _mm256_store_pd(&Y->real[i], real_vals);
        _mm256_store_pd(&Y->imag[i], imag_vals);
    }
}

void iFFT_iter_1_V2(u64 n, complexArray *X, complexArray *Y)
{

    __m256d valpha = _mm256_set1_pd(-1.0);
    #pragma omp parallel for
    for (u64 i = 0; i < n; i += 4)
    {
        __m256d vu = _mm256_load_pd(&X->imag[i]);
        __m256d vres = _mm256_mul_pd(valpha, vu);
        _mm256_store_pd(&X->imag[i], vres);
    }

    FFT_iter_1_V2(n, X, Y);

    __m256d inv_n = _mm256_set1_pd(n);
    __m256d minus_one = _mm256_set1_pd(-1.0);

    #pragma omp parallel for
    for (u64 i = 0; i < n; i += 4)
    {
        __m256d real_vals = _mm256_load_pd(&Y->real[i]);
        __m256d imag_vals = _mm256_load_pd(&Y->imag[i]);

        real_vals = _mm256_div_pd(real_vals, inv_n);
        imag_vals = _mm256_div_pd(imag_vals, inv_n);
        imag_vals = _mm256_mul_pd(imag_vals, minus_one);

        _mm256_store_pd(&Y->real[i], real_vals);
        _mm256_store_pd(&Y->imag[i], imag_vals);
    }
}


void iFFT_iter_2_V2(u64 n, complexArray *X, complexArray *Y)
{

    __m256d valpha = _mm256_set1_pd(-1.0);
    #pragma omp parallel for
    for (u64 i = 0; i < n; i += 4)
    {
        __m256d vu = _mm256_load_pd(&X->imag[i]);
        __m256d vres = _mm256_mul_pd(valpha, vu);
        _mm256_store_pd(&X->imag[i], vres);
    }

    FFT_iter_2_V2(n, X, Y);

    __m256d inv_n = _mm256_set1_pd(n);
    __m256d minus_one = _mm256_set1_pd(-1.0);

    #pragma omp parallel for
    for (u64 i = 0; i < n; i += 4)
    {
        __m256d real_vals = _mm256_load_pd(&Y->real[i]);
        __m256d imag_vals = _mm256_load_pd(&Y->imag[i]);

        real_vals = _mm256_div_pd(real_vals, inv_n);
        imag_vals = _mm256_div_pd(imag_vals, inv_n);
        imag_vals = _mm256_mul_pd(imag_vals, minus_one);

        _mm256_store_pd(&Y->real[i], real_vals);
        _mm256_store_pd(&Y->imag[i], imag_vals);
    }
}
/******************* utility functions ********************/

double wtime()
{
    struct timeval ts;
    gettimeofday(&ts, NULL);
    return (double)ts.tv_sec + ts.tv_usec / 1e6;
}

void process_command_line_options(int argc, char **argv)
{
    struct option longopts[6] = {
        {"size", required_argument, NULL, 'n'},
        {"seed", required_argument, NULL, 's'},
        {"output", required_argument, NULL, 'o'},
        {"cutoff", required_argument, NULL, 'c'},
        {"version", required_argument, NULL, 'v'},
        {NULL, 0, NULL, 0}};
    char ch;
    while ((ch = getopt_long(argc, argv, "", longopts, NULL)) != -1)
    {
        switch (ch)
        {
        case 'n':
            size = atoll(optarg);
            break;
        case 's':
            seed = atoll(optarg);
            break;
        case 'o':
            filename = optarg;
            break;
        case 'c':
            cutoff = atof(optarg);
            break;
        case 'v':
            version = atoi(optarg);
            break;
        default:
            errx(1, "Unknown option\n");
        }
    }
    /* validation */
    if (size == 0)
        errx(1, "missing --size argument");
}

/* save at most 10s of sound output in .WAV format */
void save_WAV(char *filename, u64 size, complexArray *C)
{
    assert(size < 1000000000);
    FILE *f = fopen(filename, "w");
    if (f == NULL)
        err(1, "fopen");
    printf("Writing <= 10s of audio output in %s\n", filename);
    u32 rate = 44100; // Sample rate
    u32 frame_count = 10 * rate;
    if (size < frame_count)
        frame_count = size;
    u16 chan_num = 2; // Number of channels
    u16 bits = 16;    // Bit depth
    u32 length = frame_count * chan_num * bits / 8;
    u16 byte;
    double multiplier = 32767;

    /* WAVE Header Data */
    fwrite("RIFF", 1, 4, f);
    u32 chunk_size = length + 44 - 8;
    fwrite(&chunk_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    u32 subchunk1_size = 16;
    fwrite(&subchunk1_size, 4, 1, f);
    u16 fmt_type = 1; // 1 = PCM
    fwrite(&fmt_type, 2, 1, f);
    fwrite(&chan_num, 2, 1, f);
    fwrite(&rate, 4, 1, f);
    // (Sample Rate * BitsPerSample * Channels) / 8
    uint32_t byte_rate = rate * bits * chan_num / 8;
    fwrite(&byte_rate, 4, 1, f);
    uint16_t block_align = chan_num * bits / 8;
    fwrite(&block_align, 2, 1, f);
    fwrite(&bits, 2, 1, f);

    /* Marks the start of the data */
    fwrite("data", 1, 4, f);
    fwrite(&length, 4, 1, f); // Data size
    for (u32 i = 0; i < frame_count; i++)
    {
        byte = C->real[i] * multiplier;
        fwrite(&byte, 2, 1, f);
        byte = C->imag[i] * multiplier;
        fwrite(&byte, 2, 1, f);
    }
    fclose(f);
}

/*************************** main function *********************************/

int main(int argc, char **argv)
{
    process_command_line_options(argc, argv);
    printf("nb thread dispo = %d\n",omp_get_max_threads());
    /* generate white noise */
    complexArray *A = malloc(sizeof(*A));
    complexArray *B = malloc(sizeof(*B));
    complexArray *C = malloc(sizeof(*C));

    A->real = (double *)aligned_alloc(32, size * sizeof(double));
    A->imag = (double *)aligned_alloc(32, size * sizeof(double));

    B->real = (double *)aligned_alloc(32, size * sizeof(double));
    B->imag = (double *)aligned_alloc(32, size * sizeof(double));

    C->real = (double *)aligned_alloc(32, size * sizeof(double));
    C->imag = (double *)aligned_alloc(32, size * sizeof(double));

    printf("Generating white noise...\n");
    double beginTime = wtime();

    #pragma omp parallel for
    for (u64 i = 0; i < size; i++)
    {
        double real = 2 * (PRF(seed, 0, i) * 5.42101086242752217e-20) - 1;
        double imag = 2 * (PRF(seed, 1, i) * 5.42101086242752217e-20) - 1;
        A->real[i] = real;
        A->imag[i] = imag;
    }

    printf("Forward FFT...\n");

    if (version == 1)
        FFT_V1(size, A, B);
    else if (version == 2)
        FFT_V2(size, A, B);
    else if (version == 3)
        FFT_V3(size, A, B);
    else if (version == 4)
        FFT_iter_1_V1(size, A, B);
    else if (version == 5)
        FFT_iter_1_V2(size, A, B);
    else if (version == 6)
        FFT_iter_2_V2(size, A, B);

    /* damp fourrier coefficients */
    printf("Adjusting Fourier coefficients...\n");

    #pragma omp parallel for
    for (u64 i = 0; i < size; i++)
    {

        double tmp = sin(i * 2 * M_PI / 44100);

        double cexp_real = cos(-i * 2 * M_PI / 4 / 44100);
        double cexp_imag = sin(-i * 2 * M_PI / 4 / 44100);

        double tmpBreal = (B->real[i] * cexp_real - B->imag[i] * cexp_imag) * tmp;
        B->imag[i] = (B->real[i] * cexp_imag + B->imag[i] * cexp_real) * tmp;
        B->real[i] = tmpBreal;

        B->real[i] *= (i + 1) / exp((i * cutoff) / size);
        B->imag[i] *= (i + 1) / exp((i * cutoff) / size);
    }

    printf("Inverse FFT...\n");
    if (version == 1)
        iFFT_V1(size, B, C);
    else if (version == 2)
        iFFT_V2(size, B, C);
    else if (version == 3)
        iFFT_V3(size, B, C);
    else if (version == 4)
        iFFT_iter_1_V1(size, B, C);
    else if (version == 5)
        iFFT_iter_1_V2(size, B, C);
    else if (version == 6)
        iFFT_iter_2_V2(size, B, C);

    printf("Normalizing output...\n");
    double max = 0;

    #pragma omp parallel for reduction(max : max)
    for (u64 i = 0; i < size; i++)
    {
        max = fmax(max, sqrt((C->real[i] * C->real[i]) + (C->imag[i] * C->imag[i])));
    }
    printf("max = %g\n", max);

    __m256d inv_max = _mm256_set1_pd(max);
    #pragma omp parallel for
    for (u64 i = 0; i < size; i += 4)
    {

        __m256d real_vals = _mm256_load_pd(&C->real[i]);
        __m256d imag_vals = _mm256_load_pd(&C->imag[i]);

        real_vals = _mm256_div_pd(real_vals, inv_max);
        imag_vals = _mm256_div_pd(imag_vals, inv_max);

        _mm256_store_pd(&C->real[i], real_vals);
        _mm256_store_pd(&C->imag[i], imag_vals);
    }

    double endTime = wtime();
    double totalTime = endTime - beginTime;
    printf("Time = %fs\n", totalTime);

    if (filename != NULL)
        save_WAV(filename, size, C);

    exit(EXIT_SUCCESS);
}