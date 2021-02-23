#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>
//--------------------------------------------------------------------
void fftw_r2c_2d(double *in_real, fftw_complex *in_fft, int nx, int ny) {
	int i,j;
	int nyh = ny/2+1;
	int index;

	fftw_complex *in_fft_tmp;
	in_fft_tmp = (fftw_complex *)fftw_malloc(nx*nyh*sizeof(fftw_complex));

	fftw_plan plfd;
	plfd = fftw_plan_dft_r2c_2d(nx,ny,in_real,in_fft_tmp,FFTW_ESTIMATE);
	fftw_execute(plfd);
	fftw_destroy_plan(plfd);

	for(i=0;i<nx;i++) for(j=0;j<nyh;j++) {
		index = i*nyh+j;
		in_fft[index][0] = in_fft_tmp[index][0];
		in_fft[index][1] = in_fft_tmp[index][1];
	}

	fftw_free(in_fft_tmp);
}
//--------------------------------------------------------------------
void fftw_c2r_2d(fftw_complex *in_fft, double *in_real, int nx, int ny) {

	int i,j;
	int index;

	double *in_real_tmp = (double *)malloc(nx*ny*sizeof(double));
	fftw_plan plbd;
	plbd = fftw_plan_dft_c2r_2d(nx, ny, in_fft, in_real_tmp, FFTW_ESTIMATE);
	fftw_execute(plbd);

	for(i=0;i<nx;i++) for(j=0;j<ny;j++) {
		index = i*ny+j;
		in_real[index] = in_real_tmp[index];
	}

	fftw_destroy_plan(plbd);
	free(in_real_tmp);
}
//--------------------------------------------------------------------
void convolve_fft(double *in1, double *in2, double *out, int nx, int ny, double dx, double dy) {
	int i,j;
	int nyh = ny/2+1;
	int index;
	double tmpr,tmpi;

	double in1_fftr = 0.0;
	double in1_ffti = 0.0;
	double in2_fftr = 0.0;
	double in2_ffti = 0.0;

	fftw_complex *in1_fft = (fftw_complex *)fftw_malloc(nx*nyh*sizeof(fftw_complex));
	fftw_complex *in2_fft = (fftw_complex *)fftw_malloc(nx*nyh*sizeof(fftw_complex));
	fftw_complex *out_fft = (fftw_complex *)fftw_malloc(nx*nyh*sizeof(fftw_complex));

    /* From theta-space to k-space */
	fftw_r2c_2d(in1, in1_fft, nx, ny); /* kappa */
	fftw_r2c_2d(in2, in2_fft, nx, ny); /* alpha_iso */

    /* convolution in k-space */
	for (i=0;i<nx;i++) for(j=0;j<nyh;j++) {
		index = i*nyh+j;
		in1_fftr = in1_fft[index][0];
		in1_ffti = in1_fft[index][1];

		in2_fftr = in2_fft[index][0];
		in2_ffti = in2_fft[index][1];
		tmpr = in1_fftr*in2_fftr - in1_ffti*in2_ffti;
		tmpi = in1_fftr*in2_ffti + in1_ffti*in2_fftr;
		out_fft[index][0] = tmpr;
		out_fft[index][1] = tmpi;
	}

    /* From k-space to theta-space */
	double *out_tmp = (double *)malloc(nx*ny*sizeof(double));
	fftw_c2r_2d(out_fft, out_tmp, nx, ny);

	for(i=0;i<nx;i++) for(j=0;j<ny;j++) {
		index = i*ny+j;
		out[index] = out_tmp[index]/(nx*ny)*dx*dy;
	}

	fftw_free(in1_fft);
	fftw_free(in2_fft);
	fftw_free(out_fft);

	free(out_tmp);
}
//--------------------------------------------------------------------
