#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>
#include "fft_convolve.h"
//--------------------------------------------------------------------
void zero_padding(double *in, int nx, int ny, double *out) {
    /* Treatment of End Effects caused by discerte convolution */
    /* http://www.aip.de/groups/soe/local/numres/bookfpdf/f13-1.pdf */
	int i,j,index1,index2;
	for ( i = 0; i < nx; i++ ) {
		for ( j = 0; j < ny; j++ ) {
			index1 = i*ny+j;
			index2 = i*2*ny+j;
			out[index2] = in[index1];
		}
	}
}
//--------------------------------------------------------------------
void center_matrix(double *in, int nx, int ny, double *out) {

	int i,j,index1,index2;
	for (i=nx/4;i<3*nx/4; i++ ) {
		for (j=ny/4;j<3*ny/4; j++ ) {
			index1 = i*ny+j;
			index2 = (i-nx/4)*ny/2+(j-ny/4);
			out[index2] = in[index1];
		}
	}
}
//--------------------------------------------------------------------
void corner_matrix(double *in, int nx, int ny, double *out) {

	int i,j,index1,index2;
	for (i=0;i<nx/2; i++ ) {
		for (j=0;j<ny/2; j++ ) {
			index1 = i*ny+j;
			index2 = i*ny/2+j;
			out[index2] = in[index1];
		}
	}
}
//--------------------------------------------------------------------
void kernel_alphas_iso(int Ncc,double *in1,double *in2,double Dcell) {
	int i,j;
	double x,y,r;

	for(i=0;i<Ncc;i++) for(j=0;j<Ncc;j++) {
        /* Spherically Symmetric */
		/* Calculate 1/4 and copy to the other 3/4 */
        if(i <=(Ncc/2)  && j <=(Ncc/2)) {
			x = (double)(i)*Dcell+0.5*Dcell;
			y = (double)(j)*Dcell+0.5*Dcell;
			r = sqrt(x*x+y*y);

			if(r > Dcell*(double)Ncc/2.0) {
				in1[i*Ncc+j] = 0.0;
				in2[i*Ncc+j] = 0.0;
			}
			else {
				in1[i*Ncc+j] = x/(M_PI*r*r);
				in2[i*Ncc+j] = y/(M_PI*r*r);
			}

		}
		else {
			if(i <= Ncc/2 && j > (Ncc/2)) {
				in1[i*Ncc+j]  =  in1[i*Ncc+Ncc-j];
				in2[i*Ncc+j]  = -in2[i*Ncc+Ncc-j];
			}
			if(i > (Ncc/2) && j <= (Ncc/2)) {
				in1[i*Ncc+j]  = -in1[(Ncc-i)*Ncc+j];
				in2[i*Ncc+j]  =  in2[(Ncc-i)*Ncc+j];
			}

			if(i > (Ncc/2) && j > (Ncc/2)) {
				in1[i*Ncc+j]  = -in1[(Ncc-i)*Ncc+Ncc-j];
				in2[i*Ncc+j]  = -in2[(Ncc-i)*Ncc+Ncc-j];
			}
        }
	}
}

//--------------------------------------------------------------------
void kappa0_to_alphas(double * kappa0, int Nc, double bsz, double * alpha1, double * alpha2) {
    /* Wertz, Olivier, and Jean Surdej; MNRAS 437.2 (2013): 1051-1055. */
	/* M. Bartelmann 2003 https://arxiv.org/pdf/astro-ph/0304162.pdf */
    int Nc2 = Nc*2;
	double dsx = bsz/(double)Nc;

	double *kappa = (double *)calloc(Nc2*Nc2,sizeof(double));
	zero_padding(kappa0, Nc, Nc, kappa);

    /* Bartelmann, equ. 22 isochrone Kernel */
	double *alpha1_iso = (double *)calloc(Nc2*Nc2,sizeof(double));
	double *alpha2_iso = (double *)calloc(Nc2*Nc2,sizeof(double));
	kernel_alphas_iso(Nc2, alpha1_iso, alpha2_iso, dsx);

    /* Wertz&Jean, equ. 3, convolution */
	double *alpha1_tmp = (double *)calloc(Nc2*Nc2,sizeof(double));
	double *alpha2_tmp = (double *)calloc(Nc2*Nc2,sizeof(double));
	convolve_fft(kappa, alpha1_iso, alpha1_tmp, Nc2, Nc2, dsx, dsx);
	convolve_fft(kappa, alpha2_iso, alpha2_tmp, Nc2, Nc2, dsx, dsx);

	free(kappa);
	free(alpha1_iso);
	free(alpha2_iso);

	corner_matrix(alpha1_tmp, Nc2, Nc2, alpha1);
	corner_matrix(alpha2_tmp, Nc2, Nc2, alpha2);

	free(alpha1_tmp);
	free(alpha2_tmp);
}
//--------------------------------------------------------------------
void kernel_phi_iso(int Ncc,double *in,double Dcell) {
	int i,j;
	double x,y,r;

	for(i=0;i<Ncc;i++) for(j=0;j<Ncc;j++) {
		if(i <=(Ncc/2)  && j <=(Ncc/2)) {
			x = (double)(i)*Dcell+0.5*Dcell;
			y = (double)(j)*Dcell+0.5*Dcell;
			r = sqrt(x*x+y*y);

			if(r > Dcell*(double)Ncc/2.0) {
				in[i*Ncc+j] = 0.0;
			}
			else {
				in[i*Ncc+j] = 1.0/M_PI*log(r);
			}

		}
		else {
			if(i <= Ncc/2 && j > (Ncc/2)) {
				in[i*Ncc+j] = in[i*Ncc+Ncc-j];
			}
			if(i > (Ncc/2) && j <= (Ncc/2)) {
				in[i*Ncc+j] = in[(Ncc-i)*Ncc+j];
			}

			if(i > (Ncc/2) && j > (Ncc/2)) {
				in[i*Ncc+j] = in[(Ncc-i)*Ncc+Ncc-j];
			}
		}
	}
}

//--------------------------------------------------------------------
void kappa0_to_phi(double * kappa0, int Nc, double bsz, double * phi) {
	/* M. Bartelmann 2003 https://arxiv.org/pdf/astro-ph/0304162.pdf */
	/* equ. 24 */
    int Nc2 = Nc*2;
	double dsx = bsz/(double)Nc;

	double *kappa = (double *)calloc(Nc2*Nc2,sizeof(double));
	zero_padding(kappa0, Nc, Nc, kappa);

	double *phi_iso = (double *)calloc(Nc2*Nc2,sizeof(double));
	kernel_phi_iso(Nc2, phi_iso, dsx);

	double *phi_tmp = (double *)calloc(Nc2*Nc2,sizeof(double));

    convolve_fft(kappa, phi_iso, phi_tmp, Nc2, Nc2, dsx, dsx);

	free(kappa);
	free(phi_iso);

	corner_matrix(phi_tmp, Nc2, Nc2, phi);

	free(phi_tmp);
}
