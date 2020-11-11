void zero_padding(double *in, int nx, int ny, double *out);
void center_matrix(double *in, int nx, int ny, double *out);
void corner_matrix(double *in, int nx, int ny, double *out);
void kernel_alphas_iso(int Ncc,double *in1,double *in2,double Dcell);
void kappa0_to_alphas(double * kappa0, int Nc, double bsz, double * alpha1, double * alpha2);
void kernel_phi_iso(int Ncc,double *in,double Dcell);
void kappa0_to_phi(double * kappa0, int Nc, double bsz, double * phi);
