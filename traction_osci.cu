/*The basic model of Danying's 2012 PNAS paper, without explicitly track the adhesion sites. Instead, using a rhoM dependent manner to handle the friction and then the traction force*/

//Feb-22-2018: Looks like for the diffusion term, spectral method does not work well. This is the only term that used finite differetiation in this code
// may-23-2020: set the rhoa at front half and rhom at back half (with sin a tune the percentage to put the rhoa)
// June-21-2020: set the rhoa to be periodic function

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include "cublas_v2.h"
#include "curand.h"
#include "curand_kernel.h"
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <cufft.h>

#include <stdio.h>
#include<stdlib.h>
#include"math.h"
#include<algorithm>
#include<random>
#include<time.h>

#define MIN(a,b) ((a<b) ? a:b)
#define MAX(a,b) ((a>b) ? a:b)


# define M_PI           3.14159265358979323846  /* pi */
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
using namespace std;

//simulation grid set up
const int Nx = 256;
const int Ny = 256;
dim3 blocks(Nx / BLOCK_SIZE_X, Ny / BLOCK_SIZE_Y);
dim3 threadsperblock(BLOCK_SIZE_X, BLOCK_SIZE_Y);


//grid box
const float Lx = 25.0f, Ly = 25.0f;
const float dx = 2.0f * Lx / (float)Nx, dy = 2.0f * Ly / (float)Ny;
//time steps
float dt = 2e-3;
//relaxation Gamma
extern __constant__ float d_Gamma = 0.4f;
//phase-field width
extern __constant__ float d_epsilon = 2.0f;
const float h_epsilon = 2.0f;

//runing time and recording time
float max_time = 300.0f;
const float record_itvl = 5.0f;
//intial condition
float rhoAinitial = 1.0f;
float rhoMinitial = 0.3f;

//error control
float poisson_err_limit = 0.01f;
int poisson_max_steps = 100;
float exceed_val = 2.0f; //the exceed value in constructing poisson eq
extern __constant__ float d_exceed_val  = 2.0f;
//const float lamda = 1e-4;

//some global parameters
float r0_cell = 8.0f; //radius of cell
extern __constant__ float d_tension = 20.0f; //tension per unit area
extern __constant__ float d_bending = 0.0f; //bending energy
extern __constant__ float d_Mv = 200.0f; //volume conservation strength. Large values means strong contraint, small value means soft and elastic constratint
extern __constant__ float d_areaMin = 100.0f; //min-area size 
extern __constant__ float d_areaMax = 800.0f; // max-area size 
float nu_vis = 1000.0f;

//friction and traction parameters
float xi_fric = 0.5f;
extern __constant__ float d_xiM = 4.0f; //additional friction due to myosin
extern __constant__ float d_expRhoM = 0.1f;
extern __constant__ float d_thresA = 0.5f;

float h_diffRhoA = 0.8f;
float h_diffRhoM0 = 2.0f;
float h_KdRhoM = 0.5f;
float h_RhoAtot = 350.0f;
float h_kbRhoA = 10.0f;
float h_kaRhoA = 0.01f;
float h_kcRhoA = 10.0f;
float h_K2aRhoA = 1.0f;

float h_etaRhoA = 1000.0f;
float h_etaRhoM = 80.0f;

float A_peri = 100.0f; //the osci-period of rhoa
extern __constant__ float d_phase_m = 0.0f; //the phase lag of myosin compared to actin
float h_band_width = 2.0f; //the band width of the circular rhoa and rhom distribution

//cufft and cublas handles
cufftHandle plan_R2C;
cufftHandle plan_C2R;
cublasHandle_t blashandle;
cublasHandle_t h_blashandle;

//utility functions
void Initialize(float* phi, float *rhog, float *rhor, float* ux, float* uy, float* x, float* y);
void writetofile(const char *name, float *u, int Nx, int Ny);
void fftcoeffs(float *d_d1x, float *d_d1y, float *d_d2x, float *d_d2y);
__global__ void absarray(float *absay, float *ax, float *ay, int Nx, int Ny);
__global__ void add3matrix(float *output, float *input1, float *input2, float *input3);
__global__ void add2matrix(float *output, float *input1, float *input2);
__global__ void minus_matrix(float *output, float *minuend, float *substractor, int Nx, int Ny);
__global__ void get_error(float *max_error, float *ux_new, float *ux_old, float *uy_new, float *uy_old, int *idx_eux, int *idx_ux, int *idx_euy, int *idx_uy, int Nx, int Ny);
__global__ void matrix_product(float *output, float *input1, float *input2, float alpha, int Nx, int Ny);
__global__ void xdir_center(float *xc_sin, float *xc_cos, float *phi, float *phi_area, float *x, float Lx, int m, int n);

//derivative functions
void par_deriv1_fft(float *output, float *input, int dim, float *deriv_vec, cufftComplex *R2C_buffer, cufftComplex *C2R_buffer, int Nx, int Ny);
void grad_fft(float *output_x, float *output_y, float *input, float *deriv_vec_x, float *deriv_vec_y, cufftComplex *R2C_buffer, cufftComplex *C2R_buffer, int Nx, int Ny);
void lap_fft(float *output, float *input, float *deriv_vec_x, float *deriv_vec_y, cufftComplex *R2C_buffer, cufftComplex *C2R_buffer, int Nx, int Ny);
void div_fft(float *output, float *input_x, float *input_y, float *deriv_vec_x, float *deriv_vec_y, cufftComplex *R2C_buffer, cufftComplex *C2R_buffer, int Nx, int Ny);
__global__ void fftR2C_deriv1(cufftComplex *input, cufftComplex *output, int dim, float *coeff, int Nx, int Ny);
__global__ void fftR2C_lap(cufftComplex *input, cufftComplex *output, float *coeff_x, float *coeff_y, int Nx, int Ny);
__global__ void poissonR2C(cufftComplex *input, cufftComplex *output, float *coeff_x, float *coeff_y, float coeff0, float coeff2, int Nx, int Ny);
void poisson_sol(float *sol, float *rhs, float coeff0, float coeff2, float *coeff_x, float *coeff_y, cufftComplex *R2C_buffer, cufftComplex *C2R_buffer, int Nx, int Ny);
void div_pst_fft(float *output, float *c_pst, float *rho, float coeff_diff, float *deriv_vec_x2, float *deriv_vec_y2, float *buffer_x, float *buffer_y, cufftComplex *R2C_buffer, cufftComplex *C2R_buffer, int Nx, int Ny);
__global__ void div_pst_fd(float *output, float *c_pst, float *rho, float coeff_diff, float dx, float dy, int Nx, int Ny);

//physical functions
__global__ void curv(float *curv, float *phi_gradx, float *phi_grady, float *absdphi, int Nx, int Ny, float dx, float dy);
__global__ void curv(float *curv, float *phi, float *absdphi, int m, int n, float dx, float dy);
__global__ void phievolve(float *phi_new, float *phi_old, float *grad_x, float *grad_y, float *absdphi, float *lap_phi, float *curv, float *ux, float *uy, int Nx, int Ny, float dt);
__global__ void potential_force(float *output, float *phi, float *lap_phi, float *bendcore, float *phiarea, float A0, float dx, float dy, int Nx, int Ny);
__global__ void rhs_poisson(float *rhs_x, float *rhs_y, float *ptl_force, float *phi, float *dphix, float *dphiy, float *vis_x, float *vis_y, float *act_x, float *act_y, float *rhoA, float *rhoM, float *ux, float *uy, int Nx, int Ny);
__global__ void div_vel_pre(float *div_x, float *div_y, int dim, float *phi, float nu, float *duxdx, float *duxdy, float *duydx, float *duydy, int Nx, int Ny);
__global__ void bend_core(float *core, float *lap_phi, float *phi, int Nx, int Ny);
__global__ void div_advect_pre(float *div_x, float *div_y, float *dnsty, float *rho, float *ux, float *uy, int Nx, int Ny);
__global__ void diffRhoM(float *output, float *phi, float *rhoa, float Dm0, float Kd, int Nx, int Ny);
__global__ void reactionRhoA(float *output, float *phi, float *rhoA, float *rhoAarea, float *phiarea, float rhoAtot, float kb, float K2a, float ka, float kc, float dx, float dy, int Nx, int Ny);
__global__ void RD_evolve(float *rho_new, float *dnsty_new, float *dnsty_old, float *rho_old, float *advect, float *diffuse, float *react, float dt, float lamda, int Nx, int Ny);
__global__ void RD_evolve(float *rho_new, float *dnsty_new, float *dnsty_old, float *rho_old, float *advect, float *diffuse, float dt, float lamda, int Nx, int Ny);
__global__ void normRhoM(float *rho_new, float *rho_area_new, float *rho_are_old, int Nx, int Ny);
__global__ void rhoa_right(float *rhoa, float *phi, float x_sin, float x_cos, float *x, float Lx, int m, int n);
__global__ void rhoa_osci(float *rhoa, float *rhom, float *phi, float x_sin, float x_cos, float t, float rhoa_period, float r_chi, float *x, float *y, float Lx, int m, int n);
__global__ void activeForce(float *div_x, float *div_y, int dir, float *phi, float etaRhoA, float etaRhoM, float *rhoA, float *rhoM, float *dphix, float *dphiy, float *phiarea, float dx, float dy, int Nx, int Ny);

int main()
{
	//initialize host
	FILE *ft = fopen("center_traj.txt", "w+");
	float *h_x = (float *)malloc(Nx*sizeof(float));
	float *h_y = (float *)malloc(Ny*sizeof(float));
	float *h_phi = (float *)malloc(Nx*Ny*sizeof(float));
	float *h_ux = (float *)malloc(Nx*Ny*sizeof(float));
	float *h_uy = (float *)malloc(Nx*Ny*sizeof(float));
	float *h_RhoA = (float *)malloc(Nx*Ny*sizeof(float));
	float *h_RhoM = (float *)malloc(Nx*Ny*sizeof(float));
	float h_xsin = 0.0f;
	float h_xcos = 0.0f;
	int file_record = 1;

	//initialize device
	float *d_x; cudaMalloc((void **)&d_x, Nx*sizeof(float));
	float *d_y; cudaMalloc((void **)&d_y, Ny*sizeof(float));
	float *d_phi_old; cudaMalloc((void **)&d_phi_old, Nx*Ny*sizeof(float));
	float *d_phi_new; cudaMalloc((void **)&d_phi_new, Nx*Ny*sizeof(float));
	float *d_ux_old; cudaMalloc((void **)&d_ux_old, Nx*Ny*sizeof(float));
	float *d_uy_old; cudaMalloc((void **)&d_uy_old, Nx*Ny*sizeof(float));
	float *d_ux_new; cudaMalloc((void **)&d_ux_new, Nx*Ny*sizeof(float));
	float *d_uy_new; cudaMalloc((void **)&d_uy_new, Nx*Ny*sizeof(float));
	float *d_RhoAold; cudaMalloc((void **)&d_RhoAold, Nx*Ny*sizeof(float));
	float *d_RhoAnew; cudaMalloc((void **)&d_RhoAnew, Nx*Ny*sizeof(float));
	float *d_RhoMold; cudaMalloc((void **)&d_RhoMold, Nx*Ny*sizeof(float));
	float *d_RhoMnew; cudaMalloc((void **)&d_RhoMnew, Nx*Ny*sizeof(float));

	//derivatives of velocities
	float *d_duxdx; cudaMalloc((void **)&d_duxdx, Nx*Ny*sizeof(float));
	float *d_duxdy; cudaMalloc((void **)&d_duxdy, Nx*Ny*sizeof(float));
    	
	float *d_duydx; cudaMalloc((void **)&d_duydx, Nx*Ny*sizeof(float));
	float *d_duydy; cudaMalloc((void **)&d_duydy, Nx*Ny*sizeof(float));
	//divergence of the viscosity tensor
	float *d_div_x; cudaMalloc((void **)&d_div_x, Nx*Ny*sizeof(float));
	float *d_div_y; cudaMalloc((void **)&d_div_y, Nx*Ny*sizeof(float));
	//rhs of the poisson equation
	float *d_rhs_ux; cudaMalloc((void **)&d_rhs_ux, Nx*Ny*sizeof(float));
	float *d_rhs_uy; cudaMalloc((void **)&d_rhs_uy, Nx*Ny*sizeof(float));
	
	//substrate information

	
	//forces
	float *d_ActForceX; cudaMalloc((void **)&d_ActForceX, Nx*Ny*sizeof(float));
	float *d_ActForceY; cudaMalloc((void **)&d_ActForceY, Nx*Ny*sizeof(float));
	//forces in forms of potential * grad(phi)
	float *d_ptl_force; cudaMalloc((void **)&d_ptl_force, Nx*Ny*sizeof(float));
	
	//define gradient, laplacian, curvature, Gprime, absgrad
	float *d_dphix; cudaMalloc((void **)&d_dphix, Nx*Ny*sizeof(float));
	float *d_dphiy; cudaMalloc((void **)&d_dphiy, Nx*Ny*sizeof(float));
	float *d_phi_absgrad; cudaMalloc((void **)&d_phi_absgrad, Nx*Ny*sizeof(float));
	float *d_phi_lap; cudaMalloc((void **)&d_phi_lap, Nx*Ny*sizeof(float));

	float *d_phi_curv; cudaMalloc((void **)&d_phi_curv, Nx*Ny*sizeof(float));
	float *d_ftd1_x; cudaMalloc((void **)&d_ftd1_x, Nx*sizeof(float));
	float *d_ftd1_y; cudaMalloc((void **)&d_ftd1_y, Ny*sizeof(float));
	float *d_ftd2_x; cudaMalloc((void **)&d_ftd2_x, Nx*sizeof(float));
	float *d_ftd2_y; cudaMalloc((void **)&d_ftd2_y, Ny*sizeof(float));

	float *d_xc_sin; cudaMalloc((void **)&d_xc_sin, Nx*Ny*sizeof(float)); 
	float *d_xc_cos; cudaMalloc((void **)&d_xc_cos, Nx*Ny*sizeof(float));
	
	//temporary buffers
	float *d_temp_buffer; cudaMalloc((void **)&d_temp_buffer, Nx*Ny*sizeof(float)); //cublas buffer
	cufftComplex *d_fftR2C_buffer; cudaMalloc((void **)&d_fftR2C_buffer, (Nx / 2 + 1)*Ny*sizeof(cufftComplex)); //cufft R2C buffer as fft result
	cufftComplex *d_fftC2R_buffer; cudaMalloc((void **)&d_fftC2R_buffer, (Nx / 2 + 1)*Ny*sizeof(cufftComplex)); //as C2R buffer
	float *d_buffer_x; cudaMalloc((void **)&d_buffer_x, Nx*Ny*sizeof(float));
	float *d_buffer_y; cudaMalloc((void **)&d_buffer_y, Nx*Ny*sizeof(float));
	float *d_advect_buffer; cudaMalloc((void **)&d_advect_buffer, Nx*Ny*sizeof(float));
	float *d_diffuse_buffer; cudaMalloc((void **)&d_diffuse_buffer, Nx*Ny*sizeof(float));
	float *d_reaction_buffer; cudaMalloc((void **)&d_reaction_buffer, Nx*Ny*sizeof(float));
	float *d_DiffRhoM; cudaMalloc((void **)&d_DiffRhoM, Nx*Ny*sizeof(float)); //the rhoa-dependent rhom diffusion constant
	float *d_rhoAarea; cudaMalloc((void **)&d_rhoAarea, sizeof(float)); //total active rhoA
	float *d_phiArea; cudaMalloc((void **)&d_phiArea, sizeof(float)); //total area of phi
	float *d_rhoMoldArea; cudaMalloc((void **)&d_rhoMoldArea, sizeof(float)); //total of rhoM old
	float *d_rhoMnewArea; cudaMalloc((void **)&d_rhoMnewArea, sizeof(float)); //total of rhoM new
	float *d_bendcore; cudaMalloc((void **)&d_bendcore,Nx*Ny*sizeof(float)); //the bending core
	
	//fft coefficients
	fftcoeffs(d_ftd1_x,d_ftd1_y, d_ftd2_x,d_ftd2_y);
	
	//handles
	cufftPlan2d(&plan_R2C, Nx, Ny, CUFFT_R2C);
	cufftPlan2d(&plan_C2R, Nx, Ny, CUFFT_C2R);
	cublasCreate(&blashandle);
	cublasSetPointerMode(blashandle, CUBLAS_POINTER_MODE_DEVICE); //make the cublas return value to device
	cublasCreate(&h_blashandle);

	//timer start
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	//initialize the global parameters
	//r0_cell = 6.0f; 
    
	Initialize(h_phi, h_RhoA, h_RhoM, h_ux, h_uy, h_x, h_y);
	//copy memory
	cudaMemcpy(d_x, h_x, Nx*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, Ny*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_phi_old, h_phi, Nx*Ny*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_phi_new, h_phi, Nx*Ny*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_RhoAold, h_RhoA, Nx*Ny*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_RhoMold, h_RhoM, Nx*Ny*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_RhoAnew, h_RhoA, Nx*Ny*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_RhoMnew, h_RhoM, Nx*Ny*sizeof(float), cudaMemcpyHostToDevice);	
	cudaMemcpy(d_ux_old, h_ux, Nx*Ny*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_uy_old, h_uy, Nx*Ny*sizeof(float), cudaMemcpyHostToDevice);
	//area size volume
	float A0, A_ist, h_r_chi;
	cublasSasum(h_blashandle, Nx*Ny, d_phi_old, 1, &A0);
	cublasSasum(h_blashandle, Nx*Ny, d_phi_old, 1, &A_ist);
	//error parameters
	float max_error;
	int iter_steps;


	int *idx_maxerr_ux; cudaMalloc((void **)& idx_maxerr_ux, sizeof(int));
	int *idx_max_ux; cudaMalloc((void **)&idx_max_ux, sizeof(int));
	int *idx_maxerr_uy; cudaMalloc((void **)& idx_maxerr_uy, sizeof(int));
	int *idx_max_uy; cudaMalloc((void **)&idx_max_uy, sizeof(int));
	float *d_max_error; cudaMalloc((void **)&d_max_error, sizeof(float));

	//mass center
	grad_fft(d_dphix, d_dphiy, d_phi_old, d_ftd1_x, d_ftd1_y, d_fftR2C_buffer, d_fftC2R_buffer, Nx, Ny);
	absarray <<<blocks, threadsperblock >>>(d_phi_absgrad, d_dphix, d_dphiy, Nx, Ny);
	lap_fft(d_phi_lap, d_phi_old, d_ftd2_x, d_ftd2_y, d_fftR2C_buffer, d_fftC2R_buffer, Nx, Ny);


	for (int steps = 0; steps<(int)(max_time / dt); steps++){

		//curv <<<blocks, threadsperblock >>>(d_phi_curv, d_dphix, d_dphiy, d_phi_absgrad, Nx, Ny, dx,dy);
		curv <<<blocks, threadsperblock >>>(d_phi_curv, d_phi_old, d_phi_absgrad, Nx, Ny, dx,dy);
		//solve phi
		phievolve <<<blocks, threadsperblock >>>(d_phi_new, d_phi_old, d_dphix, d_dphiy, d_phi_absgrad, d_phi_lap, d_phi_curv, d_ux_old, d_uy_old, Nx, Ny, dt);

		//update derivatives
		grad_fft(d_dphix, d_dphiy, d_phi_new, d_ftd1_x, d_ftd1_y, d_fftR2C_buffer, d_fftC2R_buffer, Nx, Ny);
		absarray <<<blocks, threadsperblock >>>(d_phi_absgrad, d_dphix, d_dphiy, Nx, Ny);
		lap_fft(d_phi_lap, d_phi_new, d_ftd2_x, d_ftd2_y, d_fftR2C_buffer, d_fftC2R_buffer, Nx, Ny);

		
		//update rhoA and rhoM
		//rhoA advection
		//div_advect_pre<<<blocks, threadsperblock >>>(d_div_x, d_div_y, d_phi_old, d_RhoAold, d_ux_old, d_uy_old, Nx, Ny);
		//div_fft(d_advect_buffer, d_div_x, d_div_y, d_ftd1_x, d_ftd1_y, d_fftR2C_buffer, d_fftC2R_buffer, Nx, Ny);
		//rhoA diffusion        	
		//div_pst_fd<<<blocks, threadsperblock >>>(d_diffuse_buffer, d_phi_old, d_RhoAold, h_diffRhoA, dx, dy, Nx, Ny);
		//rhoA reaction
		cublasSasum(blashandle, Nx * Ny, d_phi_old, 1, d_phiArea);
		cublasSasum(h_blashandle, Nx * Ny, d_phi_old, 1, &A_ist);
		A_ist = A_ist * dx * dy;
		h_r_chi = sqrt(A_ist / M_PI) - h_band_width;
		//matrix_product<<<blocks, threadsperblock >>>(d_temp_buffer, d_RhoAold, d_phi_old, 1.0f, Nx, Ny);
		//cublasSasum(blashandle, Nx * Ny, d_temp_buffer, 1, d_rhoAarea);
		//reactionRhoA<<<blocks, threadsperblock >>>(d_reaction_buffer, d_phi_old, d_RhoAold, d_rhoAarea, d_phiArea, h_RhoAtot, h_kbRhoA, h_K2aRhoA, h_kaRhoA, h_kcRhoA, dx, dy, Nx, Ny);
		//time evolve
		//RD_evolve<<<blocks, threadsperblock >>>(d_RhoAnew, d_phi_new, d_phi_old, d_RhoAold, d_advect_buffer, d_diffuse_buffer, d_reaction_buffer, dt, lamda, Nx, Ny);
		//rhoa_right<<<blocks, threadsperblock>>>(d_RhoAold, d_phi_old, h_xsin, h_xcos, d_x, Lx, Nx, Ny);
		//rhoa_right<<<blocks, threadsperblock>>>(d_RhoAnew, d_phi_new, h_xsin, h_xcos, d_x, Lx, Nx, Ny);

		rhoa_osci<<<blocks, threadsperblock>>>(d_RhoAold, d_RhoMold, d_phi_old, h_xsin, h_xcos, steps * dt, A_peri, h_r_chi, d_x, d_y, Lx, Nx, Ny);
		rhoa_osci<<<blocks, threadsperblock>>>(d_RhoAnew, d_RhoMnew, d_phi_new, h_xsin, h_xcos, steps * dt, A_peri, h_r_chi, d_x, d_y, Lx, Nx, Ny);

		
		//rhoM advection
		//div_advect_pre<<<blocks, threadsperblock >>>(d_div_x, d_div_y, d_phi_old, d_RhoMold, d_ux_old, d_uy_old, Nx, Ny);
		//div_fft(d_advect_buffer, d_div_x, d_div_y, d_ftd1_x, d_ftd1_y, d_fftR2C_buffer, d_fftC2R_buffer, Nx, Ny);
		//rhoM diffusion
		//diffRhoM<<<blocks, threadsperblock >>>(d_DiffRhoM, d_phi_old, d_RhoAold, h_diffRhoM0, h_KdRhoM, Nx, Ny);
		//div_pst_fd<<<blocks, threadsperblock >>>(d_diffuse_buffer, d_DiffRhoM, d_RhoMold, 1.0f, dx, dy, Nx, Ny);
		//no rhoM reaction
		//RD_evolve<<<blocks, threadsperblock >>>(d_RhoMnew, d_phi_new, d_phi_old, d_RhoMold, d_advect_buffer, d_diffuse_buffer, dt, lamda, Nx, Ny);
		//normRhoM rhoM
		//matrix_product<<<blocks, threadsperblock>>>(d_temp_buffer, d_RhoMold, d_phi_old, dx * dy, Nx, Ny);
		//cublasSasum(blashandle, Nx * Ny, d_temp_buffer, 1, d_rhoMoldArea);
		//matrix_product<<<blocks, threadsperblock>>>(d_temp_buffer, d_RhoMnew, d_phi_new, dx * dy, Nx, Ny);
		//cublasSasum(blashandle, Nx * Ny, d_temp_buffer, 1, d_rhoMnewArea);
		//normRhoM<<<blocks, threadsperblock>>>(d_RhoMnew, d_rhoMnewArea, d_rhoMoldArea, Nx, Ny);

		//only rhoA at the right
		

		//update forces
		bend_core<<<blocks, threadsperblock>>>(d_bendcore, d_phi_lap, d_phi_new, Nx, Ny);
		lap_fft(d_bendcore, d_bendcore, d_ftd2_x, d_ftd2_y, d_fftR2C_buffer, d_fftC2R_buffer, Nx, Ny);
		potential_force<<<blocks, threadsperblock >>>(d_ptl_force, d_phi_new, d_phi_lap, d_bendcore, d_phiArea, A0, dx, dy, Nx, Ny);
		
		//active force
		activeForce<<<blocks, threadsperblock>>>(d_div_x, d_div_y, 1, d_phi_new, h_etaRhoA, h_etaRhoM, d_RhoAnew, d_RhoMnew, d_dphix, d_dphiy, d_phiArea, dx, dy, Nx, Ny);
		div_fft(d_ActForceX, d_div_x, d_div_y, d_ftd1_x, d_ftd1_y, d_fftR2C_buffer, d_fftC2R_buffer, Nx, Ny);
		activeForce<<<blocks, threadsperblock>>>(d_div_x, d_div_y, 2, d_phi_new, h_etaRhoA, h_etaRhoM, d_RhoAnew, d_RhoMnew, d_dphix, d_dphiy, d_phiArea, dx, dy, Nx, Ny);
		div_fft(d_ActForceY, d_div_x, d_div_y, d_ftd1_x, d_ftd1_y, d_fftR2C_buffer, d_fftC2R_buffer, Nx, Ny);
		
		max_error = 10.0f;
		iter_steps = 0;
		while(max_error > poisson_err_limit && iter_steps < poisson_max_steps){
		    //grad(ux)
		    grad_fft(d_duxdx, d_duxdy, d_ux_old, d_ftd1_x, d_ftd1_y, d_fftR2C_buffer, d_fftC2R_buffer, Nx, Ny);
		    //grad(uy)
		    grad_fft(d_duydx, d_duydy, d_uy_old, d_ftd1_x, d_ftd1_y, d_fftR2C_buffer, d_fftC2R_buffer, Nx, Ny);
		    //grad(uz)
				//vel_div_x
		    div_vel_pre<<<blocks, threadsperblock >>>(d_div_x, d_div_y, 1, d_phi_new,  nu_vis, d_duxdx, d_duxdy, d_duydx, d_duydy, Nx, Ny);
		    div_fft(d_buffer_x, d_div_x, d_div_y, d_ftd1_x, d_ftd1_y, d_fftR2C_buffer, d_fftC2R_buffer, Nx, Ny);
		    //vel_div_y
		    div_vel_pre<<<blocks, threadsperblock >>>(d_div_x, d_div_y, 2, d_phi_new,  nu_vis, d_duxdx, d_duxdy, d_duydx, d_duydy, Nx, Ny);
		    div_fft(d_buffer_y, d_div_x, d_div_y, d_ftd1_x, d_ftd1_y, d_fftR2C_buffer, d_fftC2R_buffer, Nx, Ny);
		   
		    rhs_poisson<<<blocks, threadsperblock >>>(d_rhs_ux, d_rhs_uy, d_ptl_force, d_phi_new, d_dphix, d_dphiy, d_buffer_x, d_buffer_y, d_ActForceX, d_ActForceY, d_RhoAnew, d_RhoMnew, d_ux_old, d_uy_old, Nx, Ny);

		    poisson_sol(d_ux_new, d_rhs_ux, xi_fric, nu_vis*exceed_val, d_ftd2_x, d_ftd2_y, d_fftR2C_buffer, d_fftC2R_buffer, Nx, Ny);
		    poisson_sol(d_uy_new, d_rhs_uy, xi_fric, nu_vis*exceed_val, d_ftd2_x, d_ftd2_y, d_fftR2C_buffer, d_fftC2R_buffer, Nx, Ny);

		    //modify ux uy uz in regions of phi < 0.0001 not done
		    
		    //get error
		    minus_matrix<<<blocks, threadsperblock >>>(d_buffer_x, d_ux_new, d_ux_old, Nx, Ny);
			minus_matrix<<<blocks, threadsperblock >>>(d_buffer_y, d_uy_new, d_uy_old, Nx, Ny);
			cublasIsamax(blashandle, Nx*Ny, d_buffer_x, 1, idx_maxerr_ux);
			cublasIsamax(blashandle, Nx*Ny, d_ux_new, 1, idx_max_ux);
			cublasIsamax(blashandle, Nx*Ny, d_buffer_y, 1, idx_maxerr_uy);
			cublasIsamax(blashandle, Nx*Ny, d_uy_new, 1, idx_max_uy);
			get_error<<<1, 1>>>(d_max_error, d_ux_new, d_ux_old, d_uy_new, d_uy_old, idx_maxerr_ux, idx_max_ux, idx_maxerr_uy, idx_max_uy, Nx, Ny);
			
			cudaMemcpy(&max_error, d_max_error, sizeof(float), cudaMemcpyDeviceToHost);

		    iter_steps++;

		    cublasScopy(blashandle, Nx*Ny, d_ux_new, 1, d_ux_old, 1);
		    cublasScopy(blashandle, Nx*Ny, d_uy_new, 1, d_uy_old, 1);

		}
		if(max_error > poisson_err_limit && steps > 500){
		    printf("step %f wrong err %f\n", steps * dt, max_error);
		    break;
		}

		//record center
		if(steps % 500 == 0){
			xdir_center<<<blocks, threadsperblock>>>(d_xc_sin, d_xc_cos, d_phi_new, d_phiArea, d_x, Lx, Nx, Ny);
			thrust::device_ptr<float> d_sin = thrust::device_pointer_cast(d_xc_sin);
			h_xsin = thrust::reduce(d_sin, d_sin + Nx*Ny);
			thrust::device_ptr<float> d_cos = thrust::device_pointer_cast(d_xc_cos);
			h_xcos = thrust::reduce(d_cos, d_cos + Nx*Ny);
			fprintf(ft, "%f %f\n", steps*dt, atan2(h_xsin, h_xcos)/M_PI*Lx);
		}

		if(steps % (int)(record_itvl/dt)==0 && steps > 0){
			char phi_name[50];sprintf(phi_name,"phi_profile_%d.txt", file_record);
			cudaMemcpy(h_phi, d_phi_old, Nx*Ny*sizeof(float), cudaMemcpyDeviceToHost);
			writetofile(phi_name, h_phi, Nx, Ny);

			char rhoA_name[50];sprintf(rhoA_name,"rhoA_profile_%d.txt",file_record);
			cudaMemcpy(h_RhoA, d_RhoAold, Nx*Ny*sizeof(float), cudaMemcpyDeviceToHost);
			writetofile(rhoA_name, h_RhoA, Nx, Ny);

			char rhoM_name[50];sprintf(rhoM_name,"rhoM_profile_%d.txt",file_record);
			cudaMemcpy(h_RhoM, d_RhoMold, Nx*Ny*sizeof(float), cudaMemcpyDeviceToHost);
			writetofile(rhoM_name, h_RhoM, Nx, Ny);

			char ux_name[50];sprintf(ux_name,"ux_profile_%d.txt",file_record);
			cudaMemcpy(h_ux, d_ux_old, Nx*Ny*sizeof(float), cudaMemcpyDeviceToHost);
			writetofile(ux_name, h_ux, Nx, Ny);

			char uy_name[50];sprintf(uy_name,"uy_profile_%d.txt",file_record);
			cudaMemcpy(h_uy, d_uy_old, Nx*Ny*sizeof(float), cudaMemcpyDeviceToHost);
			writetofile(uy_name, h_uy, Nx, Ny);
			
			file_record++;

		}

		//swap old and new
		cublasScopy(blashandle, Nx*Ny, d_phi_new, 1, d_phi_old, 1);
		cublasScopy(blashandle, Nx*Ny, d_RhoAnew, 1, d_RhoAold, 1);
		cublasScopy(blashandle, Nx*Ny, d_RhoMnew, 1, d_RhoMold, 1);

	}
	
	//final record of the shape
	char phi_name[50]; sprintf(phi_name,"phi_profile.txt", file_record);
	cudaMemcpy(h_phi, d_phi_old, Nx*Ny*sizeof(float), cudaMemcpyDeviceToHost);
	writetofile(phi_name, h_phi, Nx, Ny);

	char rhoA_name[50]; sprintf(rhoA_name,"rhoA_profile.txt",file_record);
	cudaMemcpy(h_RhoA, d_RhoAold, Nx*Ny*sizeof(float), cudaMemcpyDeviceToHost);
	writetofile(rhoA_name, h_RhoA, Nx, Ny);

	char rhoM_name[50]; sprintf(rhoM_name,"rhoM_profile.txt",file_record);
	cudaMemcpy(h_RhoM, d_RhoMold, Nx*Ny*sizeof(float), cudaMemcpyDeviceToHost);
	writetofile(rhoM_name, h_RhoM, Nx, Ny);

	char ux_name[50];sprintf(ux_name,"ux_profile.txt",file_record);
	cudaMemcpy(h_ux, d_ux_old, Nx*Ny*sizeof(float), cudaMemcpyDeviceToHost);
	writetofile(ux_name, h_ux, Nx, Ny);

	char uy_name[50];sprintf(uy_name,"uy_profile.txt",file_record);
	cudaMemcpy(h_uy, d_uy_old, Nx*Ny*sizeof(float), cudaMemcpyDeviceToHost);
	writetofile(uy_name, h_uy, Nx, Ny);
	
	//destroy plan and handles
	cufftDestroy(plan_R2C);
	cufftDestroy(plan_C2R);
	cublasDestroy(blashandle);
	cublasDestroy(h_blashandle);
	fclose(ft);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsetime;
	cudaEventElapsedTime(&elapsetime, start, stop);

	std::printf("time need: %f s \n", elapsetime / 1000);

	return 0;
}

void Initialize(float* phi, float *rhoA, float *rhoM, float* ux, float* uy, float* x, float* y){
	for (int i = 0; i < Nx; i++)x[i] = -Lx + 2 * Lx / (float)Nx*(float)i;
	for (int i = 0; i < Ny; i++)y[i] = -Ly + 2 * Ly / (float)Ny*(float)i;
	
	for (int i = 0; i < Nx; i++){
		for (int j = 0; j < Ny; j++){
			float dis = sqrt(x[i]*x[i] + y[j]*y[j]);
			int index = i + j * Nx;
			phi[index]=0.5f + 0.5f * tanh(3.0f*(r0_cell-dis) / h_epsilon);

			rhoA[index] = 0.0f;
			if(x[i] > 0.0f){
				rhoA[index] = rhoAinitial * phi[index];
			}
			rhoM[index] = rhoMinitial * phi[index];
			ux[index] = 0.0f;
			uy[index] = 0.0f;			
		}
	}
}

void writetofile(const char *name, float *u, int Nx, int Ny){
	FILE *fp = fopen(name, "w+");
		for (int j = 0; j < Ny; j++){
			for (int i = 0; i < Nx; i++){
				fprintf(fp, "%f ", u[i + j*Nx]);
			}
			fprintf(fp, "\n");
		}
	
	fclose(fp);
}

__global__ void phievolve(float *phi_new, float *phi_old, float *grad_x, float *grad_y, float *absdphi, float *lap_phi, float *curv, float *ux, float *uy, int Nx, int Ny, float dt){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = i + j * Nx;

	float dG = 36.0f * phi_old[index] * (1.0f - phi_old[index]) * (1.0f - 2.0f * phi_old[index]);

	phi_new[index] = phi_old[index] + dt*(-ux[index] * grad_x[index] - uy[index] * grad_y[index] + d_Gamma*(d_epsilon*lap_phi[index] - dG / d_epsilon + curv[index] * d_epsilon * absdphi[index]));
}

__global__ void div_vel_pre(float *div_x, float *div_y, int dim, float *phi, float nu, float *duxdx, float *duxdy, float *duydx, float *duydy, int Nx, int Ny){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = i + j * Nx;

    	if(dim == 1){
        	div_x[index] = 2.0f * nu * phi[index] * duxdx[index] - nu * d_exceed_val * duxdx[index]; //sigma_xx
        	div_y[index] = nu * phi[index] * (duxdy[index] + duydx[index]) - nu * d_exceed_val * duxdy[index]; //sigma_xy
    	}
    	if(dim == 2){
        	div_x[index] = nu * phi[index] *(duydx[index] + duxdy[index]) - nu * d_exceed_val * duydx[index]; //sigma_yx
        	div_y[index] = 2.0f * nu * phi[index] * duydy[index] - nu * d_exceed_val * duydy[index]; //sigma_yy
    	}
}


__global__ void potential_force(float *output, float *phi, float *lap_phi, float *bendcore, float *phiarea, float A0, float dx, float dy, int Nx, int Ny){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = i + j * Nx;
	float dGphi = 36.0f * phi[index] * (1.0f - phi[index]) * (1.0f - 2.0f*phi[index]);
	float ddGphi = 36.0f*(1.0f - 6.0f*phi[index] + 6.0f*phi[index] * phi[index]);
	float core = lap_phi[index] - dGphi / d_epsilon / d_epsilon;


	float current_area = *phiarea * dx * dy; 
	float delta_area = 0.0f;
	if(current_area < d_areaMin)	
		delta_area = current_area - d_areaMin;
	if(current_area > d_areaMax)
		delta_area = current_area - d_areaMax;


	output[index] = -d_tension * d_epsilon * core + d_bending * d_epsilon * (bendcore[index] - ddGphi * core / d_epsilon / d_epsilon) + d_Mv * delta_area;
}

__global__ void rhs_poisson(float *rhs_x, float *rhs_y, float *ptl_force, float *phi, float *dphix, float *dphiy, float *vis_x, float *vis_y, float *act_x, float *act_y, float *rhoA, float *rhoM, float *ux, float *uy, int Nx, int Ny){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = i + j * Nx;
	//float Gphi = 18.0f * phi[index] * phi[index] * (1.0f - phi[index]) * (1.0f - phi[index]) / d_epsilon;
	rhs_x[index] = ptl_force[index] * dphix[index] + act_x[index] + vis_x[index] - d_xiM * rhoM[index] * ux[index] * phi[index];
	rhs_y[index] = ptl_force[index] * dphiy[index] + act_y[index] + vis_y[index] - d_xiM * rhoM[index] * uy[index] * phi[index];
}



__global__ void get_error(float *max_error, float *ux_new, float *ux_old, float *uy_new, float *uy_old, int *idx_eux, int *idx_ux, int *idx_euy, int *idx_uy, int Nx, int Ny){
	float err_ux = abs(ux_new[*idx_eux - 1] - ux_old[*idx_eux - 1]) / abs(ux_new[*idx_ux - 1]);
	float err_uy = abs(uy_new[*idx_euy - 1] - uy_old[*idx_euy - 1]) / abs(uy_new[*idx_uy - 1]);

	*max_error = MAX(err_ux, err_uy);
}


__global__ void div_advect_pre(float *div_x, float *div_y, float *dnsty, float *rho, float *ux, float *uy, int Nx, int Ny){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = i + j * Nx;
	//advection on the interface
	div_x[index] = dnsty[index] * rho[index] * ux[index];
	div_y[index] = dnsty[index] * rho[index] * uy[index];
}


__global__ void RD_evolve(float *rho_new, float *dnsty_new, float *dnsty_old, float *rho_old, float *advect, float *diffuse, float *react, float dt, float lamda, int Nx, int Ny){
	//with reaction terms
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = i + j * Nx;

	rho_new[index] = 0.0f;
	if(dnsty_old[index] > lamda){
		rho_new[index] = (2.0f * dnsty_old[index] - dnsty_new[index]) / dnsty_old[index] * rho_old[index] + dt / dnsty_old[index] *(diffuse[index] - advect[index] + react[index]);           
	}
}

__global__ void RD_evolve(float *rho_new, float *dnsty_new, float *dnsty_old, float *rho_old, float *advect, float *diffuse, float dt, float lamda, int Nx, int Ny){
	//without reaction terms
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = i + j * Nx;

	rho_new[index] = 0.0f;
	if(dnsty_old[index] > lamda){
		rho_new[index] = (2.0f * dnsty_old[index] - dnsty_new[index]) / dnsty_old[index] * rho_old[index] + dt / dnsty_old[index] *(diffuse[index] - advect[index]);           
	}

}

__global__ void normRhoM(float *rho, float *rho_area_new, float *rho_are_old, int Nx, int Ny){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = i + j * Nx;

	rho[index] = rho[index] * (*rho_are_old) / (*rho_area_new);
}


__global__ void diffRhoM(float *output, float *phi, float *rhoa, float Dm0, float Kd, int Nx, int Ny){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = i + j * Nx;

	output[index] = phi[index] * Dm0 / (1.0f + rhoa[index] / Kd);
}

__global__ void reactionRhoA(float *output, float *phi, float *rhoA, float *rhoAarea, float *phiarea, float rhoAtot, float kb, float K2a, float ka,    float kc, float dx, float dy,  int Nx, int Ny){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = i + j * Nx;
	
	float rhoAcyt = (rhoAtot - (*rhoAarea) * dx * dy) / ((*phiarea) * dx * dy);
	output[index] = phi[index] * (kb * (rhoA[index] * rhoA[index] / (K2a * K2a + rhoA[index] * rhoA[index]) + ka) * rhoAcyt - kc * rhoA[index]);

}

__global__ void bend_core(float *core, float *lap_phi, float *phi, int Nx, int Ny){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = i + j * Nx;

	if (i < Nx && j < Ny){
		core[index]=lap_phi[index]-36.0f*phi[index]*(1.0f-phi[index])*(1.0f-2.0f*phi[index])/d_epsilon/d_epsilon;
	}
}

__global__ void activeForce(float *div_x, float *div_y, int dir, float *phi, float etaRhoA, float etaRhoM, float *rhoA, float *rhoM, float *dphix, float *dphiy, float *phiarea, float dx, float dy, int Nx, int Ny){
	//active force by rhoA and rhoM
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = i + j * Nx;
	
	//float actin = 0.0f;
	//float myosin = 0.0f;
	//float current_area = *phiarea * dx * dy;
	//if(rhoA[index] > 0.0f)
	//	actin = rhoA[index];
	//if(rhoA[index] < 0.0f)
	//	myosin = abs(rhoA[index]);

	//x-dir
	if(dir == 1){
		div_x[index] = -etaRhoA * d_epsilon * phi[index] * rhoA[index] * dphix[index] * dphix[index] + etaRhoM * phi[index] * rhoM[index];
		div_y[index] = -etaRhoA * d_epsilon * phi[index] * rhoA[index] * dphix[index] * dphiy[index];
	}
	//y-dir
	if(dir == 2){
		div_x[index] = -etaRhoA * d_epsilon * phi[index] * rhoA[index] * dphix[index] * dphiy[index];
		div_y[index] = -etaRhoA * d_epsilon * phi[index] * rhoA[index] * dphiy[index] * dphiy[index] + etaRhoM * phi[index] * rhoM[index];
	}
}


/*--------------------------------------------------------------------------------------*/
/*Utility functions*/
/*--------------------------------------------------------------------------------------*/
__global__ void matrix_product(float *output, float *input1, float *input2, float alpha, int Nx, int Ny){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = i + j * Nx;

	output[index] = alpha * input1[index] * input2[index];
}

__global__ void add3matrix(float *output, float *input1, float *input2, float *input3){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = i + j * Nx;
	output[index] = input1[index] + input2[index] + input3[index];
}

__global__ void add2matrix(float *output, float *input1, float *input2){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = i + j * Nx;
	output[index] = input1[index] + input2[index];
}

__global__ void absarray(float *absay, float *ax, float *ay, int Nx, int Ny){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = i + j*Nx;
	
    absay[index] = sqrt(ax[index] * ax[index] + ay[index] * ay[index]);
}

__global__ void minus_matrix(float *output, float *minuend, float *substractor, int Nx, int Ny){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = i + j * Nx;

	output[index] = minuend[index] - substractor[index];
}

__global__ void xdir_center(float *xc_sin, float *xc_cos, float *phi, float *phi_area, float *x, float Lx, int m, int n){
	//return the matrix of phi*sin(x*pi/Lx), phi*cos(x*phi/Lx)
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = i + j*blockDim.x*gridDim.x;
	if(i< m && j<n){
		xc_sin[index] = phi[index] / (*phi_area) * sin(x[i] * M_PI / Lx);
		xc_cos[index] = phi[index] / (*phi_area) * cos(x[i] * M_PI / Lx);
	}
}

__global__ void rhoa_right(float *rhoa, float *phi, float x_sin, float x_cos, float *x, float Lx, int m, int n){
	//put rhoa at the right half plane of mass centern
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = i + j*m;
	if( i < m && j < n){
		rhoa[index] = 0.0f; 
		if(sin(x[i]*M_PI/Lx) * x_cos - cos(x[i]*M_PI/Lx)*x_sin > -0.5f && phi[index] > 1e-4){
			rhoa[index] = 1.0f;
		}
	}
}

__global__ void rhoa_osci(float *rhoa, float *rhom, float *phi, float x_sin, float x_cos, float t, float rhoa_period, float r_chi, float *x, float *y, float Lx, int m, int n){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = i + j*m;
	if( i < m && j < n){
		rhoa[index] = 0.0f; 
		rhom[index] = 0.0f;
		if(phi[index] > 1e-4){
			float dis = sqrt(x[i]*x[i] + y[j]*y[j]);
			float chi = phi[index] - (0.5f + 0.5f * tanh(3.0f * (r_chi - dis) / d_epsilon)); //phi - centered disc
			float signal = sin(2*M_PI*t/rhoa_period);
			if(signal > 0.0f)
				rhoa[index] = (1.0f + signal) / 2.0f * chi;
			//if(signal > 0.0f)
			//	rhoa[index] = tanh(signal / 0.2f);//(1.0f + signal) / 2.0f;
			
		
			//if(signal < -0.5f)
			float signal_m = sin(2*M_PI*t/rhoa_period);
			if(signal_m<= 0.0f)	
				rhom[index] = (1.0f - signal_m) / 2.0f * chi;
			//rhoa[index] = tanh(signal / 0.2f);
			//if (signal > -0.5f)
			//	rhoa[index] = 1.0f;//(signal + 1.0f) / 2.0f;
		}
	}
}

/*----------------------------------------------------------------------------------------------*/
/*Don't Change The Code After This Line if You Don't Want to Change the Differentiation Methods*/
/*----------------------------------------------------------------------------------------------*/

__global__ void curv(float *curv, float *phi_gradx, float *phi_grady, float *absdphi, int Nx, int Ny, float dx, float dy){
	//from gradient obtained by fft
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = i + j*Nx;
	//f: forward; b:backward
	int fdx = i + 1; if (i == Nx - 1) fdx = 0;
	fdx = fdx + j*Nx;
	int bdx = i - 1; if (i == 0) bdx = Nx - 1;
	bdx = bdx + j*Nx;
	int fdy = j + 1; if (j == Ny - 1) fdy = 0;
	fdy = i + fdy*Nx;
	int bdy = j - 1; if (j == 0) bdy = Ny - 1;
	bdy = i + bdy*Nx;

	curv[index] = 0.0f;
	if (absdphi[index] >= 0.01 && absdphi[fdx] >= 0.01 && absdphi[bdx] >= 0.01 && absdphi[fdy] >= 0.01 && absdphi[bdy] >= 0.01)
		curv[index] = -(phi_gradx[fdx] / absdphi[fdx] - phi_gradx[bdx] / absdphi[bdx]) / 2.0f / dx -
		(phi_grady[fdy] / absdphi[fdy] - phi_grady[bdy] / absdphi[bdy]) / 2.0f / dy;
	
}

__global__ void curv(float *curv, float *phi, float *absdphi, int m, int n, float dx, float dy){
	//from gradient obtained by fft
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = i + j*m;
	if(i<m && j<n){
		curv[index] = 0.0f;
		if(absdphi[index]>0.001f){
			int r_dx = i + 1; if (i == m - 1) r_dx = 0;
			int l_dx = i - 1; if (i == 0) l_dx = m - 1;
			int u_dy = j + 1; if (j == n - 1) u_dy = 0;
			int d_dy = j - 1; if (j == 0) d_dy = n - 1;
			//geth the 9 points needed for curvature calculation
			float phi_i_j=phi[index];          //(i,j)
			float phi_ip1_j=phi[r_dx+j*m];      //(i+1,j)
			float phi_im1_j=phi[l_dx+j*m];      //(i-1,j)
			float phi_i_jp1=phi[i+u_dy*m];      //(i,j+1)
			float phi_i_jm1=phi[i+d_dy*m];      //(i,j-1)
			float phi_ip1_jp1=phi[r_dx+u_dy*m];  //(i+1,j+1)
			float phi_ip1_jm1=phi[r_dx+d_dy*m];  //(i+1,j-1)
			float phi_im1_jp1=phi[l_dx+u_dy*m];  //(i-1,j+1)
			float phi_im1_jm1=phi[l_dx+d_dy*m];  //(i-1,j-1)
			
			float phix_iphalf_j = (phi_ip1_j - phi_i_j  )/dx;
			float phix_imhalf_j = (phi_i_j   - phi_im1_j)/dx;
			float phiy_i_jphalf = (phi_i_jp1 - phi_i_j  )/dy;
			float phiy_i_jmhalf = (phi_i_j   - phi_i_jm1)/dy;

			float phiy_iphalf_j = (phi_ip1_jp1 + phi_i_jp1   - phi_ip1_jm1 - phi_i_jm1  )/(4.0f*dy);
			float phiy_imhalf_j = (phi_i_jp1   + phi_im1_jp1 - phi_i_jm1   - phi_im1_jm1)/(4.0f*dy);
			float phix_i_jphalf = (phi_ip1_jp1 + phi_ip1_j   - phi_im1_jp1 - phi_im1_j  )/(4.0f*dx);
			float phix_i_jmhalf = (phi_ip1_j   + phi_ip1_jm1 - phi_im1_j   - phi_im1_jm1)/(4.0f*dx);

			float grad_phi_abs_iphalf_j = sqrt( phix_iphalf_j * phix_iphalf_j + phiy_iphalf_j * phiy_iphalf_j );
			float grad_phi_abs_imhalf_j = sqrt( phix_imhalf_j * phix_imhalf_j + phiy_imhalf_j * phiy_imhalf_j);
			float grad_phi_abs_i_jphalf = sqrt( phix_i_jphalf * phix_i_jphalf + phiy_i_jphalf * phiy_i_jphalf );
			float grad_phi_abs_i_jmhalf = sqrt( phix_i_jmhalf * phix_i_jmhalf + phiy_i_jmhalf * phiy_i_jmhalf );

			curv[index] = - ( phix_iphalf_j / grad_phi_abs_iphalf_j - phix_imhalf_j / grad_phi_abs_imhalf_j )/dx - ( phiy_i_jphalf / grad_phi_abs_i_jphalf - phiy_i_jmhalf / grad_phi_abs_i_jmhalf )/dy;
		}
	}
}

void fftcoeffs(float *d_d1x, float *d_d1y, float *d_d2x, float *d_d2y){
	//FFT coefficients, d_d1x, d_d1y are device 1-st derivatie grid matrixes, d_d2x, d_d2y are device 2-nd derivative matrix
	float *h_kx2 = (float *)malloc(Nx*sizeof(float));
	for (int i = 0; i <= Nx / 2; i++) h_kx2[i] = (float)i * M_PI / Lx;
	for (int i = Nx / 2 + 1; i < Nx; i++) h_kx2[i] = ((float)i - (float)Nx) * M_PI / Lx;
	cudaMemcpy(d_d2x, h_kx2, Nx*sizeof(float), cudaMemcpyHostToDevice);

	float *h_ky2 = (float *)malloc(Ny*sizeof(float));
	for (int i = 0; i <= Ny / 2; i++) h_ky2[i] = (float)i * M_PI / Ly;
	for (int i = Ny / 2 + 1; i < Ny; i++) h_ky2[i] = ((float)i - (float)Ny) * M_PI / Ly;
	cudaMemcpy(d_d2y, h_ky2, Ny*sizeof(float), cudaMemcpyHostToDevice);

	float *h_kx1 = (float *)malloc(Nx*sizeof(float));
	for (int i = 0; i < Nx / 2; i++) h_kx1[i] = (float)i * M_PI / Lx;
	h_kx1[Nx / 2] = 0.0f;
	for (int i = Nx / 2 + 1; i < Nx; i++) h_kx1[i] = ((float)i - (float)Nx) * M_PI / Lx;
	cudaMemcpy(d_d1x, h_kx1, Nx*sizeof(float), cudaMemcpyHostToDevice);

	float *h_ky1 = (float *)malloc(Ny*sizeof(float));
	for (int i = 0; i < Ny / 2; i++)h_ky1[i] = (float)i * M_PI / Ly;
	h_ky1[Ny / 2] = 0.0f;
	for (int i = Ny / 2 + 1; i < Ny; i++) h_ky1[i] = ((float)i - (float)Ny) * M_PI / Ly;
	cudaMemcpy(d_d1y, h_ky1, Ny*sizeof(float), cudaMemcpyHostToDevice);

	free(h_kx1);
	free(h_kx2);
	free(h_ky1);
	free(h_ky2);
}

__global__ void fftR2C_deriv1(cufftComplex *input, cufftComplex *output, int dim, float *coeff, int Nx, int Ny){
	//x:dim=1; y:dim=2;
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = i + j*(Nx / 2 + 1);
	int dir[2]={i,j};
	if (i < (Nx / 2 + 1) && j < Ny){
		cufftComplex temp = input[index];
		output[index].x = -temp.y*coeff[dir[dim-1]] / (float)Nx / (float)Ny;
		output[index].y = temp.x*coeff[dir[dim-1]] / (float)Nx / (float)Ny;
	}
}

__global__ void fftR2C_lap(cufftComplex *input, cufftComplex *output, float *coeff_x, float *coeff_y, int Nx, int Ny){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = i + j*(Nx / 2 + 1);
	if (i < (Nx / 2 + 1) && j < Ny){
		cufftComplex temp = input[index];
		float coeff = coeff_x[i] * coeff_x[i] + coeff_y[j] * coeff_y[j];
		output[index].x = -temp.x*coeff / (float)Nx / (float)Ny;
		output[index].y = -temp.y*coeff / (float)Nx / (float)Ny;
	}
}

void grad_fft(float *output_x, float *output_y, float *input, float *deriv_vec_x, float *deriv_vec_y, cufftComplex *R2C_buffer, cufftComplex *C2R_buffer, int Nx, int Ny){
	
	cufftExecR2C(plan_R2C, input, R2C_buffer);

	fftR2C_deriv1 <<<blocks, threadsperblock >>>(R2C_buffer, C2R_buffer,1, deriv_vec_x, Nx, Ny);
	cufftExecC2R(plan_C2R, C2R_buffer, output_x);

	fftR2C_deriv1 <<<blocks, threadsperblock >>>(R2C_buffer, C2R_buffer,2, deriv_vec_y, Nx, Ny);
	cufftExecC2R(plan_C2R, C2R_buffer, output_y);
}

void lap_fft(float *output, float *input, float *deriv_vec_x, float *deriv_vec_y, cufftComplex *R2C_buffer, cufftComplex *C2R_buffer, int Nx, int Ny){
	cufftExecR2C(plan_R2C, input, R2C_buffer);
	fftR2C_lap <<<blocks, threadsperblock >>>(R2C_buffer, C2R_buffer, deriv_vec_x,deriv_vec_y, Nx, Ny);
	cufftExecC2R(plan_C2R, C2R_buffer, output);
}

void div_fft(float *output, float *input_x, float *input_y, float *deriv_vec_x, float *deriv_vec_y, cufftComplex *R2C_buffer, cufftComplex *C2R_buffer, int Nx, int Ny){
	//in-space transform, notice that input_x and input_y will change values when function is called
	cufftExecR2C(plan_R2C, input_x, R2C_buffer);
	fftR2C_deriv1 <<<blocks, threadsperblock >>>(R2C_buffer, C2R_buffer, 1, deriv_vec_x, Nx, Ny);
	cufftExecC2R(plan_C2R, C2R_buffer, input_x);

	cufftExecR2C(plan_R2C, input_y, R2C_buffer);
	fftR2C_deriv1 <<<blocks, threadsperblock >>>(R2C_buffer, C2R_buffer, 2, deriv_vec_y, Nx, Ny);
	cufftExecC2R(plan_C2R, C2R_buffer, input_y);

	add2matrix<<<blocks, threadsperblock>>>(output, input_x, input_y);
}

void div_pst_fft(float *output, float *c_pst, float *rho, float coeff_diff, float *deriv_vec_x2, float *deriv_vec_y2, float *buffer_x, float *buffer_y, cufftComplex *R2C_buffer, cufftComplex *C2R_buffer, int Nx, int Ny){
	//notice this differential process uses the 2-nd kind of derivative vector, corresponding to 2-nd order derivative
	grad_fft(buffer_x, buffer_y, rho, deriv_vec_x2, deriv_vec_y2, R2C_buffer, C2R_buffer, Nx, Ny);

	matrix_product<<<blocks, threadsperblock >>>(buffer_x, buffer_x, c_pst, coeff_diff, Nx, Ny);
	matrix_product<<<blocks, threadsperblock >>>(buffer_y, buffer_y, c_pst, coeff_diff, Nx, Ny);

	div_fft(output, buffer_x, buffer_y, deriv_vec_x2, deriv_vec_y2, R2C_buffer, C2R_buffer, Nx, Ny);
}

__global__ void div_pst_fd(float *output, float *c_pst, float *rho, float coeff_diff, float dx, float dy, int Nx, int Ny){
	//finite differential of the diffusion term
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = i + j * Nx;

	int r_dx = i + 1; if (i == Nx - 1) r_dx = 0;
	r_dx = r_dx + j * Nx;
	int l_dx = i - 1; if (i == 0) l_dx = Nx - 1;
	l_dx = l_dx + j * Nx;
	int u_dy = j + 1; if (j == Ny - 1) u_dy = 0;
	u_dy = i + u_dy * Nx;
	int d_dy = j - 1; if (j == 0) d_dy = Ny - 1;
	d_dy = i + d_dy * Nx;

	output[index] = coeff_diff * ((c_pst[r_dx] + c_pst[index]) / 2.0f * (rho[r_dx] - rho[index]) / dx - (c_pst[l_dx] + c_pst[index]) / 2.0f * (rho[index] - rho[l_dx]) / dx) / dx +
		coeff_diff * ((c_pst[u_dy] + c_pst[index]) / 2.0f * (rho[u_dy] - rho[index]) / dy - (c_pst[d_dy] + c_pst[index]) / 2.0f * (rho[index] - rho[d_dy]) / dy) / dy;

}

void par_deriv1_fft(float *output, float *input, int dim, float *deriv_vec, cufftComplex *R2C_buffer, cufftComplex *C2R_buffer, int Nx, int Ny){
	
	cufftExecR2C(plan_R2C, input, R2C_buffer);
	if(dim == 1){
		fftR2C_deriv1 <<<blocks, threadsperblock >>>(R2C_buffer, C2R_buffer,1, deriv_vec, Nx, Ny);
		cufftExecC2R(plan_C2R, C2R_buffer, output);
	}
   	 if(dim == 2){
	    fftR2C_deriv1<<<blocks, threadsperblock >>>(R2C_buffer, C2R_buffer,2, deriv_vec, Nx, Ny);
		cufftExecC2R(plan_C2R, C2R_buffer, output);
	}
}

__global__ void poissonR2C(cufftComplex *input, cufftComplex *output, float *coeff_x, float *coeff_y, float coeff0, float coeff2, int Nx, int Ny){
    //equation of coeff0*u - coeff2*Delta(u) = f
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = i + j*(Nx / 2 + 1);
	if (i < (Nx / 2 + 1) && j < Ny){
		float coeff = coeff_x[i] * coeff_x[i] + coeff_y[j] * coeff_y[j];
		cufftComplex temp = input[index];
		output[index].x = temp.x / (coeff0 + coeff2 * coeff) / (float)Nx / (float)Ny;
		output[index].y = temp.y / (coeff0 + coeff2 * coeff) / (float)Nx / (float)Ny;
	}
}

void poisson_sol(float *sol, float *rhs, float coeff0, float coeff2, float *coeff_x, float *coeff_y, cufftComplex *R2C_buffer, cufftComplex *C2R_buffer, int Nx, int Ny){
	cufftExecR2C(plan_R2C, rhs, R2C_buffer);
	poissonR2C <<<blocks, threadsperblock >>>(R2C_buffer, C2R_buffer, coeff_x, coeff_y, coeff0, coeff2, Nx, Ny);
	cufftExecC2R(plan_C2R, C2R_buffer, sol);
}

/*----------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------*/
