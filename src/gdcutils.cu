extern "C" {
   #include "lua.h"
   #include "lualib.h"
   #include "lauxlib.h"
}

#include "luaT.h"
#include "THC.h"

#include "cuda_runtime.h"
#include "npp.h"

#define TB 128

#define NPP_CALL(x) {const NppStatus a = (x); if (a != NPP_SUCCESS) {printf("\nNPP Error: (err_num=%d) \n", a);} }

THCState* getCutorchState(lua_State* L)
{
	lua_getglobal(L, "cutorch");
	lua_getfield(L, -1, "getState");
	lua_call(L, 0, 1);
	THCState *state = (THCState*) lua_touserdata(L, -1);
	lua_pop(L, 2);
	return state;
}

void checkCudaError(lua_State *L) {
	cudaError_t status = cudaPeekAtLastError();
	if (status != cudaSuccess) {
		luaL_error(L, cudaGetErrorString(status));
	}
}

THCudaTensor *new_tensor_like(THCState *state, THCudaTensor *x)
{
	THCudaTensor *y = THCudaTensor_new(state);
	THCudaTensor_resizeAs(state, y, x);
	return y;
}

__global__ void depth_filter(float *img, float*out, int size, int height, int width, int threshold)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id == 0)
	{
		printf("Size is %d\n", size);
		printf("Height is %d\n", height);
		printf("Width is %d\n", width);
		printf("Threshold is %d\n", threshold);
		printf("Img first value is %f\n", img[0]);
	}
	if(id < size)
	{
		//int col = id % width;
		//int row = id / width;
		if(img[id] < threshold)
		{
			out[id] = 0;
		}
		else
		{
			out[id] = img[id];
		}
	}
}

int depth_filter(lua_State *L)
{
	printf("Entering depth_filter\n");
	THCState *state = getCutorchState(L);
	printf("Got the state\n");
	THCudaTensor *img = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	printf("Got the img\n");
	int threshold = luaL_checknumber(L, 2);
	printf("Got the threshold\n");
	THCudaTensor *out = new_tensor_like(state, img);
	printf("Made the out tensor\n");
	
	depth_filter<<<(THCudaTensor_nElement(state, out) - 1) / TB + 1, TB>>>(
			THCudaTensor_data(state, img),
			THCudaTensor_data(state, out),
			THCudaTensor_nElement(state, out),
			THCudaTensor_size(state, out, 2),
			THCudaTensor_size(state, out, 3),
			threshold);
	printf("Executed depth_filter\n");
	checkCudaError(L);
	printf("Checked cuda error\n");
	luaT_pushudata(L, out, "torch.CudaTensor");
	printf("Pushed data\n");
	return 1;
}

int erode(lua_State *L)
{
	printf("Entered to ERODE method\n");
	
	THCState *state = getCutorchState(L);
	
	THCudaTensor *img = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *kernel = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *out = new_tensor_like(state, img);
	
	printf("Parameters read correctly\n");
	
	long half_kernel_width = (kernel->size[0] - 1) / 2;
	long int y = img->size[2];
	long int x = img->size[3];
	long int c = img->size[1];
	
	printf("Dimensions retrieved correctly: y:%d, x:%d, c:%d, half kernel:%d\n", y, x, c, half_kernel_width);
			
	NppiSize oSizeRoi;
	oSizeRoi.width = x - (half_kernel_width * 2);
	oSizeRoi.height = y - (half_kernel_width * 2);
	
	//oSizeRoi.width = 3;
	//oSizeRoi.height = 1;
	
	printf("ROI size generated: width:%d, height:%d\n", oSizeRoi.width, oSizeRoi.height);
	
	NppiSize oMaskSize;
	oMaskSize.width = kernel->size[0];
	oMaskSize.height = kernel->size[1];
	
	printf("Mask size generated: width:%d, height:%d\n", oMaskSize.width, oMaskSize.height);
	
	NppiPoint oAnchor;
	oAnchor.x = half_kernel_width;
	oAnchor.y = half_kernel_width;
	
	printf("Anchor point generated: x:%d, y:%d\n", oAnchor.x, oAnchor.y);
	
	printf("Test accesing to an element of the img. Element 0: %f\n", &(img->storage->data)[1242]);
	printf("Test size of the img data. Element 0: %f\n", sizeof(&(img->storage->data)));
	printf("Test size to an element of the img. Element 0: %d\n", sizeof(typeof(&(img->storage->data)[1242])));
	
	Npp32f *pSrc = img->storage->data + img->stride[2] * half_kernel_width + img->stride[3] * half_kernel_width;
	
	Npp32f *pDst = out->storage->data + out->stride[0] * half_kernel_width + out->stride[1] * half_kernel_width;
	
	
	Npp8u pMask[9];
		
	/*float *kernel_data = THCudaTensor_data(state, kernel);
	
	for(int i = 0; i < oMaskSize.width; i++)
	{
		printf("Reading kernel data %f\n", kernel_data[i]);
		pMask[i] = static_cast<unsigned int>(kernel_data[i]);
	}*/
	
	pMask[0] = 0;
	pMask[1] = 1;
	pMask[2] = 0;
	pMask[3] = 1;
	pMask[4] = 1;
	pMask[5] = 1;
	pMask[6] = 0;
	pMask[7] = 1;
	pMask[8] = 0;
	
	
	
	printf("Checking step sizes: %d, %d\n", sizeof(Npp32f) * img->stride[2], sizeof(Npp32f) * out->stride[3]);
	
	NPP_CALL(nppiErode_32f_C1R(pSrc, sizeof(Npp32f) * (img->stride[2] - 2), out->storage->data, sizeof(Npp32f) * img->stride[2], oSizeRoi, pMask, oMaskSize, oAnchor));
	
	printf("nppiErode executed...\n");
	
	
	checkCudaError(L);
	
	luaT_pushudata(L, out, "torch.CudaTensor");
	
	return 1;
}

static const struct luaL_Reg funcs[] = {
	{"depth_filter", depth_filter},
	{"erode", erode},
	{NULL, NULL}
};

extern "C" int luaopen_libgdcutils(lua_State *L) {
	srand(42);
	luaL_openlib(L, "gdcutils", funcs, 0);
	return 1;
}