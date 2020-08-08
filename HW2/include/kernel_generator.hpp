#ifndef KERNEL_GENERATOR_HPP__
#define KERNEL_GENERATOR_HPP__

#include "common.hpp"

class KernelGenerator{
public:
	static const float pi;

public:
	static Mat getGaussianBlur3x3() {
		return (Mat_<float>(3, 3) << 1.0 / 16, 2.0 / 16, 1.0 / 16, 
									 2.0 / 16, 4.0 / 16, 2.0 / 16, 
									 1.0 / 16, 2.0 / 16, 1.0 / 16);
	}
	static Mat getGaussianBlur5x5() {
		return (Mat_<float>(5, 5) << 2.0 / 159, 4.0 / 159, 5.0 / 159, 4.0 / 159, 2.0 / 159, 
			                         4.0 / 159, 9.0 / 159, 12.0 / 159, 9.0 / 159, 4.0 / 159,
									 5.0 / 159, 12.0 / 159, 15.0 / 159, 12.0 / 159, 5.0 / 159, 
									 4.0 / 159, 9.0 / 159, 12.0 / 159, 9.0 / 159, 4.0 / 159, 
									 2.0 / 159, 4.0 / 159, 5.0 / 159, 4.0 / 159, 2.0 / 159);
	}
	static Mat getSobelKernelGx(){
        return (Mat_<float>(3,3) << -1, 0, 1, 
								  -2, 0, 2, 
			                      -1, 0, 1);
    }

    static Mat getSobelKernelGy(){
        return (Mat_<float>(3,3) << -1, -2, -1, 
			                       0, 0, 0, 
			                       1, 2, 1);
    }

    static Mat getPrewittKernelGx(){
        return (Mat_<float>(3,3) << -1, 0, 1,
								  -1, 0, 1, 
			                      -1, 0, 1);
    }

    static Mat getPrewittKernelGy(){
        return (Mat_<float>(3,3) << -1, -1, -1, 
			                       0, 0, 0, 
			                       1, 1, 1);
    }

    static Mat getLaplaceKernel(){ //fix
        return (Mat_<float>(3,3) << 0, -1, 0, 
			                     -1, 4, -1, 
			                      0, -1, 0);
    }

	static Mat createGaussianKernel(int gaussSize, float gaussStd){
		Mat gaussKernel = Mat::zeros(gaussSize, gaussSize, CV_32FC1);
		float sum = 0.0;
		float var = 2*gaussStd*gaussStd;
		float r;
		
		for(int y = -(gaussSize/2); y <= gaussSize/2 ; y++){
			for(int x = -(gaussSize/2); x <= gaussSize/2; x++){
				r = sqrt(x*x + y*y);
				gaussKernel.at<float>(y + gaussSize/2, x + gaussSize/2) = exp(-(r*r)/var) / (pi*var);
				sum += gaussKernel.at<float>(y + gaussSize/2, x + gaussSize/2);
			}
		}

		for(int i = 0; i < gaussSize; i++){
			for(int j = 0; j < gaussSize; j++){
				gaussKernel.at<float>(i, j)/=sum;
			}
		}

		return gaussKernel;
	}

};

#endif // KERNEL_GENERATOR_HPP__
