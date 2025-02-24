# Neural Network-Driven Adaptive Parameter Selection for the Local Method of Fundamental Solutions (LMFSNN)
This repository provides numerical examples of the **Local method of fundamental solutions neural network** (**LMFSNN**). 

Recently, the **Local Method of Fundamental Solutions (LMFS)** has demonstrated remarkable capabilities in solving partial differential equations (PDEs). However, the **source point radius R**, as an artificial parameter, significantly impacts the accuracy of numerical results. 

Inspired by Physics-Informed Neural Networks (PINNs), we propose **Neural Network-Driven Adaptive Parameter Selection for the Local Method of Fundamental Solutions (LMFSNN)** that adaptively determines the optimal source point radius. Our numerical experiments indicate that LMFSNN achieves superior accuracy and stability compared to the conventional LMFS. Furthermore, the methodology developed in LMFSNN can be readily extended to other numerical methods, such as the **Boundary Knot Method (BKM)**.

To illustrate this, we present Numerical Example 1:

2D Laplace equation (**Eq. 24** in the manuscript)

**PDE**: $\Delta u(x,y) = 0,\quad x, y \in \Omega.$

For more details in terms of Local Method of Fundamental Solutions Neural Network and numerical examples, please refer to our paper.

# Enviornmental settings
 - Python  3.10.10
 - Pytorch  2.2.0 
 - Numpy  1.24.4
 - SciPy  1.11.4
 - Matlab  0.1
 - Matlabengine  23.2.3
 - Matplotlib  3.7.1

# Contact us
For questions regarding the code, please contact:

leimin@tyut.edu.cn

1552022371@qq.com
