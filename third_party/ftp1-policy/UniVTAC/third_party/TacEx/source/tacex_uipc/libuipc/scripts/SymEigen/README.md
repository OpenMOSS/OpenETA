# SymEigen

## Introduction

This is a simple Sympy extension to generate Eigen C++ code from the given expression. 
I concentrate on the Vector/Vector derivative, which is the most common operation in the optimization problem.

To use it, you need to install the `sympy` library. You can install it using pip:

```bash
pip install sympy
```

## Install

This library is so simple that you can just copy the `SymEigen.py` file to your project directory.

## Usage

The [QuickStart](./QuickStart.ipynb)  [Highly Recommended :white_check_mark:] can be run using Jupyter Notebook. 
You can use VSCode with Jupyter Notebook extension to run the code, or any other way you like.

## Quick View

Say we are calculating the Energy of a spring. The equation is as follows:

$$
E = \frac{1}{2} k (\|\mathbf{x}-\mathbf{y}\| - L_0)^2
$$

To compactly write the equation, we can define the following variables, fully 6 Dof:

$$
\mathbf{X} = 
\begin{bmatrix}
\mathbf{x}_l \\
\mathbf{y}_r
\end{bmatrix}
$$

I paste the code here for your quick view:

```python
from SymEigen import *
# Then we can define such a matrix as follows:
X = Eigen.Vector('X', 6)
# Other coefficients are defined as follows:
k = Eigen.Scalar('k')
L0 = Eigen.Scalar('L0')
# It's easy to calculate the Energy as follows:
X_l = Matrix(X[0:3])
X_r = Matrix(X[3:6])
d = X_l - X_r 
E = k * (sqrt(d.T * d) - L0)**2 / 2
# We use `VecDiff` to calculate the Vec/Vec derivative, so the Gradient of the Energy is:
G = VecDiff(E, X)
# So for the Hessian, we have:
H = VecDiff(G, X)
# To generate Eigen Cpp code, we should do the following:
# 1. Declare a `EigenFunctionGenerator` as a context.
# 2. Wrap the Input Variable to a `Closure`.
# 3. Call the `Closure` by inputting the function name and `Expr` (e.g. the `E`, `G`, `H`).
Gen = EigenFunctionGenerator()
Closure = Gen.Closure(k, L0, X)
# First, we generate the Eigen Cpp code for the Energy:
print(Closure('SpringEnergy', E, 'E'))
# Then, Gradient:
print(Closure('SpringGradient', G, 'G'))
# Finally, Hessian:
print(Closure('SpringHessian', H, 'H'))
```

And I just paste some output here, because the output is too long:

```cpp
template <typename T>
__device__ __host__ void SpringGradient(Eigen::Vector<T,6>& G, const T& k, const T& L0, const Eigen::Vector<T,6>& X)
{
/*****************************************************************************************************************************
Symbol Name Mapping:
k:
    -> {}
    -> Matrix([[k]])
L0:
    -> {}
    -> Matrix([[L0]])
X:
    -> {}
    -> Matrix([[X(0)], [X(1)], [X(2)], [X(3)], [X(4)], [X(5)]])
*****************************************************************************************************************************/
/* Sub Exprs */
auto x0 = X(0) - X(3);
auto x1 = X(1) - X(4);
auto x2 = X(2) - X(5);
auto x3 = std::pow(std::pow(x0, 2) + std::pow(x1, 2) + std::pow(x2, 2), 1.0/2.0);
auto x4 = k*(-L0 + x3)/x3;
/* Simplified Expr */
G(0) = x0*x4;
G(1) = x1*x4;
G(2) = x2*x4;
G(3) = -x0*x4;
G(4) = -x1*x4;
G(5) = -x2*x4;
}
```

## Further

`SymEigen` concentrates on the Vector/Vector derivative. If you want to differentiate between Matrix/Matrix, Matrix/Vectors or other operations, the recommended way is to use the `Vectorize` function to convert the Matrix to a Vector, then use the `VecDiff` function to calculate the derivative.

For example, in Continuum Mechanics, we have the Deformation Gradient `F`, which is a 3 by 3 matrix, we can convert it to a 9 by 1 vector, then use the `VecDiff` function to calculate the derivative.

```python
F = Eigen.Matrix('F', 3, 3)
VecF = F.Vectorize('VecF')
Ic =  Trace(F.T*F)
dIcdVecF = VecDiff(Ic, VecF)
```

The output will be:

$$
\begin{bmatrix}
2 F(0,0)\\
2 F(1,0)\\
2 F(2,0)\\
2 F(0,1)\\
2 F(1,1)\\
2 F(2,1)\\
2 F(0,2)\\
2 F(1,2)\\
2 F(2,2)
\end{bmatrix}
$$

SymEigen does the Element Name Mapping for you, which means, SymEigen doesn't care about the layout of the elements, the only thing it cares about is the unique element name. All differentiations are performed element-wise. The layout only affects the generated code. And you can freely substitute the `VecF` as `F`, when you generate the Eigen C++ code.

```python
Gen = EigenFunctionGenerator()
Closure = Gen.Closure(VecF)
print(Closure('dIcdVecF', dIcdVecF))
Closure = Gen.Closure(F)
print(Closure('dIcdF', ddICddVecF))
```

The output function will be (I take the most important part here):

```cpp
template <typename T>
void dIcdVecF(Eigen::Vector<T,9>& R, const Eigen::Vector<T,9>& VecF)
{
R(0) = 2*VecF(0);
R(1) = 2*VecF(1);
R(2) = 2*VecF(2);
R(3) = 2*VecF(3);
R(4) = 2*VecF(4);
R(5) = 2*VecF(5);
R(6) = 2*VecF(6);
R(7) = 2*VecF(7);
R(8) = 2*VecF(8);
}

template <typename T>
void dIcdF(Eigen::Vector<T,9>& R, const Eigen::Matrix<T,3,3>& F)
{
R(0) = 2*F(0,0);
R(1) = 2*F(1,0);
R(2) = 2*F(2,0);
R(3) = 2*F(0,1);
R(4) = 2*F(1,1);
R(5) = 2*F(2,1);
R(6) = 2*F(0,2);
R(7) = 2*F(1,2);
R(8) = 2*F(2,2);
}
```

The `R` is the same, but the input is different. You can use the `VecF` or `F` as you like. But I recommend using the `VecF` as the input, because it's clearer to understand and use the code. Because the `Vectorize` operation is well-defined.

### Final Takeaway

- Always vectorize the Matrix to a Vector, then use the `VecDiff` function to calculate the derivative. The `SymEigen` will do the rest for you.
- The default [Vectorization](https://en.wikipedia.org/wiki/Vectorization_(mathematics)) is column-wise.

