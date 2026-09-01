from sympy import *
from sympy.codegen import *
import sympy.printing.c as ccode

AuthorName = '''MuGdxy'''
AuthorGitHub = '''https://github.com/MuGdxy/SymEigen'''
AuthorEmail = '''lxy819469559@gmail.com'''
__version__ = '0.1.0'
class Sym:
    def Vectorize(M : Matrix, expand_dir : str = 'col'):
        # if type(M) == EigenMatrix:
        #     raise ValueError("Don't call Vectorize on EigenMatrix, use EigenMatrix.Vectorize instead")
        V = zeros(M.shape[0]*M.shape[1], 1)
        if expand_dir == 'col':
            i = 0
            # expand in column major order
            for j in range(M.shape[1]):
                for k in range(M.shape[0]):
                    V[i] = M[k, j]
                    i += 1
        elif expand_dir == 'row':
            i = 0
            # expand in row major order
            for j in range(M.shape[0]):
                for k in range(M.shape[1]):
                    V[i] = M[j, k]
                    i += 1
        else:
            raise ValueError('expand_dir must be either "col" or "row"')
        return V


def VecDiff(VecF, VecX):
    # if VecF has no shape attribute, it is a scalar
    if not hasattr(VecF, 'shape'):
        VecF = Matrix([VecF])
    if VecF.shape[1] != 1 or VecX.shape[1] != 1:
        raise ValueError(f'Inputs of VecDiff must be a column vector or a scalar, VecF[{VecF.shape[0]},{VecF.shape[1]}], VecX[{VecX.shape[0]},{VecX.shape[1]}]')
    if(VecF.shape[0] == 1 and VecF.shape[1] == 1): # scalar
        return VecF.jacobian(VecX).reshape(VecX.shape[0], 1)
    else:
        return VecF.jacobian(VecX).reshape(VecF.shape[0], VecX.shape[0])

class EigenMatrix(MutableDenseMatrix):
    def __init__(self, *args, **kwargs):
        self.name = None
        self.to_origin_element_name = {}
        self.from_origin_element_name = {}
        self.origin_matrix = None
        self.IS_EIGEN_MATRIX = True

    def ValueType(self, Type = 'T'):
        if self.shape[0] == 1 and self.shape[1] == 1:
            return Type
        elif self.shape[1] == 1:
            return f'Eigen::Vector<{Type},{self.shape[0]}>'
        elif self.shape[0] == 1:
            return f'Eigen::RowVector<{Type},{self.shape[1]}>'
        else:
            return f'Eigen::Matrix<{Type},{self.shape[0]},{self.shape[1]}>'
    
    def RefType(self, Type = 'T'):
        return f'{self.ValueType(Type)}&'
    
    def CRefType(self, Type = 'T'):
        return f'const {self.ValueType(Type)}&'

    # def [i, j]
    def At(self, i, j):
        '''
        return the string representation of the element at (i, j)
        e.g.
            A.at(0, 0) -> 'A(0, 0)' for a matrix A
            A.at(0, 0) -> 'A(0)' for a vector A
            A.at(0, 0) -> 'A' for a scalar A
        '''
        if self.shape[0] == 1 and self.shape[1] == 1:
            assert i == 0 and j == 0, 'Scalar has only one element'
            return self.name
        elif self.shape[1] == 1:
            assert j == 0, 'Vector has only one column'
            return self.name + f'({i})'
        elif self.shape[0] == 1:
            assert i == 0, 'Vector has only one row'
            return self.name + f'({j})'
        return self.name + f'({i},{j})'

    def OriginMatrixName(self):
        if self.origin_matrix is None:
            return self.name
        return self.origin_matrix.name
    
    def MatrixName(self):
        return self.name
    
    def IsIndependent(self):
        return self.origin_matrix is None

    def Vectorize(self, Name, expand_dir : str = 'col'):
        SymV = Sym.Vectorize(self, expand_dir)
        Vector = EigenMatrix(SymV)
        Vector.name = Name
        Vector._build_remap(SymV)
        Vector.origin_matrix = self
        return Vector
    
    def _build_remap(self, M : MutableDenseMatrix):
        assert self.shape == M.shape, 'Shape mismatch'
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if(not M[i, j].is_number): # if it is a symbol
                    self.to_origin_element_name[self.At(i, j)] = str(M[i, j])
                    self.from_origin_element_name[str(M[i, j])] = self.At(i, j)
    
    def __rtruediv__(self, other):
        if(self.shape[0]==1 and self.shape[1]==1):
            return other / self[0, 0]
        else:
            raise ValueError('Only scalar is supported for right division')

class Eigen:
    def Matrix(Name, M, N):
        EMat = EigenMatrix(zeros(M,N))
        if(M == 1 and N == 1):
            EMat[0, 0] = Symbol(Name)
        elif(N == 1):
            for i in range(M):
                EMat[i, 0] = Symbol(f'{Name}({i})')
        elif(M == 1):
            for i in range(N):
                EMat[0, i] = Symbol(f'{Name}({i})')
        else:
            for i in range(M):
                for j in range(N):
                    EMat[i, j] = Symbol(f'{Name}({i},{j})')
        EMat.name = Name
        EMat.origin_matrix = EMat
        return EMat
    
    def FromSympy(Name, M : MutableDenseMatrix):
        if hasattr(M, 'shape'):
            EMat = EigenMatrix(M)
            EMat.name = Name
            EMat._build_remap(M)
            EMat.origin_matrix = None
        else:
            EMat = Eigen.Scalar(Name)
        return EMat

    def Vector(Name, N):
        return Eigen.Matrix(Name, N, 1)
    
    def RowVector(Name, N):
        return Eigen.Matrix(Name, 1, N)
    
    def Scalar(Name):
        return Eigen.Matrix(Name, 1, 1)

class EigenPrinter(ccode.C11CodePrinter):
    def _print_Pow(self, expr):
        base = f'{self._print(expr.base)}'
        exp = f'{self._print(expr.exp)}'
        # if expr.exp == -1:
        #     return f'(1.0 / {base})'
        # if expr.exp == 0.5:
        #     return f'sqrt({base})'
        # if expr.exp == 3/2:
        #     return f'({base} * sqrt({base}))'
        # if expr.exp == 2:
        #     return f'({base} * {base})'
        # if expr.exp == 5/2:
        #     return f'({base} * {base} * sqrt({base}))'
        # if expr.exp == 3:
        #     return f'({base} * {base} * {base})'
        # else:
        return f'std::pow({base}, {exp})'

    def _print_not_supported(self, expr):
        print(f'Not supported: {expr}')

class EigenFunctionInputClosure:
    def __init__(self, printer : EigenPrinter, option_dict : dict, *args : EigenMatrix):
        Vars = [ var for var in args]
        for i in range(len(Vars)):
            if not hasattr(Vars[i], 'IS_EIGEN_MATRIX'):
                Vars[i] = Eigen.FromSympy(Vars[i].name, Vars[i])
                
        self.Args = Vars
        self.printer = printer
        self.option_dict = option_dict
        
    def __call__(self, FunctionName : str, expr : Expr, return_value_name:str = 'R'):
        # if expr has no shape attribute, it is a scalar
        if(not hasattr(expr, 'shape')):
            R = Eigen.Scalar(return_value_name)
        else:
            R = Eigen.Matrix(return_value_name, expr.shape[0], expr.shape[1])
            
        FunctionDef = self._make_function_def(FunctionName, R, expr)
        MaxLineLen = 120
        # MaxLineLen = len(FunctionDef)
        
        Comment = self._make_comment(R, expr)
        CommentStr = '\n'.join(Comment)
        # MaxLineLen = max([len(line) for line in Comment] + [MaxLineLen])
        
        Content = self._make_content(R, expr)
        ContentStr = '\n'.join(Content)
        # MaxLineLen = max([len(line) for line in Content] + [MaxLineLen])
        
        SepLine = '*' * (MaxLineLen + 4)
        
        return f'''{FunctionDef}
{{
/*{SepLine}
Function generated by SymEigen.py 
Author: {AuthorName}
GitHub: {AuthorGitHub}
E-Mail: {AuthorEmail}
**{SepLine}
{CommentStr}
{SepLine}*/
{ContentStr}
}}'''

    def _make_function_def(self, FunctionName, R, expr): 
        Vars = self.Args
               
        T = 'T'
        OutputT = R.RefType(T)
        
        Args = []
        for i in range(len(Vars)):
            Args.append(Vars[i].CRefType(T) + f' {Vars[i].MatrixName()}')
        
        ArgStr = ', '.join(Args)
        
        FOWARD_MACRO = self.option_dict['MacroBeforeFunction']
        if len(FOWARD_MACRO) > 0:
            FOWARD_MACRO = f'{FOWARD_MACRO} '
        
        return f'''template <typename T>
{FOWARD_MACRO}void {FunctionName}({OutputT} {R.MatrixName()}, {ArgStr})'''

    def _make_comment(self, R, expr):
        Vars = self.Args
        Comment = []
        if self.option_dict['LatexComment']:
            Comment.append(f'''LaTeX expression:
//tex:$${R.MatrixName()} = {latex(expr)}$$\n''')
        
        Comment.append(f'''Symbol Name Mapping:''')
        for var in Vars:
            Comment.append(f'''{var.name}:
    -> {var.to_origin_element_name}
    -> {var}''')
        return Comment

    
    def _make_content(self, R, expr):
        Vars = self.Args
        Content = []
        
        if self.option_dict['CommonSubExpression']:
            sub_exprs, simplified = cse(expr)
            
            Content.append('/* Sub Exprs */')
            for i in range(len(sub_exprs)):
                E = sub_exprs[i][1]
                EStr = self.printer._print(E)
                EStr = self._replace_symbol(EStr, Vars)
                Content.append(f'auto {sub_exprs[i][0]} = {EStr};')
            
            Content.append('/* Simplified Expr */')
            for S in simplified:
                # S has shape
                for i in range(R.shape[0]):
                    for j in range(R.shape[1]):
                        if S.is_Matrix:
                            E = S[i,j]
                        else:
                            E = S
                        EStr = self.printer._print(E)
                        EStr = self._replace_symbol(EStr, Vars)
                        Content.append(f'{R.At(i, j)} = {EStr};')
                
        else:
            for i in range(R.shape[0]):
                for j in range(R.shape[1]):
                    if(hasattr(expr, 'shape')):
                        E = expr[i, j]
                    else:
                        E = expr
                    EStr = self.printer._print(E)
                    EStr = self._replace_symbol(EStr, Vars)
                    Content.append(f'{R.At(i, j)} = {EStr};')
        return Content
    
    def _replace_symbol(self, Str, Vars):
        for var in Vars:
            for key, value in var.from_origin_element_name.items():
                Str = Str.replace(key, value)
        return Str
    

class EigenFunctionGenerator:
    def __init__(self, printer = EigenPrinter()):
        self.printer = printer
        self.option_dict = {
            'MacroBeforeFunction': '',
            'CommonSubExpression': True,
            'LatexComment': True
        }
    
    def MacroBeforeFunction(self, macro: str):
        self.option_dict['MacroBeforeFunction'] = macro
    
    def DisableCommonSubExpression(self):
        self.option_dict['CommonSubExpression'] = False
    
    def DisableLatexComment(self):
        self.option_dict['LatexComment'] = False

    def Closure(self, *args : EigenMatrix):
        return EigenFunctionInputClosure(self.printer, self.option_dict, *args)
    
    def EigenVectorizeCode(self):
        return '''template <typename T, int M, int N>
void Vectorize(Eigen::Vector<T, M * N>& Vec, const Eigen::Matrix<T, M, N>& Mat)
{
/******************************************************************************
Function generated by SymEigen.py 
Author: %s
GitHub: %s
E-Mail: %s
******************************************************************************/
    static_assert(M > 0 && N > 0, "Invalid Input Matrix");
    for(int j = 0; j < N; ++j)
        Vec.template segment<M>(M * j) = Mat.template block<M,1>(0, j);
}
''' % (AuthorName, AuthorGitHub, AuthorEmail)

    