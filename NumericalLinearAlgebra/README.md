# NumericalLinearAlgebra
Direct methods in optimization with contstraints:

The goal of this project is to investigate some of the basic numerical linear algebra ideas behind
optimization problems. For simplicity we consider convex optimization problems. Concretely,
we look for x ∈ R n that solves

                              minimize f (x) = x T Gx + g T x
                              subject to A T x = b, C T x ≥ d,

where G ∈ Rn×n is symmetric semidefinite positive, g ∈ Rn , A ∈ Rn×p , C ∈ Rn×m , b ∈ Rp and
d ∈ R m .

- ex_c1.py --> file containing the function used in the iterations to compute alpha so that the step-size guarantees feasibility.
- ex_c2_c3_c4.py --> given a n number of dimensions, it allows the user to decide if it wants to compute the computational time that the program will take and the type of factorization it wants to use (to just use an iterative method or to also use LDLt and Cholesky factorization).
- build_variables.py --> functions to create the problem given and the different matrices needed in each case (MKKT matrix, LDLt factorization matrix and Cholesky facotrization matrix).
- solver_functions.py --> solves the problem depending on which case we are using.
