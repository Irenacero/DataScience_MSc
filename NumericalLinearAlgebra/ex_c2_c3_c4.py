from build_variables import create_test_problem
from solver_functions import solve_kkt_system
import time

n = int(input('Test problem. Enter dimension n: '))
c_time = str(input('Calculate computation time? (y/n): '))
c_factorizations = str(input('Solve and compare with LDL* and Cholesky fact? (y/n): '))

G_n, x_n, g_n, C_n, lam_0_n, s_0_n, d_n = create_test_problem(n)


if c_time == 'y':
    ini = time.perf_counter()
    sol_n, iterations, g, cond = solve_kkt_system(G=G_n, x_0=x_n, g=g_n, C=C_n, lam_0=lam_0_n, s_0=s_0_n, d=d_n)
    fin = time.perf_counter()

    print("Iterations needed = " + str(iterations) +
          "\nx = " + str(sol_n) +
          "\ng = " + str(g) +
          "\nComputation time " + str(fin - ini) +
          "\nCondition number k2 " + str(cond))

    if c_factorizations == 'y':
        ini = time.perf_counter()
        sol_n, iterations, g, cond = solve_kkt_system(G=G_n, x_0=x_n, g=g_n, C=C_n, lam_0=lam_0_n, s_0=s_0_n, d=d_n,
                                             factorization='ldl')
        fin = time.perf_counter()

        print("Solving the system with LDL* factorization\n Iterations needed = " + str(iterations) +
              "\nx = " + str(sol_n) +
              "\nComputation time " + str(fin - ini) +
          "\nCondition number k2 " + str(cond))

        ini = time.perf_counter()
        sol_n, iterations, g, cond = solve_kkt_system(G=G_n, x_0=x_n, g=g_n, C=C_n, lam_0=lam_0_n, s_0=s_0_n, d=d_n,
                                             factorization='cholesky')
        fin = time.perf_counter()
        
        print("Solving the system with Cholesky factorization\n Iterations needed = " + str(iterations) +
              "\nx = " + str(sol_n) +
              "\nComputation time " + str(fin - ini) +
          "\nCondition number k2 " + str(cond))

elif c_time == 'n':
    sol_n, iterations, g, cond = solve_kkt_system(G=G_n, x_0=x_n, g=g_n, C=C_n, lam_0=lam_0_n, s_0=s_0_n, d=d_n)

    print("Iterations needed = " + str(iterations) +
          "\nx = " + str(sol_n) +
          "\ng = " + str(g) +
          "\nCondition number k2 " + str(cond))
    if c_factorizations == 'y':
        sol_n, iterations, g, cond = solve_kkt_system(G=G_n, x_0=x_n, g=g_n, C=C_n, lam_0=lam_0_n, s_0=s_0_n, d=d_n,
                                             factorization='ldl')

        print("Solving the system with LDL* factorization\n Iterations needed = " + str(iterations) +
              "\nx = " + str(sol_n) +
          "\nCondition number k2 " + str(cond))
        sol_n, iterations, g, cond = solve_kkt_system(G=G_n, x_0=x_n, g=g_n, C=C_n, lam_0=lam_0_n, s_0=s_0_n, d=d_n,
                                             factorization='cholesky')

        print("Solving the system with Cholesky factorization\n Iterations needed = " + str(iterations) +
              "\nx = " + str(sol_n) +
          "\nCondition number k2 " + str(cond))
else:
    print('OOPS, type y or n')
