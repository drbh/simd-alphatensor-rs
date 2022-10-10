import numpy as np
from ast import literal_eval as make_tuple

np.random.seed(1)

"""
The *.npz files contain a dict with keys like "(2,3,4)" and values containing
a list of matrices U, V and W. For example, for the 2-by-2 times 2-by-2 case,
we have the following matrices:

U =
[[ 0  1  1  0  1  1  0]
 [ 0  0 -1  1  0  0  0]
 [ 1  1  1  0  1  0  0]
 [-1 -1 -1  0  0  0  1]]

V =
[[0 0 0 0 1 1 0]
 [1 1 0 0 1 0 1]
 [0 1 1 1 1 0 0]
 [0 1 1 0 1 0 1]]

W =
[[ 0  0  0  1  0  1  0]
 [ 0 -1  0  0  1 -1 -1]
 [-1  1 -1 -1  0  0  0]
 [ 1  0  0  0  0  0  1]]

Each column of U is multiplied with the vectorized matrix A.
Likewise, Each column of V is multiplied with the vectorized matrix B.
The resulting vectors are multiplied pointwise and their product is
multiplied with W, which forms the entries of the product matrix C = A times B.
Also see the function `multiply` below.
"""

# There are two factorizations, one for useful numbers and one for mod 2 math.
for filename, mod in [
    ("data/factorizations_r.npz", None),
    # ("factorizations_f2.npz", 2),
]:
    # Load the factorizations. Note that allow_pickle=True allows arbitrary
    # code execution. A JSON file would have been a better format choice
    # since nothing here is stored in NumPy format anyway.
    factorizations = dict(np.load(filename, allow_pickle=True))

    # Test each factorization
    for key, UVW in factorizations.items():
        U, V, W = map(np.array, UVW)

        m, k, n = make_tuple(key)

        print(f"\nMultiply {m}-by-{k} matrix A with {k}-by-{n} matrix B")
        if mod is not None:
            print(f"using mod {mod} arithmetic")
        print()

        # Check that shapes are correct
        assert m * k == U.shape[0]
        assert k * n == V.shape[0]
        assert m * n == W.shape[0]
        assert U.shape[1] == V.shape[1]
        assert U.shape[1] == W.shape[1]

        # Generate two random matrices for testing
        A = np.random.randint(10, size=(m, k)) * 10_000
        B = np.random.randint(10, size=(k, n)) * 10_000

        def multiply(A, B, U, V, W):
            # Multiply two matrices A and B using index matrices U, V and W
            a = A.ravel()
            b = B.ravel()

            tmp = (U.T @ a) * (V.T @ b)
            c = W @ tmp
            C = c.reshape(n, m).T

            return C

        # Multiply matrices
        C = multiply(A, B, U, V, W)

        # Check that result is correct, taking potential mod 2 into account
        if mod is None:
            assert np.allclose(C, A @ B)
        else:
            assert np.allclose(C % mod, (A @ B) % mod)

        def make_code(variables, factors):
            # Generate code like "(a11 + a21 - a22)"
            parts = []

            for variable, factor in zip(variables, factors):
                flag = False
                # Simplify +1 and -1 factors
                if factor == 1:
                    factor = " + "
                elif factor == -1:
                    factor = " - "
                elif factor == 0:
                    continue
                else:
                    flag = True
                    abs_factor = float(abs(factor))

                    res = None
                    if abs_factor == 0.125:
                        res = f"((({variable} >> 1) >> 1) >> 1)" # == / 8
                    if abs_factor == 0.5:
                        res = f"({variable} >> 1)" # == / 2
                    elif abs_factor == 2.0:
                        res = f"({variable} << 1)" # == * 2
                    elif abs_factor == 3.0:
                        res = f"(({variable} << 1 ) + {variable})" # == * 3
                    else:
                        pass
                    if factor < 0:
                        factor = f" - {res}"
                    else:
                        factor = f" + {res}"
                if flag:
                    parts.append(factor)
                else:
                    parts.append(factor + variable)
            
            code = "".join(parts).lstrip(" +")

            if len(parts) > 1:
                code = "(" + code + ")"

            return code

        def make_variables(var, m, n):
            # Generate variables like a11, a12, a21, a22
            # or maybe a_1_1, a_1_2, a_2_1, a_2_2.
            # For larger matrices, we need a separator to avoid
            # confusing e.g. a_1_11 with a_11_1.
            separator = "_" if max(m, n, k) > 9 else ""
            return [f"{var}{separator}{i + 1}{separator}{j + 1}"
                for i in range(m) for j in range(n)]

        A_variables = make_variables("a", m, k)
        B_variables = make_variables("b", k, n)
        C_variables = make_variables("c", m, n)
        h_variables = [f"h{i + 1}" for i in range(U.shape[1])]

        lines = [
            ", ".join(A_variables) + " = A.ravel()",
            ", ".join(B_variables) + " = B.ravel()",
        ]

        # Generate code for computation of temporary vector
        for h, u, v in zip(h_variables, U.T, V.T):
            sa = make_code(A_variables, u).replace("()", "0")
            sb = make_code(B_variables, v).replace("()", "0")

            lines.append(f"{h} = {sa} * {sb}")

        # Generate code for computation
        for c, w in zip(C_variables, W):
            lines.append(f"{c} = " + 
                make_code(h_variables, w).replace("()", "0")
            )

        lines.append("C = np.array([" + ", ".join(C_variables) +
            f"]).reshape({n}, {m}).T.astype(int)")

        code = "\n".join(lines)

        print(code)

        # Verify that code generates the correct result
        exec(code)

        # issue with
        # Multiply 3-by-4 matrix A with 4-by-11 matrix B
        # 16 c_2_6 93 99 -6
        # 17 c_2_7 97 105 -8
        # 22 c_3_1 72 78 -6
        # (b_4_6 * 0.5) != (b_4_6 >> 1)
        # if b_4_6 == 3
        # 1.5 !=  1
        if mod is None:
            assert np.allclose(C, A @ B)
        else:
            assert np.allclose(C % mod, (A @ B) % mod)