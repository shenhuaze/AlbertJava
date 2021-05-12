package com.huaze.shen;

/**
 * @author shenhuaze
 * @date 2021-01-27
 *
 * 四维矩阵
 */
public class Matrix4D {
    private double[][][][] A;

    private int m, n, p, q;

    public Matrix4D(int m, int n, int p, int q) {
        this.m = m;
        this.n = n;
        this.p = p;
        this.q = q;
        A = new double[m][n][p][q];
    }

    public Matrix4D(int m, int n, int p, int q, double s) {
        this.m = m;
        this.n = n;
        this.p = p;
        this.q = q;
        A = new double[m][n][p][q];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < p; k++) {
                    for (int l = 0; l < q; l++) {
                        A[i][j][k][l] = s;
                    }
                }
            }
        }
    }

    public Matrix4D(double[][][][] A) {
        m = A.length;
        n = A[0].length;
        p = A[0][0].length;
        q = A[0][0][0].length;
        for (int i = 0; i < m; i++) {
            if (A[i].length != n) {
                throw new IllegalArgumentException("All rows must have the same length.");
            }
            for (int j = 0; j < n; j++) {
                if (A[i][j].length != p) {
                    throw new IllegalArgumentException("All rows must have the same length.");
                }
                for (int k = 0; k < p; k++) {
                    if (A[i][j][k].length != q) {
                        throw new IllegalArgumentException("All rows must have the same length.");
                    }
                }
            }
        }
        this.A = A;
    }

    public Matrix4D(double[][][][] A, int m, int n, int p, int q) {
        this.A = A;
        this.m = m;
        this.n = n;
        this.p = p;
        this.q = q;
    }

    public double[][][][] getArray() {
        return A;
    }

    public Matrix4D transpose_0213() {
        Matrix4D result = new Matrix4D(m, p, n, q);
        double[][][][] resultArray = result.getArray();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < p; k++) {
                    resultArray[i][k][j] = A[i][j][k];
                }
            }
        }
        return result;
    }

    /** Multiply a matrix by a scalar, C = s*A
     @param s    scalar
     @return     s*A
     */
    public Matrix4D times(double s) {
        Matrix4D X = new Matrix4D(m, n, p, q);
        double[][][][] C = X.getArray();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < p; k++) {
                    for (int l = 0; l < q; l++) {
                        C[i][j][k][l] = s * A[i][j][k][l];
                    }
                }
            }
        }
        return X;
    }

    /** Add a matrix by a scalar, C = s*A
     @param s    scalar
     @return     s*A
     */
    public Matrix4D plus(double s) {
        Matrix4D X = new Matrix4D(m, n, p, q);
        double[][][][] C = X.getArray();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < p; k++) {
                    for (int l = 0; l < q; l++) {
                        C[i][j][k][l] = s + A[i][j][k][l];
                    }
                }
            }
        }
        return X;
    }

    /** C = A + B
     @param B    another matrix
     @return     A + B
     */
    public Matrix4D plus(Matrix4D B) {
        Matrix4D X = new Matrix4D(m,n,p,q);
        double[][][][] C = X.getArray();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < p; k++) {
                    for (int l = 0; l < q; l++) {
                        C[i][j][k][l] = A[i][j][k][l] + B.A[i][j][k][l];
                    }
                }
            }
        }
        return X;
    }

    public Matrix4D softmax() {
        Matrix4D X = new Matrix4D(m, n, p, q);
        double[][][][] C = X.getArray();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < p; k++) {
                    C[i][j][k] = Activation.softmax(A[i][j][k]);
                }
            }
        }
        return X;
    }

    public Matrix reshapeTo2D(int a, int b) {
        double[][] result = new double[a][b];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < p; k++) {
                    for (int l = 0; l < q; l++) {
                        int globalId = i * n * p * q + j * p * q + k * q + l;

                        int groupId1 = globalId / b;
                        int remainder1 = globalId % b;

                        result[groupId1][remainder1] = A[i][j][k][l];
                    }
                }
            }
        }
        return new Matrix(result);
    }
}
