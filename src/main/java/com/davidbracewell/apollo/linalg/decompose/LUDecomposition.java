package com.davidbracewell.apollo.linalg.decompose;

import com.davidbracewell.apollo.linalg.DenseMatrix;
import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.linalg.SparseMatrix;
import com.davidbracewell.conversion.Cast;
import lombok.NonNull;
import org.jblas.Decompose;
import org.jblas.DoubleMatrix;

/**
 * The type LU decomposition.
 *
 * @author David B. Bracewell
 */
public class LUDecomposition implements Decomposition {
    @Override
    public Matrix[] decompose(@NonNull Matrix input) {
        if (input instanceof SparseMatrix) {
            org.apache.commons.math3.linear.LUDecomposition lu = new org.apache.commons.math3.linear.LUDecomposition(Cast.<SparseMatrix>as(
                input).asRealMatrix());
            return new Matrix[]{
                new SparseMatrix(lu.getL()),
                new SparseMatrix(lu.getP()),
                new SparseMatrix(lu.getU()),
            };
        }

        Decompose.LUDecomposition<DoubleMatrix> r = Decompose.lu(input
                                                                     .toDense()
                                                                     .asDoubleMatrix());
        return new Matrix[]{
            new DenseMatrix(r.l),
            new DenseMatrix(r.p),
            new DenseMatrix(r.u)
        };
    }
}// END OF LUDecomposition
