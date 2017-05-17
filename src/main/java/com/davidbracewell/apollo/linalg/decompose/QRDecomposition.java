package com.davidbracewell.apollo.linalg.decompose;

import com.davidbracewell.apollo.linalg.DenseMatrix;
import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.linalg.SparseMatrix;
import com.davidbracewell.conversion.Cast;
import lombok.NonNull;
import org.jblas.Decompose;
import org.jblas.DoubleMatrix;

/**
 * The type Qr decomposition.
 *
 * @author David B. Bracewell
 */
public class QRDecomposition implements Decomposition {
    @Override
    public Matrix[] decompose(@NonNull Matrix input) {
        if (input instanceof SparseMatrix) {
            org.apache.commons.math3.linear.QRDecomposition qr = new org.apache.commons.math3.linear.QRDecomposition(Cast.<SparseMatrix>as(
                input).asRealMatrix());
            return new Matrix[]{
                new SparseMatrix(qr.getQ()),
                new SparseMatrix(qr.getR()),
            };
        }

        Decompose.QRDecomposition<DoubleMatrix> r = Decompose.qr(input
                                                                     .toDense()
                                                                     .asDoubleMatrix());
        return new Matrix[]{
            new DenseMatrix(r.q),
            new DenseMatrix(r.r)
        };
    }
}// END OF QRDecomposition
