package com.davidbracewell.apollo.linalg.decompose;

import com.davidbracewell.apollo.linalg.DenseMatrix;
import com.davidbracewell.apollo.linalg.Matrix;
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
        Decompose.QRDecomposition<DoubleMatrix> r = Decompose.qr(input
                                                                     .toDense()
                                                                     .asDoubleMatrix());
        return new Matrix[]{
            new DenseMatrix(r.q),
            new DenseMatrix(r.r)
        };
    }
}// END OF QRDecomposition
