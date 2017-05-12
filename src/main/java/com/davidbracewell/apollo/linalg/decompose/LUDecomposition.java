package com.davidbracewell.apollo.linalg.decompose;

import com.davidbracewell.apollo.linalg.DenseMatrix;
import com.davidbracewell.apollo.linalg.Matrix;
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
