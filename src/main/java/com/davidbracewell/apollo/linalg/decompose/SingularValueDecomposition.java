package com.davidbracewell.apollo.linalg.decompose;

import com.davidbracewell.apollo.linalg.DenseMatrix;
import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.conversion.Cast;
import lombok.NonNull;
import org.jblas.DoubleMatrix;
import org.jblas.Singular;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class SingularValueDecomposition implements Decomposition, Serializable {
  private static final long serialVersionUID = 1L;

  @NonNull
  public Matrix[] decompose(@NonNull Matrix m) {
    DenseMatrix dense;
    if (m instanceof DenseMatrix) {
      dense = Cast.as(m);
    } else {
      dense = new DenseMatrix(m);
    }
    DoubleMatrix[] result = Singular.sparseSVD(dense.asDoubleMatrix());
    DenseMatrix[] asDense = new DenseMatrix[result.length];
    for (int i = 0; i < result.length; i++) {
      asDense[i] = new DenseMatrix(result[i]);
    }
    return asDense;
  }

}// END OF SingularValueDecomposition
