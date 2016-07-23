package com.davidbracewell.apollo.linalg.decompose;

import com.davidbracewell.apollo.linalg.DenseMatrix;
import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.conversion.Cast;
import lombok.NonNull;
import org.jblas.ComplexDoubleMatrix;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class EigenValueDecomposition implements Decomposition, Serializable {
  private static final long serialVersionUID = 1L;

  @Override
  public Matrix[] decompose(@NonNull Matrix m) {
    DenseMatrix dense;
    if (m instanceof DenseMatrix) {
      dense = Cast.as(m);
    } else {
      dense = new DenseMatrix(m);
    }
    ComplexDoubleMatrix result = org.jblas.Eigen.eigenvalues(dense.asDoubleMatrix());
    return new DenseMatrix[]{new DenseMatrix(result.toArray2())};
  }

}// END OF EigenValueDecomposition
