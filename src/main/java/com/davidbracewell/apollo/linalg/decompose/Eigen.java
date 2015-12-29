package com.davidbracewell.apollo.linalg.decompose;

import com.davidbracewell.apollo.linalg.DenseMatrix;
import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.conversion.Cast;
import lombok.NonNull;
import org.jblas.ComplexDoubleMatrix;

/**
 * @author David B. Bracewell
 */
public class Eigen {
  public static void main(String[] args) {
    DenseMatrix matrix = new DenseMatrix(100, 100);
    for (int r = 0; r < 100; r++) {
      for (int c = 0; c < 100; c++) {
        matrix.set(r, c, Math.random());
      }
    }

    Eigen svd = new Eigen();
    System.out.println(svd.decomppse(matrix));

  }

  public Matrix decomppse(@NonNull Matrix m) {
    DenseMatrix dense;
    if (m instanceof DenseMatrix) {
      dense = Cast.as(m);
    } else {
      dense = new DenseMatrix(m);
    }
    ComplexDoubleMatrix result = org.jblas.Eigen.eigenvalues(dense.asDoubleMatrix());
    return new DenseMatrix(result.toArray2());
  }
}// END OF Eigen
