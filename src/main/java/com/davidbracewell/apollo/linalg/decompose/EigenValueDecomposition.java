package com.davidbracewell.apollo.linalg.decompose;

import com.davidbracewell.apollo.linalg.DenseMatrix;
import com.davidbracewell.apollo.linalg.Matrix;
import lombok.NonNull;
import org.jblas.ComplexDoubleMatrix;

import java.io.Serializable;

/**
 * <a href="https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix">Eigen decomposition</a>, which is also
 * commonly refereed to as <code>Spectral decomposition</code>.
 *
 * @author David B. Bracewell
 */
public class EigenValueDecomposition implements Decomposition, Serializable {
   private static final long serialVersionUID = 1L;

   @Override
   public Matrix[] decompose(@NonNull Matrix m) {
      ComplexDoubleMatrix result = org.jblas.Eigen.eigenvalues(m.toDense().asDoubleMatrix());
      return new DenseMatrix[]{new DenseMatrix(result.toArray2())};
   }

}// END OF EigenValueDecomposition
