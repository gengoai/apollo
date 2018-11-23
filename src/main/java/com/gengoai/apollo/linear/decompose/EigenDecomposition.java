package com.gengoai.apollo.linear.decompose;

import com.gengoai.apollo.linear.DenseNDArray;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.linear.RealMatrixWrapper;
import org.jblas.ComplexFloatMatrix;
import org.jblas.Eigen;

import java.io.Serializable;

/**
 * <p>Performs <a href="https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix">Eigen Decomposition</a> on the
 * given input NDArray. The returned array is in order {V,D}</p>
 *
 * @author David B. Bracewell
 */
public class EigenDecomposition implements Decomposition, Serializable {
   private static final long serialVersionUID = 1L;

   @Override
   public NDArray[] decompose(NDArray input) {
      if (input instanceof DenseNDArray) {
         ComplexFloatMatrix[] r = Eigen.eigenvectors(input.toFloatMatrix());
         NDArray[] toReturn = new NDArray[r.length];
         for (int i = 0; i < r.length; i++) {
            toReturn[i] = new DenseNDArray(r[i].getReal());
         }
         return toReturn;
      }
      org.apache.commons.math3.linear.EigenDecomposition decomposition =
         new org.apache.commons.math3.linear.EigenDecomposition(new RealMatrixWrapper(input));
      return new NDArray[]{
         NDArrayFactory.matrix(decomposition.getV().getData()),
         NDArrayFactory.matrix(decomposition.getD().getData())
      };
   }
}// END OF EigenDecomposition
