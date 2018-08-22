package com.gengoai.apollo.linear.decompose;

import com.gengoai.apollo.linear.DenseNDArray;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.linear.RealMatrixWrapper;
import org.jblas.ComplexFloatMatrix;
import org.jblas.Eigen;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class EigenDecomposition implements Decomposition, Serializable {
   private static final long serialVersionUID = 1L;

   @Override
   public NDArray[] decompose(NDArray input) {
//      if (input instanceof De) {
//         ComplexDoubleMatrix[] r = Eigen.eigenvectors(input.toDoubleMatrix());
//         NDArray[] toReturn = new NDArray[r.length];
//         for (int i = 0; i < r.length; i++) {
//            toReturn[i] = new DenseDoubleNDArray(r[i].getReal());
//         }
//         return toReturn;
//      } else
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
         NDArrayFactory.wrap(decomposition.getV().getData()),
         NDArrayFactory.wrap(decomposition.getD().getData())
      };
   }
}// END OF EigenDecomposition
