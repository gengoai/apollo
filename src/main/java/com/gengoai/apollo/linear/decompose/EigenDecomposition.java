package com.gengoai.apollo.linear.decompose;

import com.gengoai.apollo.linear.*;
import org.jblas.ComplexDoubleMatrix;
import org.jblas.DoubleMatrix;
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
      NDArray[][] toReturn = new NDArray[2][input.shape().sliceLength];
      for (int i = 0; i < input.shape().sliceLength; i++) {
         if (input.isDense()) {
            DoubleMatrix slice = input.slice(i).toDoubleMatrix()[0];
            ComplexDoubleMatrix[] result = Eigen.eigenvectors(slice);
            for (int j = 0; j < result.length; j++) {
               toReturn[j][i] = new DenseMatrix(result[j].getReal());
            }
         } else {
            org.apache.commons.math3.linear.EigenDecomposition decomposition =
               new org.apache.commons.math3.linear.EigenDecomposition(new RealMatrixWrapper(input.slice(i)));
            toReturn[0][i] = NDArrayFactory.ND.array(decomposition.getV().getData());
            toReturn[1][i] = NDArrayFactory.ND.array(decomposition.getD().getData());
         }
      }
      return new Tensor[]{
         new Tensor(input.kernels(),
                    input.channels(),
                    toReturn[0]),
         new Tensor(input.kernels(),
                    input.channels(),
                    toReturn[1])
      };
   }

   public static void main(String[] args) throws Exception {
      NDArray nd = NDArrayFactory.DENSE.rand(2, 2, 100, 100);
      EigenDecomposition decomposition = new EigenDecomposition();
      System.out.println(decomposition.decompose(nd)[0]);
   }

}// END OF EigenDecomposition
