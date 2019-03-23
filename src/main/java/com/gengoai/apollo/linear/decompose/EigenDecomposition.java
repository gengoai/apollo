package com.gengoai.apollo.linear.decompose;

import com.gengoai.apollo.linear.DenseMatrix;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.RealMatrixWrapper;
import org.jblas.ComplexDoubleMatrix;
import org.jblas.DoubleMatrix;
import org.jblas.Eigen;

import static com.gengoai.apollo.linear.NDArrayFactory.ND;

/**
 * <p>Performs <a href="https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix">Eigen Decomposition</a> on the
 * given input NDArray. The returned array is in order {V,D}</p>
 *
 * @author David B. Bracewell
 */
public class EigenDecomposition extends Decomposition {
   private static final long serialVersionUID = 1L;

   public EigenDecomposition() {
      super(2);
   }

   @Override
   protected NDArray[] onMatrix(NDArray input) {
      if (input.isDense()) {
         DoubleMatrix slice = input.toDoubleMatrix()[0];
         ComplexDoubleMatrix[] result = Eigen.eigenvectors(slice);
         return new NDArray[]{
            new DenseMatrix(result[0].getReal()),
            new DenseMatrix(result[1].getReal())
         };
      } else {
         org.apache.commons.math3.linear.EigenDecomposition decomposition =
            new org.apache.commons.math3.linear.EigenDecomposition(new RealMatrixWrapper(input));
         return new NDArray[]{
            ND.array(decomposition.getV().getData()),
            ND.array(decomposition.getD().getData())
         };
      }
   }

}// END OF EigenDecomposition
