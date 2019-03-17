package com.gengoai.apollo.linear.decompose;

import com.gengoai.Validation;
import com.gengoai.apollo.linear.DenseMatrix;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.RealMatrixWrapper;
import org.jblas.Decompose;
import org.jblas.DoubleMatrix;

import static com.gengoai.apollo.linear.NDArrayFactory.ND;

/**
 * <p>Performs <a href="https://en.wikipedia.org/wiki/LU_decomposition">LU Decomposition</a> on the
 * given input NDArray. The returned array is in order {L, U, P}</p>
 *
 * @author David B. Bracewell
 */
public class LUDecomposition extends Decomposition {
   private static final long serialVersionUID = 1L;

   public LUDecomposition() {
      super(3);
   }

   @Override
   protected NDArray[] onMatrix(NDArray m) {
      Validation.checkArgument(m.shape().isSquare(), "Only square matrices are supported");
      if (m instanceof DenseMatrix) {
         Decompose.LUDecomposition<DoubleMatrix> r = Decompose.lu(m.toDoubleMatrix()[0]);
         return new NDArray[]{new DenseMatrix(r.l),
            new DenseMatrix(r.u),
            new DenseMatrix(r.p)};
      }
      org.apache.commons.math3.linear.LUDecomposition luDecomposition =
         new org.apache.commons.math3.linear.LUDecomposition(new RealMatrixWrapper(m));
      return new NDArray[]{
         ND.array(luDecomposition.getL().getData()),
         ND.array(luDecomposition.getU().getData()),
         ND.array(luDecomposition.getP().getData()),
      };
   }

}// END OF LUDecomposition
