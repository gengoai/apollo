package com.gengoai.apollo.linear.decompose;

import com.gengoai.Validation;
import com.gengoai.apollo.linear.DenseMatrix;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import org.jblas.Decompose;
import org.jblas.DoubleMatrix;

import java.io.Serializable;

/**
 * <p>Performs <a href="https://en.wikipedia.org/wiki/LU_decomposition">LU Decomposition</a> on the
 * given input NDArray. The returned array is in order {L, U, P}</p>
 *
 * @author David B. Bracewell
 */
public class LUDecomposition implements Decomposition, Serializable {
   private static final long serialVersionUID = 1L;

   public NDArray[] decompose(NDArray m) {
      Validation.checkArgument(m.shape().isSquare(), "Only square matrices are supported");
      if (m instanceof DenseMatrix) {
         Decompose.LUDecomposition<DoubleMatrix> r = Decompose.lu(m.toDoubleMatrix()[0]);
         return new NDArray[]{new DenseMatrix(r.l),
            new DenseMatrix(r.u),
            new DenseMatrix(r.p)};
      }

      int nr = m.rows();
      NDArray L = NDArrayFactory.ND.array(nr, nr);
      NDArray U = NDArrayFactory.ND.array(nr, nr);
      NDArray P = m.pivot();
      NDArray A2 = P.mmul(m);

      for (int j = 0; j < nr; j++) {
         L.set(j, j, 1d);
         for (int i = 0; i < j + 1; i++) {
            double s1 = 0d;
            for (int k = 0; k < i; k++) {
               s1 += U.get(k, j) * L.get(i, k);
            }
            U.set(i, j, (A2.get(i, j) - s1));
         }
         for (int i = j; i < nr; i++) {
            double s2 = 0d;
            for (int k = 0; k < j; k++) {
               s2 += U.get(k, j) * L.get(i, k);
            }
            L.set(i, j, (A2.get(i, j) - s2) / U.get(j, j));
         }
      }
      return new NDArray[]{L, U, P};
   }


}// END OF LUDecomposition
