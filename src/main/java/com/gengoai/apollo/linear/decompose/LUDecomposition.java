package com.gengoai.apollo.linear.decompose;

import com.gengoai.Validation;
import com.gengoai.apollo.linear.DenseNDArray;
import com.gengoai.apollo.linear.NDArray;
import lombok.NonNull;
import org.jblas.Decompose;
import org.jblas.FloatMatrix;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class LUDecomposition implements Decomposition, Serializable {
   private static final long serialVersionUID = 1L;

   public NDArray[] decompose(@NonNull NDArray m) {
      Validation.checkArgument(m.isSquare(), "Only square matrices are supported");
//      if (m instanceof DenseDoubleNDArray) {
//         Decompose.LUDecomposition<DoubleMatrix> r = Decompose.lu(m.toDoubleMatrix());
//         return new NDArray[]{new DenseDoubleNDArray(r.l),
//            new DenseDoubleNDArray(r.u),
//            new DenseDoubleNDArray(r.p)};
//      } else

      if (m instanceof DenseNDArray) {
         Decompose.LUDecomposition<FloatMatrix> r = Decompose.lu(m.toFloatMatrix());
         return new NDArray[]{new DenseNDArray(r.l),
            new DenseNDArray(r.u),
            new DenseNDArray(r.p)};
      }

      int nr = m.numRows();
      NDArray L = m.getFactory().zeros(nr, nr);
      NDArray U = m.getFactory().zeros(nr, nr);
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
