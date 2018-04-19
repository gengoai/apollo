package com.gengoai.apollo.linear.decompose;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.dense.DenseDoubleNDArray;
import com.gengoai.apollo.linear.dense.DenseFloatNDArray;
import org.jblas.Decompose;
import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class QRDecomposition implements Decomposition, Serializable {
   private static final long serialVersionUID = 1L;

   @Override
   public NDArray[] decompose(NDArray m) {
      if (m instanceof DenseFloatNDArray) {
         Decompose.QRDecomposition<FloatMatrix> r = Decompose.qr(m.toFloatMatrix());
         return new NDArray[]{new DenseFloatNDArray(r.q), new DenseFloatNDArray(r.r)};
      } else {
         Decompose.QRDecomposition<DoubleMatrix> r = Decompose.qr(m.toDoubleMatrix());
         return new NDArray[]{new DenseDoubleNDArray(r.q), new DenseDoubleNDArray(r.r)};
      }
   }
}// END OF QRDecomposition
