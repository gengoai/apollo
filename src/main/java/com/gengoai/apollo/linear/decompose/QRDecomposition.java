package com.gengoai.apollo.linear.decompose;

import com.gengoai.apollo.linear.DenseNDArray;
import com.gengoai.apollo.linear.NDArray;
import org.jblas.Decompose;
import org.jblas.FloatMatrix;

import java.io.Serializable;

/**
 * <p>Performs <a href="https://en.wikipedia.org/wiki/QR_decomposition">QR Decomposition</a> on the given input
 * NDArray. The returned array is in order {Q,R}</p>
 *
 * @author David B. Bracewell
 */
public class QRDecomposition implements Decomposition, Serializable {
   private static final long serialVersionUID = 1L;

   @Override
   public NDArray[] decompose(NDArray m) {
      Decompose.QRDecomposition<FloatMatrix> r = Decompose.qr(m.toFloatMatrix());
      return new NDArray[]{new DenseNDArray(r.q), new DenseNDArray(r.r)};
   }

}// END OF QRDecomposition
