package com.davidbracewell.apollo.linear;

import com.davidbracewell.apollo.linear.dense.DenseDoubleNDArray;
import com.davidbracewell.guava.common.base.Preconditions;
import lombok.NonNull;
import org.jblas.DoubleMatrix;
import org.jblas.Singular;

/**
 * The type Svd.
 *
 * @author David B. Bracewell
 */
public final class SVD {

   private SVD() {
      throw new IllegalAccessError();
   }

   /**
    * Svd nd array [ ].
    *
    * @param matrix the matrix
    * @return the nd array [ ]
    */
   public static NDArray[] svd(@NonNull NDArray matrix) {
      return toNDArray(svd(matrix.toDoubleMatrix(), matrix.isSparse()));
   }

   /**
    * Svd double matrix [ ].
    *
    * @param matrix the matrix
    * @param sparse the sparse
    * @return the double matrix [ ]
    */
   private static DoubleMatrix[] svd(@NonNull DoubleMatrix matrix, boolean sparse) {
      if (sparse) {
         return Singular.sparseSVD(matrix);
      }
      return Singular.fullSVD(matrix);
   }

   private static NDArray[] toNDArray(DoubleMatrix[] matrices) {
      NDArray[] toReturn = new NDArray[matrices.length];
      for (int i = 0; i < matrices.length; i++) {
         toReturn[i] = new DenseDoubleNDArray(matrices[i]);
      }
      return toReturn;
   }

   /**
    * Truncated svd double matrix [ ].
    *
    * @param matrix the matrix
    * @param K      the k
    * @param sparse the sparse
    * @return the double matrix [ ]
    */
   private static DoubleMatrix[] truncatedSVD(@NonNull DoubleMatrix matrix, int K, boolean sparse) {
      Preconditions.checkArgument(K > 0, "Number of singular values must be greater than zero.");
      DoubleMatrix[] result = svd(matrix, sparse);
      DoubleMatrix u = result[0].getRange(0, result[0].getRows(), 0, K);
      DoubleMatrix s = DoubleMatrix.diag(result[1].getRange(0, K));
      DoubleMatrix v = result[2].getRange(0, result[2].getRows(), 0, K);
      return new DoubleMatrix[]{u, s, v};
   }

   /**
    * Truncated svd nd array [ ].
    *
    * @param matrix the matrix
    * @param K      the k
    * @return the nd array [ ]
    */
   public static NDArray[] truncatedSVD(@NonNull NDArray matrix, int K) {
      return toNDArray(truncatedSVD(matrix.toDoubleMatrix(), K, matrix.isSparse()));
   }
}// END OF SVD
