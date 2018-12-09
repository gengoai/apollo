package com.gengoai.apollo.linear.decompose;

import com.gengoai.apollo.linear.DenseNDArray;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.SparkLinearAlgebra;
import org.jblas.FloatMatrix;
import org.jblas.Singular;

import java.io.Serializable;

/**
 * <p>Performs <a href="https://en.wikipedia.org/wiki/Singular_value_decomposition">Singular Value Decomposition</a> on
 * the given input NDArray. The returned array is in order {U, S, V}</p>
 *
 *
 * @author David B. Bracewell
 */
public class SingularValueDecomposition implements Decomposition, Serializable {
   private static final long serialVersionUID = 1L;

   private final boolean distributed;
   private final boolean sparse;
   private final int K;

   /**
    * Instantiates a new Singular value decomposition with K=-1
    */
   public SingularValueDecomposition() {
      this(false, false, -1);
   }

   /**
    * Instantiates a new Singular value decomposition.
    *
    * @param K the number of components to truncate the SVD to
    */
   public SingularValueDecomposition(int K) {
      this(false, false, K);
   }

   /**
    * Instantiates a new Singular value decomposition.
    *
    * @param distributed True - run using Spark in distributed mode, False locally use JBlas
    * @param sparse      True - run using SparseSVD, False full SVD (only used when not distributed).
    */
   public SingularValueDecomposition(boolean distributed, boolean sparse) {
      this(distributed, sparse, -1);
   }

   /**
    * Instantiates a new Singular value decomposition.
    *
    * @param distributed True - run using Spark in distributed mode, False locally use JBlas
    * @param sparse      True - run using SparseSVD, False full SVD (only used when not distributed).
    * @param K           the number of components to truncate the SVD to
    */
   public SingularValueDecomposition(boolean distributed, boolean sparse, int K) {
      this.distributed = distributed;
      this.sparse = sparse;
      this.K = K;
   }

   @Override
   public NDArray[] decompose(NDArray input) {
      if (distributed) {
         return SparkLinearAlgebra.svd(input, K <= 0 ? input.numCols() : K);
      }

      NDArray[] result;
      FloatMatrix[] r;
      if (sparse) {
         r = Singular.sparseSVD(input.toFloatMatrix());
      } else {
         r = Singular.fullSVD(input.toFloatMatrix());
      }
      result = new NDArray[]{
         new DenseNDArray(r[0]),
         new DenseNDArray(FloatMatrix.diag(r[1])),
         new DenseNDArray(r[2]),
      };

      if (K > 0) {
         result[0] = result[0].slice(0, result[0].numRows(), 0, K);
         result[1] = result[1].slice(0, K, 0, K);
         result[2] = result[2].slice(0, result[2].numRows(), 0, K);
      }

      return result;
   }
}// END OF SingularValueDecomposition
