package com.davidbracewell.apollo.linalg.decompose;

import com.davidbracewell.apollo.linalg.DenseMatrix;
import com.davidbracewell.apollo.linalg.Matrix;
import lombok.Getter;
import lombok.NonNull;
import org.jblas.DoubleMatrix;
import org.jblas.Singular;

import java.io.Serializable;

/**
 * <a href="https://en.wikipedia.org/wiki/Singular_value_decomposition">Singular Value decomposition</a> for matrices,
 * which is a generalization of <code>Eigen decompositions</code>.
 *
 * @author David B. Bracewell
 */
public class SingularValueDecomposition implements Decomposition, Serializable {
   private static final long serialVersionUID = 1L;
   @Getter
   private final boolean sparse;

   /**
    * Instantiates a new Singular value decomposition using a <code>Full</code> svd method.
    */
   public SingularValueDecomposition() {
      this(false);
   }

   /**
    * Instantiates a new Singular value decomposition.
    *
    * @param sparse True use JBlas's <code>sparseSVD</code>, False use JBlas's <code>fullSVD</code>
    */
   public SingularValueDecomposition(boolean sparse) {
      this.sparse = sparse;
   }

   @NonNull
   public Matrix[] decompose(@NonNull Matrix m) {
      DoubleMatrix[] result = sparse ?
                              Singular.sparseSVD(m.toDense().asDoubleMatrix()) :
                              Singular.fullSVD(m.toDense().asDoubleMatrix());
      return new DenseMatrix[]{
         new DenseMatrix(result[0]),
         new DenseMatrix(DoubleMatrix.diag(result[1], m.numberOfRows(), m.numberOfColumns())),
         new DenseMatrix(result[2])
      };
   }

}// END OF SingularValueDecomposition
