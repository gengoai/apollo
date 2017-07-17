package com.davidbracewell.apollo.linalg;

import com.davidbracewell.apollo.linalg.decompose.TruncatedSVD;
import lombok.NonNull;

import java.util.Arrays;
import java.util.Collection;

/**
 * <p>Common vector compositions</p>
 *
 * @author David B. Bracewell
 */
public enum VectorCompositions implements VectorComposition {
   /**
    * Averages the elements of the vectors
    */
   Average {
      @Override
      public Vector compose(int dimension, @NonNull Collection<Vector> vectors) {
         if (vectors.size() == 0) {
            return new SparseVector(dimension);
         }
         Vector rval = new DenseVector(dimension);
         int count = 0;
         for (Vector vector : vectors) {
            if (vector.magnitude() > 0) {
               rval.addSelf(vector);
               count++;
            }
         }
         return rval.mapDivideSelf(count);
      }
   },
   /**
    * Sums the elements of the vectors
    */
   Sum {
      @Override
      public Vector compose(int dimension, @NonNull Collection<Vector> vectors) {
         if (vectors.size() == 0) {
            return new SparseVector(dimension);
         }
         Vector rval = new DenseVector(dimension);
         vectors.forEach(rval::addSelf);
         return rval;
      }
   },
   /**
    * Performs a point-wise multiply of the elements
    */
   PointWiseMultiply {
      @Override
      public Vector compose(int dimension, @NonNull Collection<Vector> vectors) {
         if (vectors.size() == 0) {
            return new SparseVector(dimension);
         }
         Vector rval = DenseVector.ones(dimension);
         vectors.stream().filter(vector -> vector.magnitude() > 0).forEach(rval::multiplySelf);
         return rval;
      }
   },
   /**
    * Assigns the maximum element
    */
   Max {
      @Override
      public Vector compose(int dimension, @NonNull Collection<Vector> vectors) {
         if (vectors.size() == 0) {
            return new SparseVector(dimension);
         }
         double[] values = new double[dimension];
         Arrays.fill(values, Double.NEGATIVE_INFINITY);
         Vector rval = new DenseVector(values);
         vectors.stream().filter(vector -> vector.magnitude() > 0)
                .forEach(v -> v.forEach(e -> {
                            if (e.getValue() > rval.get(e.getIndex())) {
                               rval.set(e.getIndex(), e.getValue());
                            }
                         })
                        );
         return rval;
      }
   },
   /**
    * Assigns the minimum element
    */
   Min {
      @Override
      public Vector compose(int dimension, @NonNull Collection<Vector> vectors) {
         if (vectors.size() == 0) {
            return new SparseVector(dimension);
         }
         double[] values = new double[dimension];
         Arrays.fill(values, Double.POSITIVE_INFINITY);
         Vector rval = new DenseVector(values);
         vectors.stream().filter(vector -> vector.magnitude() > 0)
                .forEach(v -> v.forEach(e -> {
                            if (e.getValue() < rval.get(e.getIndex())) {
                               rval.set(e.getIndex(), e.getValue());
                            }
                         })
                        );
         return rval;
      }
   },
   SVD {
      @Override
      public Vector compose(int dimension, @NonNull Collection<Vector> vectors) {
         if (vectors.size() == 0) {
            return new SparseVector(dimension);
         }
         DenseMatrix matrix = new DenseMatrix(vectors.size(), dimension);
         int r = 0;
         for (Vector vector : vectors) {
            matrix.setRow(r, vector);
            r++;
         }
         TruncatedSVD svd = new TruncatedSVD(1);
         return svd.decompose(matrix)[2].row(0).copy();
      }
   }

}//END OF VectorCompositions
