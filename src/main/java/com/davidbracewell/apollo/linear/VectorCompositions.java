package com.davidbracewell.apollo.linear;

import lombok.NonNull;

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
      public NDArray compose(@NonNull Collection<NDArray> vectors) {
         return Sum.compose(vectors).divi(vectors.size());
      }
   },
   /**
    * Sums the elements of the vectors
    */
   Sum {
      @Override
      public NDArray compose(@NonNull Collection<NDArray> vectors) {
         if (vectors.size() == 0) {
            return new EmptyNDArray();
         }
         NDArray toReturn = null;
         for (NDArray v : vectors) {
            if (toReturn == null) {
               toReturn = v.copy();
            } else {
               toReturn.addi(v);
            }
         }
         return toReturn;
      }
   },
   /**
    * Performs a point-wise multiply of the elements
    */
   PointWiseMultiply {
      @Override
      public NDArray compose(@NonNull Collection<NDArray> vectors) {
         if (vectors.size() == 0) {
            return new EmptyNDArray();
         }
         NDArray toReturn = null;
         for (NDArray v : vectors) {
            if (toReturn == null) {
               toReturn = v.copy();
            } else {
               toReturn.mul(v);
            }
         }
         return toReturn;
      }
   },
   /**
    * Assigns the maximum element
    */
   Max {
      @Override
      public NDArray compose(@NonNull Collection<NDArray> vectors) {
         if (vectors.size() == 0) {
            return new EmptyNDArray();
         }
         NDArray toReturn = null;
         for (NDArray v : vectors) {
            if (toReturn == null) {
               toReturn = v.copy();
            } else {
               toReturn.mapi(v, Math::max);
            }
         }
         return toReturn;
      }
   },
   /**
    * Assigns the minimum element
    */
   Min {
      @Override
      public NDArray compose(@NonNull Collection<NDArray> vectors) {
         if (vectors.size() == 0) {
            return new EmptyNDArray();
         }
         NDArray toReturn = null;
         for (NDArray v : vectors) {
            if (toReturn == null) {
               toReturn = v.copy();
            } else {
               toReturn.mapi(v, Math::min);
            }
         }
         return toReturn;
      }
   },
   SVD {
      @Override
      public NDArray compose(@NonNull Collection<NDArray> vectors) {
         if (vectors.size() == 0) {
            return new EmptyNDArray();
         }
         return com.davidbracewell.apollo.linear.SVD.truncatedSVD(NDArrayFactory.DENSE_DOUBLE.fromRowVectors(vectors),
                                                                  1)[2].getVector(0, Axis.ROW);
      }
   }

}//END OF VectorCompositions
