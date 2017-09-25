package com.davidbracewell.apollo.linalg;

import org.jblas.DoubleMatrix;

/**
 * @author David B. Bracewell
 */
public interface NDArrayFactory {

   default NDArray ones(int dim1, int dim2) {
      return zeros(dim1, dim2).mapi(d -> 1d);
   }

   default NDArray ones(Shape shape) {
      return zeros(shape).mapi(d -> 1d);
   }

   default NDArray ones(int dimension, Axis axis) {
      switch (axis) {
         case ROW:
            return ones(1, dimension);
         case COlUMN:
            return ones(dimension, 1);
      }
      throw new IllegalArgumentException();
   }

   default NDArray rand(Shape shape) {
      NDArray zeros = zeros(shape);
      for (int i = 0; i < zeros.length(); i++) {
         zeros.set(i, Math.random());
      }
      return zeros;
   }

   default NDArray rand(int dim1, int dim2) {
      return rand(Shape.shape(dim1, dim2));
   }

   NDArray zeros(Shape dimensions);

   default NDArray zeros(int dim1, int dim2) {
      return zeros(Shape.shape(dim1, dim2));
   }

   default NDArray zeros(int dimension, Axis axis) {
      switch (axis) {
         case ROW:
            return zeros(1, dimension);
         case COlUMN:
            return zeros(dimension, 1);
      }
      throw new IllegalArgumentException();
   }

   enum SparseNDArrayFactory implements NDArrayFactory {
      INSTANCE;

      @Override
      public NDArray zeros(Shape dimensions) {
         return new SparseNDArray(dimensions);
      }

   }

   enum DenseDoubleNDArrayFactor implements NDArrayFactory {
      INSTANCE {
         @Override
         public NDArray zeros(Shape dimensions) {
            return new DenseDoubleNDArray(DoubleMatrix.zeros(dimensions.i, dimensions.j));
         }
      };
   }

}//END OF NDArrayFactory
