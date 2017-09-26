package com.davidbracewell.apollo.linalg;

import com.davidbracewell.guava.common.base.Preconditions;
import lombok.NonNull;
import org.jblas.DoubleMatrix;

import java.util.Random;

public enum NDArrayFactory {
   SPARSE_DOUBLE {
      @Override
      public NDArray copyOf(@NonNull NDArray array) {
         if (array instanceof SparseDoubleNDArray) {
            return array.copy();
         }
         return zeros(array.shape()).addi(array).setLabel(array.getLabel());
      }

      @Override
      public NDArray zeros(int r, int c) {
         Preconditions.checkArgument(r > 0, "r must be > 0");
         Preconditions.checkArgument(c > 0, "c must be > 0");
         return new SparseDoubleNDArray(r, c);
      }

   },
   SPARSE_INT {
      @Override
      public NDArray copyOf(@NonNull NDArray array) {
         if (array instanceof SparseIntNDArray) {
            return array.copy();
         }
         return zeros(array.shape()).addi(array).setLabel(array.getLabel());
      }

      @Override
      public NDArray zeros(int r, int c) {
         Preconditions.checkArgument(r > 0, "r must be > 0");
         Preconditions.checkArgument(c > 0, "c must be > 0");
         return new SparseIntNDArray(r, c);
      }

   },
   SPARSE_FLOAT {
      @Override
      public NDArray copyOf(@NonNull NDArray array) {
         if (array instanceof SparseFloatNDArray) {
            return array.copy();
         }
         return zeros(array.shape()).addi(array).setLabel(array.getLabel());
      }

      @Override
      public NDArray zeros(int r, int c) {
         Preconditions.checkArgument(r > 0, "r must be > 0");
         Preconditions.checkArgument(c > 0, "c must be > 0");
         return new SparseFloatNDArray(r, c);
      }

   },
   DENSE_DOUBLE {
      @Override
      public NDArray copyOf(@NonNull NDArray array) {
         if (array instanceof DenseDoubleNDArray) {
            return array.copy();
         }
         return zeros(array.shape()).addi(array).setLabel(array.getLabel());
      }

      @Override
      public NDArray zeros(int r, int c) {
         Preconditions.checkArgument(r > 0, "r must be > 0");
         Preconditions.checkArgument(c > 0, "c must be > 0");
         return new DenseDoubleNDArray(DoubleMatrix.zeros(r, c));
      }
   };


   public abstract NDArray copyOf(NDArray array);

   public NDArray diag(@NonNull NDArray other) {
      Preconditions.checkArgument(other.isVector());
      int dim = Math.max(other.shape().i, other.shape().j);
      NDArray toReturn = zeros(dim,dim);
      for (int i = 0; i < dim; i++) {
         toReturn.set(i, i, other.get(i));
      }
      return toReturn;
   }

   public NDArray empty() {
      return new EmptyNDArray();
   }

   public NDArray eye(int n) {
      NDArray toReturn = zeros(n, n);
      for (int i = 0; i < n; i++) {
         toReturn.set(i, i, 1);
      }
      return toReturn;
   }

   public NDArray from(int dimension, double[] data) {
      NDArray z = zeros(dimension);
      for (int i = 0; i < data.length; i++) {
         z.set(i, data[i]);
      }
      return z;
   }

   public NDArray from(int r, int c, double[] data) {
      NDArray z = zeros(r, c);
      for (int i = 0; i < data.length; i++) {
         z.set(i, data[i]);
      }
      return z;
   }

   public NDArray from(int r, int c, double[][] data) {
      NDArray z = zeros(r, c);
      for (int i = 0; i < r; i++) {
         for (int j = 0; j < c; j++) {
            z.set(i, j, data[i][j]);
         }
      }
      return z;
   }

   public NDArray ones(int... dimensions) {
      return zeros(dimensions).fill(1d);
   }

   public NDArray ones(@NonNull Shape shape) {
      return zeros(shape).fill(1d);
   }

   public NDArray ones(@NonNull Axis a1, int dim1, @NonNull Axis a2, int dim2) {
      int[] dimensions = {-1, -1};
      dimensions[a1.index] = dim1;
      dimensions[a2.index] = dim2;
      return ones(dimensions[0], dimensions[1]);
   }

   public NDArray ones(int dimension, @NonNull Axis axis) {
      return ones(axis, dimension, axis.T(), 1);
   }

   public NDArray rand(@NonNull Shape shape) {
      return rand(shape, new Random());
   }

   public NDArray rand(@NonNull Shape shape, @NonNull Random rnd) {
      return zeros(shape).mapi(d -> rnd.nextDouble());
   }

   public NDArray rand(@NonNull Random random, int... dimensions) {
      return zeros(dimensions).mapi(d -> random.nextDouble());
   }

   public NDArray rand(int... dimensions) {
      return rand(new Random(), dimensions);
   }

   public NDArray rand(@NonNull Axis a1, int dim1, @NonNull Axis a2, int dim2) {
      return rand(a1, dim1, a2, dim2, new Random());
   }

   public NDArray rand(@NonNull Axis a1, int dim1, @NonNull Axis a2, int dim2, @NonNull Random random) {
      int[] dimensions = {-1, -1};
      dimensions[a1.index] = dim1;
      dimensions[a2.index] = dim2;
      return rand(random, dimensions[0], dimensions[1]);
   }

   public NDArray rand(int dimension, @NonNull Axis axis) {
      return rand(axis, dimension, axis.T(), 1);
   }

   public NDArray rand(int dimension, @NonNull Axis axis, @NonNull Random random) {
      return rand(axis, dimension, axis.T(), 1, random);
   }

   public NDArray scalar(double value) {
      return new ScalarNDArray(value);
   }

   public NDArray zeros(@NonNull Shape dimensions) {
      return zeros(dimensions.i, dimensions.j);
   }

   public abstract NDArray zeros(int r, int c);

   public NDArray zeros(int dimension, @NonNull Axis axis) {
      return zeros(axis.T(), dimension, axis, 1);
   }

   public NDArray zeros(@NonNull int... dimensions) {
      switch (dimensions.length) {
         case 0:
            return new EmptyNDArray();
         case 1:
            return zeros(dimensions[0], 1);
         case 2:
            return zeros(dimensions[0], dimensions[1]);
      }
      throw new IllegalArgumentException("Invalid number of dimensions: " + dimensions.length);
   }

   public NDArray zeros(@NonNull Axis a1, int dim1, @NonNull Axis a2, int dim2) {
      int[] dimensions = {-1, -1};
      dimensions[a1.index] = dim1;
      dimensions[a2.index] = dim2;
      return zeros(dimensions[0], dimensions[1]);
   }


}//END OF NDArrayFactory
