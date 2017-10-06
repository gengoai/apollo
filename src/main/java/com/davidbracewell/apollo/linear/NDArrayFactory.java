package com.davidbracewell.apollo.linear;

import com.davidbracewell.apollo.linear.dense.DenseDoubleNDArray;
import com.davidbracewell.apollo.linear.sparse.SparseDoubleNDArray;
import com.davidbracewell.apollo.linear.sparse.SparseFloatNDArray;
import com.davidbracewell.apollo.linear.sparse.SparseIntNDArray;
import com.davidbracewell.config.Config;
import com.davidbracewell.guava.common.base.Preconditions;
import com.davidbracewell.guava.common.collect.Iterables;
import lombok.NonNull;
import org.apache.mahout.math.set.OpenIntHashSet;
import org.jblas.DoubleMatrix;

import java.util.Collection;
import java.util.Random;

/**
 * The enum Nd array factory.
 */
public enum NDArrayFactory {
   /**
    * The Sparse double.
    */
   SPARSE_DOUBLE {
      @Override
      public NDArray copyOf(@NonNull NDArray array) {
         if (array instanceof SparseDoubleNDArray) {
            return array.copy();
         }
         return zeros(array.shape()).addi(array);
      }

      @Override
      public NDArray zeros(int r, int c) {
         Preconditions.checkArgument(r > 0, "r must be > 0");
         Preconditions.checkArgument(c > 0, "c must be > 0");
         return new SparseDoubleNDArray(r, c);
      }

   },
   /**
    * The Sparse int.
    */
   SPARSE_INT {
      @Override
      public NDArray copyOf(@NonNull NDArray array) {
         if (array instanceof SparseIntNDArray) {
            return array.copy();
         }
         return zeros(array.shape()).addi(array);
      }

      @Override
      public NDArray zeros(int r, int c) {
         Preconditions.checkArgument(r > 0, "r must be > 0");
         Preconditions.checkArgument(c > 0, "c must be > 0");
         return new SparseIntNDArray(r, c);
      }

   },
   /**
    * The Sparse float.
    */
   SPARSE_FLOAT {
      @Override
      public NDArray copyOf(@NonNull NDArray array) {
         if (array instanceof SparseFloatNDArray) {
            return array.copy();
         }
         return zeros(array.shape()).addi(array);
      }

      @Override
      public NDArray zeros(int r, int c) {
         Preconditions.checkArgument(r > 0, "r must be > 0");
         Preconditions.checkArgument(c > 0, "c must be > 0");
         return new SparseFloatNDArray(r, c);
      }

   },
   /**
    * The Dense double.
    */
   DENSE_DOUBLE {
      @Override
      public NDArray copyOf(@NonNull NDArray array) {
         if (array instanceof DenseDoubleNDArray) {
            return array.copy();
         }
         return zeros(array.shape()).addi(array);
      }

      @Override
      public NDArray from(int r, int c, double[] data) {
         return new DenseDoubleNDArray(new DoubleMatrix(r, c, data));
      }

      @Override
      public NDArray zeros(int r, int c) {
         Preconditions.checkArgument(r > 0, "r must be > 0");
         Preconditions.checkArgument(c > 0, "c must be > 0");
         return new DenseDoubleNDArray(DoubleMatrix.zeros(r, c));
      }
   };


   private static volatile NDArrayFactory DEFAULT_INSTANCE;

   public static NDArrayFactory defaultFactory() {
      if (DEFAULT_INSTANCE == null) {
         synchronized (NDArrayFactory.class) {
            if (DEFAULT_INSTANCE == null) {
               DEFAULT_INSTANCE = Config.get("ndarray.factory").as(NDArrayFactory.class, DENSE_DOUBLE);
            }
         }
      }
      return DEFAULT_INSTANCE;
   }

   /**
    * Wrap nd array.
    *
    * @param rows   the rows
    * @param cols   the cols
    * @param values the values
    * @return the nd array
    */
   public static NDArray wrap(int rows, int cols, @NonNull double[] values) {
      return new DenseDoubleNDArray(new DoubleMatrix(rows, cols, values));
   }

   /**
    * Wrap nd array.
    *
    * @param values the values
    * @return the nd array
    */
   public static NDArray wrap(@NonNull double[] values) {
      return new DenseDoubleNDArray(new DoubleMatrix(values));
   }

   /**
    * Copy of nd array.
    *
    * @param array the array
    * @return the nd array
    */
   public abstract NDArray copyOf(NDArray array);

   /**
    * Diag nd array.
    *
    * @param other the other
    * @return the nd array
    */
   public NDArray diag(@NonNull NDArray other) {
      Preconditions.checkArgument(other.isVector());
      int dim = Math.max(other.shape().i, other.shape().j);
      NDArray toReturn = zeros(dim, dim);
      for (int i = 0; i < dim; i++) {
         toReturn.set(i, i, other.get(i));
      }
      return toReturn;
   }

   /**
    * Empty nd array.
    *
    * @return the nd array
    */
   public NDArray empty() {
      return new EmptyNDArray();
   }

   /**
    * Eye nd array.
    *
    * @param n the n
    * @return the nd array
    */
   public NDArray eye(int n) {
      NDArray toReturn = zeros(n, n);
      for (int i = 0; i < n; i++) {
         toReturn.set(i, i, 1);
      }
      return toReturn;
   }

   /**
    * From nd array.
    *
    * @param dimension the dimension
    * @param data      the data
    * @return the nd array
    */
   public NDArray from(int dimension, double[] data) {
      NDArray z = zeros(dimension);
      for (int i = 0; i < data.length; i++) {
         z.set(i, data[i]);
      }
      return z;
   }

   /**
    * From nd array.
    *
    * @param r    the r
    * @param c    the c
    * @param data the data
    * @return the nd array
    */
   public NDArray from(int r, int c, double[] data) {
      NDArray z = zeros(r, c);
      for (int i = 0; i < data.length; i++) {
         z.set(i, data[i]);
      }
      return z;
   }

   /**
    * From nd array.
    *
    * @param r    the r
    * @param c    the c
    * @param data the data
    * @return the nd array
    */
   public NDArray from(int r, int c, double[][] data) {
      NDArray z = zeros(r, c);
      for (int i = 0; i < r; i++) {
         for (int j = 0; j < c; j++) {
            z.set(i, j, data[i][j]);
         }
      }
      return z;
   }

   public NDArray fromColumnVectors(@NonNull Collection<NDArray> vectors) {
      if (vectors.isEmpty()) {
         return new EmptyNDArray();
      }
      int rowdim = Iterables.getFirst(vectors, null).shape().i;
      NDArray toReturn = zeros(rowdim, vectors.size());
      int idx = 0;
      for (NDArray vector : vectors) {
         toReturn.setVector(idx, vector, Axis.COlUMN);
         idx++;
      }
      return toReturn;
   }

   /**
    * From row vectors nd array.
    *
    * @param vectors the vectors
    * @return the nd array
    */
   public NDArray fromRowVectors(@NonNull Collection<NDArray> vectors) {
      if (vectors.isEmpty()) {
         return new EmptyNDArray();
      }
      int rowdim = Iterables.getFirst(vectors, null).shape().j;
      NDArray toReturn = zeros(vectors.size(), rowdim);
      int idx = 0;
      for (NDArray vector : vectors) {
         toReturn.setVector(idx, vector, Axis.ROW);
         idx++;
      }
      return toReturn;
   }

   /**
    * Ones nd array.
    *
    * @param dimensions the dimensions
    * @return the nd array
    */
   public NDArray ones(int... dimensions) {
      return zeros(dimensions).fill(1d);
   }

   /**
    * Ones nd array.
    *
    * @param shape the shape
    * @return the nd array
    */
   public NDArray ones(@NonNull Shape shape) {
      return zeros(shape).fill(1d);
   }

   /**
    * Ones nd array.
    *
    * @param a1   the a 1
    * @param dim1 the dim 1
    * @param a2   the a 2
    * @param dim2 the dim 2
    * @return the nd array
    */
   public NDArray ones(@NonNull Axis a1, int dim1, @NonNull Axis a2, int dim2) {
      int[] dimensions = {-1, -1};
      dimensions[a1.index] = dim1;
      dimensions[a2.index] = dim2;
      return ones(dimensions[0], dimensions[1]);
   }

   /**
    * Ones nd array.
    *
    * @param dimension the dimension
    * @param axis      the axis
    * @return the nd array
    */
   public NDArray ones(int dimension, @NonNull Axis axis) {
      return ones(axis, dimension, axis.T(), 1);
   }

   /**
    * Rand nd array.
    *
    * @param shape the shape
    * @return the nd array
    */
   public NDArray rand(@NonNull Shape shape) {
      return rand(shape, new Random());
   }

   /**
    * Rand nd array.
    *
    * @param shape the shape
    * @param rnd   the rnd
    * @return the nd array
    */
   public NDArray rand(@NonNull Shape shape, @NonNull Random rnd) {
      return zeros(shape).mapi(d -> rnd.nextDouble());
   }

   /**
    * Rand nd array.
    *
    * @param random     the random
    * @param dimensions the dimensions
    * @return the nd array
    */
   public NDArray rand(@NonNull Random random, int... dimensions) {
      return zeros(dimensions).mapi(d -> random.nextDouble());
   }

   /**
    * Rand nd array.
    *
    * @param sparsity   the sparsity
    * @param random     the random
    * @param dimensions the dimensions
    * @return the nd array
    */
   public NDArray rand(double sparsity, @NonNull Random random, int... dimensions) {
      NDArray toReturn = zeros(dimensions);
      int nonZero = (int) Math.floor((1d - sparsity) * toReturn.length());
      OpenIntHashSet indexes = new OpenIntHashSet();
      while (indexes.size() < nonZero) {
         indexes.add(random.nextInt(toReturn.length()));
      }
      indexes.forEachKey(i -> {
         toReturn.set(i, random.nextDouble());
         return true;
      });
      return toReturn;
   }

   /**
    * Rand nd array.
    *
    * @param dimensions the dimensions
    * @return the nd array
    */
   public NDArray rand(int... dimensions) {
      return rand(new Random(), dimensions);
   }

   /**
    * Rand nd array.
    *
    * @param sparsity   the sparsity
    * @param dimensions the dimensions
    * @return the nd array
    */
   public NDArray rand(double sparsity, int... dimensions) {
      return rand(sparsity, new Random(), dimensions);
   }

   /**
    * Rand nd array.
    *
    * @param a1   the a 1
    * @param dim1 the dim 1
    * @param a2   the a 2
    * @param dim2 the dim 2
    * @return the nd array
    */
   public NDArray rand(@NonNull Axis a1, int dim1, @NonNull Axis a2, int dim2) {
      return rand(a1, dim1, a2, dim2, new Random());
   }

   /**
    * Rand nd array.
    *
    * @param a1     the a 1
    * @param dim1   the dim 1
    * @param a2     the a 2
    * @param dim2   the dim 2
    * @param random the random
    * @return the nd array
    */
   public NDArray rand(@NonNull Axis a1, int dim1, @NonNull Axis a2, int dim2, @NonNull Random random) {
      int[] dimensions = {-1, -1};
      dimensions[a1.index] = dim1;
      dimensions[a2.index] = dim2;
      return rand(random, dimensions[0], dimensions[1]);
   }

   /**
    * Rand nd array.
    *
    * @param dimension the dimension
    * @param axis      the axis
    * @return the nd array
    */
   public NDArray rand(int dimension, @NonNull Axis axis) {
      return rand(axis, dimension, axis.T(), 1);
   }

   /**
    * Rand nd array.
    *
    * @param dimension the dimension
    * @param axis      the axis
    * @param random    the random
    * @return the nd array
    */
   public NDArray rand(int dimension, @NonNull Axis axis, @NonNull Random random) {
      return rand(axis, dimension, axis.T(), 1, random);
   }

   /**
    * Scalar nd array.
    *
    * @param value the value
    * @return the nd array
    */
   public NDArray scalar(double value) {
      return new ScalarNDArray(value);
   }

   /**
    * Zeros nd array.
    *
    * @param dimensions the dimensions
    * @return the nd array
    */
   public NDArray zeros(@NonNull Shape dimensions) {
      return zeros(dimensions.i, dimensions.j);
   }

   /**
    * Zeros nd array.
    *
    * @param r the r
    * @param c the c
    * @return the nd array
    */
   public abstract NDArray zeros(int r, int c);

   /**
    * Zeros nd array.
    *
    * @param dimension the dimension
    * @param axis      the axis
    * @return the nd array
    */
   public NDArray zeros(int dimension, @NonNull Axis axis) {
      return zeros(axis.T(), dimension, axis, 1);
   }

   /**
    * Zeros nd array.
    *
    * @param dimensions the dimensions
    * @return the nd array
    */
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

   /**
    * Zeros nd array.
    *
    * @param a1   the a 1
    * @param dim1 the dim 1
    * @param a2   the a 2
    * @param dim2 the dim 2
    * @return the nd array
    */
   public NDArray zeros(@NonNull Axis a1, int dim1, @NonNull Axis a2, int dim2) {
      int[] dimensions = {-1, -1};
      dimensions[a1.index] = dim1;
      dimensions[a2.index] = dim2;
      return zeros(dimensions[0], dimensions[1]);
   }

}//END OF NDArrayFactory
