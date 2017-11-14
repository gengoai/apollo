package com.davidbracewell.apollo.linear;

import com.davidbracewell.apollo.linear.dense.DenseDoubleNDArray;
import com.davidbracewell.apollo.linear.sparse.Sparse2dStorage;
import com.davidbracewell.apollo.linear.sparse.SparseDoubleNDArray;
import com.davidbracewell.apollo.ml.optimization.WeightInitializer;
import com.davidbracewell.config.Config;
import com.davidbracewell.guava.common.base.Preconditions;
import com.davidbracewell.guava.common.collect.Iterables;
import lombok.NonNull;
import org.apache.mahout.math.set.OpenIntHashSet;
import org.jblas.DoubleMatrix;

import java.util.Collection;
import java.util.Random;

/**
 * Factory methods for creating <code>NDArray</code>s.
 */
public enum NDArrayFactory {
   /**
    * The Sparse double.
    */
   SPARSE_DOUBLE {
      @Override
      public NDArray hstack(@NonNull NDArray... columns) {
         if (columns.length == 0) {
            return empty();
         } else if (columns.length == 1) {
            return columns[0];
         }
         return new SparseDoubleNDArray(new Sparse2dStorage(columns));
      }

      @Override
      public NDArray copy(@NonNull NDArray array) {
         if (array instanceof SparseDoubleNDArray) {
            return array.copy();
         }
         return zeros(array.numRows(), array.numCols()).addi(array);
      }

      @Override
      public NDArray zeros(int r, int c) {
         Preconditions.checkArgument(r > 0, "r must be > 0");
         Preconditions.checkArgument(c > 0, "c must be > 0");
         return new SparseDoubleNDArray(r, c);
      }

   },
   /**
    * The Dense double.
    */
   DENSE_DOUBLE {
      @Override
      public NDArray hstack(@NonNull NDArray... columns) {
         if (columns.length == 0) {
            return empty();
         } else if (columns.length == 1) {
            return columns[0];
         }
         if (columns.length == 2) {
            return new DenseDoubleNDArray(DoubleMatrix.concatHorizontally(columns[0].toDoubleMatrix(),
                                                                          columns[1].toDoubleMatrix()));
         }
         int l = columns[0].length();
         double[] a = new double[l * columns.length];
         for (int i = 0; i < columns.length; i++) {
            System.arraycopy(columns[i].toArray(), 0, a, i * l, l);
         }
         return new DenseDoubleNDArray(new DoubleMatrix(columns[0].length(), columns.length, a));
      }

      @Override
      public NDArray copy(@NonNull NDArray array) {
         if (array instanceof DenseDoubleNDArray) {
            return array.copy();
         }
         return zeros(array.numRows(), array.numCols()).addi(array);
      }

      @Override
      public NDArray from(int r, int c, double[] data) {
         return new DenseDoubleNDArray(new DoubleMatrix(r, c, data));
      }

      @Override
      public NDArray from(double[] data) {
         return new DenseDoubleNDArray(new DoubleMatrix(data));
      }

      @Override
      public NDArray zeros(int r, int c) {
         Preconditions.checkArgument(r > 0, "r must be > 0");
         Preconditions.checkArgument(c > 0, "c must be > 0");
         return new DenseDoubleNDArray(DoubleMatrix.zeros(r, c));
      }
   };


   private static volatile NDArrayFactory DEFAULT_INSTANCE;


   /**
    * Gets the default factory defined either in the config setting <code>ndarray.factory</code> or defaults to
    * <code>DENSE_DOUBLE</code>
    *
    * @return The NDArray Factory
    */
   public static NDArrayFactory DEFAULT() {
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
    * Creates a new Empty NDArray
    *
    * @return the NDArray
    */
   public static NDArray empty() {
      return new EmptyNDArray();
   }

   /**
    * Creates a new {@link DenseDoubleNDArray} that wraps the given set of values
    *
    * @param rows   the number of rows
    * @param cols   the number of columns
    * @param values the values
    * @return the NDArray
    */
   public static NDArray wrap(int rows, int cols, @NonNull double[] values) {
      return new DenseDoubleNDArray(new DoubleMatrix(rows, cols, values));
   }

   /**
    * Creates a new {@link DenseDoubleNDArray} that wraps the given set of values
    *
    * @param values the values
    * @return the NDArray
    */
   public static NDArray wrap(@NonNull double[] values) {
      return new DenseDoubleNDArray(new DoubleMatrix(values));
   }

   /**
    * Creates a copy of the given NDArray
    *
    * @param array the array to copy
    * @return the NDArray
    */
   public abstract NDArray copy(NDArray array);

   /**
    * Creates a new NDArray of given shape and initializes using the given initializer
    *
    * @param i           The number of rows
    * @param j           The number of columns
    * @param initializer How to initialize the values in the NDArray
    * @return The NDArray
    */
   public NDArray create(int i, int j, @NonNull WeightInitializer initializer) {
      return initializer.initialize(zeros(i, j));
   }

   /**
    * Creates a new NDArray of given dimension and initializes using the given initializer
    *
    * @param dimension   The dimension of the vector
    * @param initializer How to initialize the values in the NDArray
    * @return The NDArray
    */
   public NDArray create(int dimension, @NonNull WeightInitializer initializer) {
      return initializer.initialize(zeros(dimension));
   }

   /**
    * Creates a diagonal 2D NDArray from a 1D NDArray
    *
    * @param vector the 1D NDArray, or vector, to use as the diagonal
    * @return the NDArray
    */
   public NDArray diag(@NonNull NDArray vector) {
      Preconditions.checkArgument(vector.isVector());
      int dim = Math.max(vector.numRows(), vector.numCols());
      NDArray toReturn = zeros(dim, dim);
      for (int i = 0; i < dim; i++) {
         toReturn.set(i, i, vector.get(i));
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
      int rowdim = Iterables.getFirst(vectors, null).numCols();
      NDArray toReturn = zeros(vectors.size(), rowdim);
      int idx = 0;
      for (NDArray vector : vectors) {
         toReturn.setVector(idx, vector, Axis.ROW);
         idx++;
      }
      return toReturn;
   }

   /**
    * Creates an identity matrix
    *
    * @param n the number of rows and columns
    * @return the NDArray
    */
   public NDArray eye(int n) {
      NDArray toReturn = zeros(n, n);
      for (int i = 0; i < n; i++) {
         toReturn.set(i, i, 1);
      }
      return toReturn;
   }

   /**
    * Creates a new NDArray  that wraps the given set of values
    *
    * @param data the values
    * @return the NDArray
    */
   public NDArray from(double[] data) {
      NDArray z = zeros(data.length);
      for (int i = 0; i < data.length; i++) {
         z.set(i, data[i]);
      }
      return z;
   }

   /**
    * Creates a new NDArray that wraps the given set of values
    *
    * @param r    the number of rows
    * @param c    the number of columns
    * @param data the values
    * @return the NDArray
    */
   public NDArray from(int r, int c, double[] data) {
      NDArray z = zeros(r, c);
      for (int i = 0; i < data.length; i++) {
         z.set(i, data[i]);
      }
      return z;
   }

   /**
    * Creates a new NDArray that wraps the given set of values
    *
    * @param data the values
    * @return the NDArray
    */
   public NDArray from(double[][] data) {
      NDArray z = zeros(data[0].length, data.length);
      for (int j = 0; j < data.length; j++) {
         for (int i = 0; i < data[0].length; i++) {
            z.set(i, j, data[i][j]);
         }
      }
      return z;
   }


   /**
    * Concatenates a series of column vectors into a single NDArray
    *
    * @param columns columns to concatenate
    * @return the NDArray
    */
   public abstract NDArray hstack(NDArray... columns);

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
    * Rand nd array.
    *
    * @param random     the random
    * @param dimensions the dimensions
    * @return the nd array
    */
   public NDArray randn(@NonNull Random random, int... dimensions) {
      return zeros(dimensions).mapi(d -> random.nextGaussian());
   }

   /**
    * Rand nd array.
    *
    * @param sparsity   the sparsity
    * @param random     the random
    * @param dimensions the dimensions
    * @return the nd array
    */
   public NDArray randn(double sparsity, @NonNull Random random, int... dimensions) {
      NDArray toReturn = zeros(dimensions);
      int nonZero = (int) Math.floor((1d - sparsity) * toReturn.length());
      OpenIntHashSet indexes = new OpenIntHashSet();
      while (indexes.size() < nonZero) {
         indexes.add(random.nextInt(toReturn.length()));
      }
      indexes.forEachKey(i -> {
         toReturn.set(i, random.nextGaussian());
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
   public NDArray randn(int... dimensions) {
      return randn(new Random(), dimensions);
   }

   /**
    * Rand nd array.
    *
    * @param sparsity   the sparsity
    * @param dimensions the dimensions
    * @return the nd array
    */
   public NDArray randn(double sparsity, int... dimensions) {
      return randn(sparsity, new Random(), dimensions);
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
   public NDArray randn(@NonNull Axis a1, int dim1, @NonNull Axis a2, int dim2) {
      return randn(a1, dim1, a2, dim2, new Random());
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
   public NDArray randn(@NonNull Axis a1, int dim1, @NonNull Axis a2, int dim2, @NonNull Random random) {
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
   public NDArray randn(int dimension, @NonNull Axis axis) {
      return randn(axis, dimension, axis.T(), 1);
   }

   /**
    * Rand nd array.
    *
    * @param dimension the dimension
    * @param axis      the axis
    * @param random    the random
    * @return the nd array
    */
   public NDArray randn(int dimension, @NonNull Axis axis, @NonNull Random random) {
      return randn(axis, dimension, axis.T(), 1, random);
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
    * Creates a zero valued matrix
    *
    * @param r the number of rows
    * @param c the  number of columns
    * @return the nd array
    * @throws IllegalArgumentException if the number of rows or columns <= 0
    */
   public abstract NDArray zeros(int r, int c);

   /**
    * Creates a zero valued vector for the given axis
    *
    * @param dimension the dimension
    * @param axis      the axis of the vector (row vs column vector)
    * @return the nd array
    */
   public NDArray zeros(int dimension, @NonNull Axis axis) {
      return zeros(axis.T(), dimension, axis, 1);
   }

   /**
    * Creates a zero-value vector with the given dimensions
    *
    * @param dimensions the dimensions
    * @return the nd array
    */
   public NDArray zeros(int... dimensions) {
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
    * Creates a zero-value array with the given axis dimension
    *
    * @param a1   First axis
    * @param dim1 dimension of axis one
    * @param a2   Second axis
    * @param dim2 dimension of axis two
    * @return the nd array
    * @throws IllegalArgumentException if the two axis are the same
    */
   public NDArray zeros(@NonNull Axis a1, int dim1, @NonNull Axis a2, int dim2) {
      Preconditions.checkArgument(a1 != a2, "Axis one and Axis 2 must not be the same");
      int[] dimensions = {-1, -1};
      dimensions[a1.index] = dim1;
      dimensions[a2.index] = dim2;
      return zeros(dimensions[0], dimensions[1]);
   }


}//END OF NDArrayFactory
