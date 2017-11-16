package com.davidbracewell.apollo.linear;

import com.davidbracewell.apollo.linear.dense.DenseDoubleNDArray;
import com.davidbracewell.apollo.linear.dense.DenseFloatNDArray;
import com.davidbracewell.apollo.linear.sparse.Sparse2dStorage;
import com.davidbracewell.apollo.linear.sparse.SparseDoubleNDArray;
import com.davidbracewell.config.Config;
import com.davidbracewell.guava.common.base.Preconditions;
import com.davidbracewell.guava.common.collect.Iterables;
import lombok.NonNull;
import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;

import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;

/**
 * Factory methods for creating <code>NDArray</code>s.
 */
public enum NDArrayFactory {
   /**
    * Factory for creating sparse double NDArrays
    */
   SPARSE_DOUBLE {
      @Override
      public NDArray hstack(@NonNull Collection<NDArray> columns) {
         if (columns.isEmpty()) {
            return empty();
         } else if (columns.size() == 1) {
            return Iterables.getOnlyElement(columns).copy();
         }
         return new SparseDoubleNDArray(new Sparse2dStorage(columns));
      }

      @Override
      public NDArray copy(@NonNull NDArray array) {
         if (array instanceof SparseDoubleNDArray) {
            return array.copy();
         }
         return zeros(array.numRows(), array.numCols())
                   .addi(array)
                   .setLabel(array.getLabel())
                   .setWeight(array.getWeight())
                   .setPredicted(array.getPredicted());
      }

      @Override
      public NDArray zeros(int r, int c) {
         Preconditions.checkArgument(r > 0, "r must be > 0");
         Preconditions.checkArgument(c > 0, "c must be > 0");
         return new SparseDoubleNDArray(r, c);
      }

   },
   /**
    * Factory for creating dense double NDArrays
    */
   DENSE_DOUBLE {
      @Override
      public NDArray hstack(@NonNull Collection<NDArray> columns) {
         if (columns.isEmpty()) {
            return empty();
         } else if (columns.size() == 1) {
            return Iterables.getOnlyElement(columns).copy();
         }
         if (columns.size() == 2) {
            Iterator<NDArray> itr = columns.iterator();
            return new DenseDoubleNDArray(DoubleMatrix.concatHorizontally(itr.next().toDoubleMatrix(),
                                                                          itr.next().toDoubleMatrix()));
         }
         int l = Iterables.getFirst(columns, null).length();
         double[] a = new double[l * columns.size()];
         int i = 0;
         for (NDArray column : columns) {
            System.arraycopy(column.toArray(), 0, a, i * l, l);
            i++;
         }
         return new DenseDoubleNDArray(new DoubleMatrix(l, columns.size(), a));
      }

      @Override
      public NDArray copy(@NonNull NDArray array) {
         if (array instanceof DenseDoubleNDArray) {
            return array.copy();
         }
         return zeros(array.numRows(), array.numCols())
                   .addi(array)
                   .setLabel(array.getLabel())
                   .setWeight(array.getWeight())
                   .setPredicted(array.getPredicted());
      }

      @Override
      public NDArray create(int r, int c, double[] data) {
         return new DenseDoubleNDArray(new DoubleMatrix(r, c, data));
      }

      @Override
      public NDArray create(double[] data) {
         return new DenseDoubleNDArray(new DoubleMatrix(data));
      }

      @Override
      public NDArray zeros(int r, int c) {
         Preconditions.checkArgument(r > 0, "r must be > 0");
         Preconditions.checkArgument(c > 0, "c must be > 0");
         return new DenseDoubleNDArray(DoubleMatrix.zeros(r, c));
      }
   },
   /**
    * Factory for creating dense double NDArrays
    */
   DENSE_FLOAT {
      @Override
      public NDArray hstack(@NonNull Collection<NDArray> columns) {
         if (columns.isEmpty()) {
            return empty();
         } else if (columns.size() == 1) {
            return Iterables.getOnlyElement(columns).copy();
         }
         if (columns.size() == 2) {
            Iterator<NDArray> itr = columns.iterator();
            return new DenseFloatNDArray(FloatMatrix.concatHorizontally(itr.next().toFloatMatrix(),
                                                                        itr.next().toFloatMatrix()));
         }
         int l = Iterables.getFirst(columns, null).length();
         float[] a = new float[l * columns.size()];
         int i = 0;
         for (NDArray column : columns) {
            System.arraycopy(column.toFloatArray(), 0, a, i * l, l);
            i++;
         }
         return new DenseFloatNDArray(new FloatMatrix(l, columns.size(), a));
      }

      @Override
      public NDArray copy(@NonNull NDArray array) {
         if (array instanceof DenseFloatNDArray) {
            return array.copy();
         }
         return zeros(array.numRows(), array.numCols())
                   .addi(array)
                   .setLabel(array.getLabel())
                   .setWeight(array.getWeight())
                   .setPredicted(array.getPredicted());
      }

      private float[] convert(double[] in){
         float[] out = new float[in.length];
         for (int i = 0; i < in.length; i++) {
            out[i] = (float)in[i];
         }
         return out;
      }

      @Override
      public NDArray create(int r, int c, double[] data) {
         return new DenseFloatNDArray(new FloatMatrix(r, c, convert(data)));
      }

      @Override
      public NDArray create(double[] data) {
         return new DenseFloatNDArray(new FloatMatrix(convert(data)));
      }

      @Override
      public NDArray zeros(int r, int c) {
         Preconditions.checkArgument(r > 0, "r must be > 0");
         Preconditions.checkArgument(c > 0, "c must be > 0");
         return new DenseFloatNDArray(FloatMatrix.zeros(r, c));
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
               DEFAULT_INSTANCE = Config.get("ndarray.factory").as(NDArrayFactory.class, DENSE_FLOAT);
            }
         }
      }
      return DEFAULT_INSTANCE;
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
   public NDArray create(int i, int j, @NonNull NDArrayInitializer initializer) {
      return initializer.initialize(zeros(i, j));
   }

   /**
    * Creates a new NDArray of given dimension and initializes using the given initializer
    *
    * @param dimension   The dimension of the vector
    * @param initializer How to initialize the values in the NDArray
    * @return The NDArray
    */
   public NDArray create(int dimension, @NonNull NDArrayInitializer initializer) {
      return initializer.initialize(zeros(dimension));
   }

   /**
    * Creates a new NDArray  that wraps the given set of values
    *
    * @param data the values
    * @return the NDArray
    */
   public NDArray create(double[] data) {
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
   public NDArray create(int r, int c, double[] data) {
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
   public NDArray create(double[][] data) {
      NDArray z = zeros(data[0].length, data.length);
      for (int j = 0; j < data.length; j++) {
         for (int i = 0; i < data[0].length; i++) {
            z.set(i, j, data[i][j]);
         }
      }
      return z;
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
    * Creates a new Empty NDArray
    *
    * @return the NDArray
    */
   public NDArray empty() {
      return EmptyNDArray.INSTANCE;
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
    * Concatenates a series of column vectors into a single NDArray
    *
    * @param columns columns to concatenate
    * @return the NDArray
    */
   public NDArray hstack(@NonNull NDArray... columns) {
      return hstack(Arrays.asList(columns));
   }

   /**
    * Concatenates a series of column vectors into a single NDArray
    *
    * @param columns columns to concatenate
    * @return the NDArray
    */
   public abstract NDArray hstack(@NonNull Collection<NDArray> columns);

   /**
    * Creates an NDArray with the given dimensions filled with ones.
    *
    * @param dimensions the dimensions
    * @return the NDArray
    */
   public NDArray ones(@NonNull int... dimensions) {
      return zeros(dimensions).fill(1d);
   }

   /**
    * Creates a one-valued array with the given axis dimension
    *
    * @param a1   First axis
    * @param dim1 dimension of axis one
    * @param a2   Second axis
    * @param dim2 dimension of axis two
    * @return the nd array
    * @throws IllegalArgumentException if the two axis are the same
    */
   public NDArray ones(@NonNull Axis a1, int dim1, @NonNull Axis a2, int dim2) {
      Preconditions.checkArgument(a1 != a2, "Axis one and Axis 2 must not be the same");
      int[] dimensions = {-1, -1};
      dimensions[a1.index] = dim1;
      dimensions[a2.index] = dim2;
      return ones(dimensions[0], dimensions[1]);
   }

   /**
    * Creates a one valued vector for the given axis
    *
    * @param axis      the axis of the vector (row vs column vector)
    * @param dimension the dimension
    * @return the nd array
    */
   public NDArray ones(@NonNull Axis axis, int dimension) {
      return ones(axis, dimension, axis.T(), 1);
   }

   /**
    * Creates an NDArray of given dimensions initialized with Random values
    *
    * @param dimension the dimension
    * @return the NDArray
    */
   public NDArray rand(@NonNull int... dimension) {
      if (dimension.length == 0) {
         return empty();
      } else if (dimension.length == 1) {
         return create(dimension[0], NDArrayInitializer.rand());
      }
      return create(dimension[0], dimension[1], NDArrayInitializer.rand());
   }

   /**
    * Creates an NDArray of given dimensions initialized with Random values following a normal distribution.
    *
    * @param dimension the dimension
    * @return the NDArray
    */
   public NDArray randn(@NonNull int... dimension) {
      if (dimension.length == 0) {
         return empty();
      } else if (dimension.length == 1) {
         return create(dimension[0], NDArrayInitializer.randn());
      }
      return create(dimension[0], dimension[1], NDArrayInitializer.randn());
   }

   /**
    * Creates an NDArray containing a single scalar value
    *
    * @param value the value
    * @return the NDArray
    */
   public NDArray scalar(double value) {
      return new ScalarNDArray(value);
   }

   /**
    * Concatenates a series of row vectors into a single NDArray
    *
    * @param rows rows to concatenate
    * @return the NDArray
    */
   public NDArray vstack(@NonNull Collection<NDArray> rows) {
      if (rows.isEmpty()) {
         return EmptyNDArray.INSTANCE;
      } else if (rows.size() == 1) {
         return Iterables.getOnlyElement(rows);
      }
      int rowdim = Iterables.getFirst(rows, null).numCols();
      NDArray toReturn = zeros(rows.size(), rowdim);
      int idx = 0;
      for (NDArray vector : rows) {
         toReturn.setVector(idx, vector, Axis.ROW);
         idx++;
      }
      return toReturn;
   }

   /**
    * Concatenates a series of row vectors into a single NDArray
    *
    * @param rows rows to concatenate
    * @return the NDArray
    */
   public NDArray vstack(@NonNull NDArray... rows) {
      return vstack(Arrays.asList(rows));
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
    * @param axis      the axis of the vector (row vs column vector)
    * @param dimension the dimension
    * @return the nd array
    */
   public NDArray zeros(@NonNull Axis axis, int dimension) {
      return zeros(axis, dimension, axis.T(), 1);
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
            return EmptyNDArray.INSTANCE;
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
