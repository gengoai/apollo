package com.gengoai.apollo.linear;

import com.gengoai.config.Config;
import com.gengoai.conversion.Cast;
import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import static com.gengoai.Validation.checkArgument;

/**
 * Factory for NDArrays
 *
 * @author David B. Bracewell
 */
public enum NDArrayFactory {
   /**
    * Dense JBlas FloatMatrix backed NDArrays
    */
   DENSE {
      @Override
      public NDArray zeros(int... dimensions) {
         dimensions = ensureDimensions(dimensions);
         FloatMatrix[] matrices = new FloatMatrix[dimensions[2] * dimensions[3]];
         for (int i = 0; i < matrices.length; i++) {
            matrices[i] = FloatMatrix.zeros(dimensions[0], dimensions[1]);
         }
         return new DenseNDArray(matrices, dimensions);
      }

      @Override
      public NDArray scalar(double value) {
         return new DenseNDArray(FloatMatrix.scalar((float) value));
      }

      @Override
      public NDArray fromLayers(int kernels, int channels, NDArray... slices) {
         checkArgument(kernels > 0, "Number of kernels must be > 0");
         checkArgument(channels > 0, "Number of channels must be > 0");
         checkArgument(kernels * channels == slices.length,
                       "Number of slices is more than number of kernels * channels");
         checkArgument(slices[0].isDense(), "Only Dense Layers supported");
         int[] shape = new int[]{slices[0].numRows(), slices[0].numCols(), kernels, channels};
         FloatMatrix[] matrices = Arrays.stream(slices).map(NDArray::toFloatMatrix).toArray(FloatMatrix[]::new);
         return new DenseNDArray(matrices, shape);
      }


   },
   /**
    * Sparse Matrices with boolean array indexes for fast lookup
    */
   SPARSE {
      @Override
      public NDArray zeros(int... dimensions) {
         return new SparseNDArray(dimensions);
      }

      @Override
      public NDArray fromLayers(int kernels, int channels, NDArray... slices) {
         checkArgument(kernels > 0, "Number of kernels must be > 0");
         checkArgument(channels > 0, "Number of channels must be > 0");
         checkArgument(kernels * channels == slices.length,
                       "Number of slices is more than number of kernels * channels");
         checkArgument(slices[0].isSparse(), "Only Sparse Layers supported");
         int[] shape = new int[]{slices[0].numRows(), slices[0].numCols(), kernels, channels};
         List<SparseNDArray> sliceList = new ArrayList<>();
         for (NDArray slice : slices) {
            if (slice instanceof SparseNDArray) {
               sliceList.add(Cast.as(slice));
            } else {
               sliceList.add(new SparseNDArray(slice));
            }
         }
         return new SparseNDArray(shape, sliceList);
      }
   };

   /**
    * Gets the default NDArray factory checking first if it is set via Config and if not defaulting to DENSE.
    *
    * @return the default NDArray Factory
    */
   public static NDArrayFactory DEFAULT() {
      return Config.get("ndarray.factory").as(NDArrayFactory.class, DENSE);
   }

   /**
    * Creates a dense column vector wrapping the given float array (i.e. changes to the array will be reflected in the
    * NDArray).
    *
    * @param vector the float array vector to wrap
    * @return the dense NDArray
    */
   public static NDArray columnVector(float[] vector) {
      return new DenseNDArray(new FloatMatrix(vector.length, 1, vector));
   }

   /**
    * Creates a dense column vector from the given double array.
    *
    * @param vector the double array vector to wrap
    * @return the dense NDArray
    */
   public static NDArray columnVector(double[] vector) {
      return new DenseNDArray(new DoubleMatrix(vector));
   }

   private static int[] ensureDimensions(int... dimensions) {
      checkArgument(dimensions.length <= 4, () -> NDArray.invalidNumberOfIndices(dimensions.length));
      if (dimensions.length == 4) {
         return dimensions;
      }
      int[] shape = new int[]{1, 1, 1, 1};
      System.arraycopy(dimensions, 0, shape, 0, dimensions.length);
      return shape;
   }

   /**
    * Wraps the given 2d float array as a matrix NDArray.
    *
    * @param matrix the matrix
    * @return the dense NDArray
    */
   public static NDArray matrix(float[][] matrix) {
      return new DenseNDArray(new FloatMatrix(matrix));
   }

   /**
    * Wraps the given 2d double array as a matrix NDArray.
    *
    * @param matrix the matrix
    * @return the dense NDArray
    */
   public static NDArray matrix(double[][] matrix) {
      return new DenseNDArray(new DoubleMatrix(matrix));
   }

   /**
    * Creates a dense row vector wrapping the given float array (i.e. changes to the array will be reflected in the
    * NDArray).
    *
    * @param vector the float array vector to wrap
    * @return the dense NDArray
    */
   public static NDArray rowVector(float[] vector) {
      return new DenseNDArray(new FloatMatrix(1, vector.length, vector));
   }

   /**
    * Creates a dense column vector from the given double array.
    *
    * @param vector the double array vector to wrap
    * @return the dense NDArray
    */
   public static NDArray rowVector(double[] vector) {
      return new DenseNDArray(new DoubleMatrix(1, vector.length, vector));
   }

   /**
    * Creates an NDArray of given dimensions filled with the given value.
    *
    * @param value      the constant value to fill the new NDArray with
    * @param dimensions the dimensions of the NDArray
    * @return the NDArray
    */
   public NDArray constant(double value, int... dimensions) {
      return zeros(dimensions).fill(value);
   }

   /**
    * Creates an NDArray of given dimensions initializing it with the given {@link NDArrayInitializer}
    *
    * @param initializer the initializer to use to initialize the values of the NDArray
    * @param dimensions  the dimensions of the NDArray
    * @return the NDArray
    */
   public NDArray create(NDArrayInitializer initializer, int... dimensions) {
      NDArray out = zeros(dimensions);
      initializer.accept(out);
      return out;
   }

   /**
    * Creates an Empty NDArray
    *
    * @return the NDArray
    */
   public NDArray empty() {
      return zeros(0);
   }

   /**
    * Creates an identity matrix
    *
    * @param n the number of rows and columns
    * @return the identity matrix NDArray
    */
   public NDArray eye(int n) {
      NDArray toReturn = zeros(n, n);
      for (int i = 0; i < n; i++) {
         toReturn.set(i, i, 1);
      }
      return toReturn;
   }

   /**
    * Creates a 3D NDArray from the given slices. (Slices are kept as-is meaning external changes to them will be
    * represented in the new NDArray).
    *
    * @param slices the slices
    * @return the NDArray
    */
   public NDArray fromLayers(NDArray... slices) {
      return fromLayers(slices.length, 1, slices);
   }

   /**
    * Creates a 3D or 4D NDArray from the given slices. (Slices are kept as-is meaning external changes to them will be
    * represented in the new NDArray).
    *
    * @param kernels  the number of kernels
    * @param channels the number of channels
    * @param slices   the slices
    * @return the NDArray
    */
   public abstract NDArray fromLayers(int kernels, int channels, NDArray... slices);

   /**
    * Horizontally stacks the given NDArrays concatenating the columns.
    *
    * @param columns the column NDArrays
    * @return the NDArray
    */
   public NDArray hstack(Collection<NDArray> columns) {
      return stack(Axis.COLUMN, columns);
   }

   /**
    * Horizontally stacks the given NDArrays concatenating the columns.
    *
    * @param columns the column NDArrays
    * @return the NDArray
    */
   public NDArray hstack(NDArray... columns) {
      return stack(Axis.COLUMN, Arrays.asList(columns));
   }

   /**
    * Creates a matrix NDArray from the given double array.
    *
    * @param rows    the number of rows
    * @param columns the number of columns
    * @param data    the matrix values
    * @return the NDArray
    */
   public NDArray matrix(int rows, int columns, double[] data) {
      NDArray toReturn = zeros(rows, columns);
      for (int i = 0; i < data.length; i++) {
         toReturn.set(i, (float) data[i]);
      }
      return toReturn;
   }

   /**
    * Creates a matrix NDArray from the given float array.
    *
    * @param rows    the number of rows
    * @param columns the number of columns
    * @param data    the matrix values
    * @return the NDArray
    */
   public NDArray matrix(int rows, int columns, float[] data) {
      NDArray toReturn = zeros(rows, columns);
      for (int i = 0; i < data.length; i++) {
         toReturn.set(i, data[i]);
      }
      return toReturn;
   }

   /**
    * Creates a new NDArray of given dimensions where all values are <code>1.0</code>.
    *
    * @param dimensions the dimensions of the NDArray
    * @return the NDArray
    */
   public NDArray ones(int... dimensions) {
      return create(NDArrayInitializer.ones, dimensions);
   }


   /**
    * Creates a scalar NDArray of size <code>(1,1,1,1)</code> with the given value.
    *
    * @param value the value
    * @return the NDArray
    */
   public NDArray scalar(double value) {
      return constant(value, 1, 1, 1, 1);
   }


   private NDArray stack(Axis axis, Collection<NDArray> arrays) {
      checkArgument(axis.isRowOrColumn(), "Axis (" + axis + ") is not supported.");
      if (arrays.size() == 0) {
         return empty();
      } else if (arrays.size() == 1) {
         return arrays.iterator().next().copy();
      }
      int[] shape = arrays.iterator().next().shape();
      shape[axis.index] = arrays.stream().mapToInt(n -> n.dimension(axis)).sum();
      NDArray toReturn = zeros(shape);

      int globalAxisIndex = 0;
      for (NDArray array : arrays) {
         for (int i = 0; i < array.dimension(axis); i++) {
            toReturn.setVector(globalAxisIndex, axis, array.getVector(i, axis));
            globalAxisIndex++;
         }
      }
      return toReturn;
   }

   /**
    * Vertically stacks the given NDArrays concatenating the rows.
    *
    * @param rows the row NDArrays
    * @return the NDArray
    */
   public NDArray vstack(NDArray... rows) {
      return stack(Axis.ROW, Arrays.asList(rows));
   }

   /**
    * Vertically stacks the given NDArrays concatenating the rows.
    *
    * @param rows the row NDArrays
    * @return the NDArray
    */
   public NDArray vstack(Collection<NDArray> rows) {
      return stack(Axis.ROW, rows);
   }

   /**
    * Creates a new NDArray of given dimensions where all values are <code>0.0</code>.
    *
    * @param dimensions the dimensions of the NDArray
    * @return the NDArray
    */
   public abstract NDArray zeros(int... dimensions);

}//END OF NDArrayFactory
