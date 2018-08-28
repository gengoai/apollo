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
 * The enum Nd array factory.
 *
 * @author David B. Bracewell
 */
public enum NDArrayFactory {
   /**
    * The Dense.
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
    * The Sparse.
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
    * Default nd array factory.
    *
    * @return the nd array factory
    */
   public static NDArrayFactory DEFAULT() {
      return Config.get("ndarray.factory").as(NDArrayFactory.class, DENSE);
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
    * Wrap nd array.
    *
    * @param vector the vector
    * @return the nd array
    */
   public static NDArray wrap(float[] vector) {
      return new DenseNDArray(new FloatMatrix(vector));
   }

   /**
    * Wrap nd array.
    *
    * @param vector the vector
    * @return the nd array
    */
   public static NDArray wrap(double[] vector) {
      return new DenseNDArray(new DoubleMatrix(vector));
   }

   /**
    * Wrap nd array.
    *
    * @param matrix the matrix
    * @return the nd array
    */
   public static NDArray wrap(float[][] matrix) {
      return new DenseNDArray(new FloatMatrix(matrix));
   }

   /**
    * Wrap nd array.
    *
    * @param matrix the matrix
    * @return the nd array
    */
   public static NDArray wrap(double[][] matrix) {
      return new DenseNDArray(new DoubleMatrix(matrix));
   }

   /**
    * Constant nd array.
    *
    * @param value      the value
    * @param dimensions the dimensions
    * @return the nd array
    */
   public NDArray constant(float value, int... dimensions) {
      return zeros(dimensions).fill(value);
   }

   /**
    * Create nd array.
    *
    * @param initializer the initializer
    * @param dimensions  the dimensions
    * @return the nd array
    */
   public NDArray create(NDArrayInitializer initializer, int... dimensions) {
      NDArray out = zeros(dimensions);
      initializer.accept(out);
      return out;
   }

   /**
    * Create nd array.
    *
    * @param rows    the rows
    * @param columns the columns
    * @param data    the data
    * @return the nd array
    */
   public NDArray create(int rows, int columns, double[] data) {
      NDArray toReturn = zeros(rows, columns);
      for (int i = 0; i < data.length; i++) {
         toReturn.set(i, (float) data[i]);
      }
      return toReturn;
   }

   /**
    * Create nd array.
    *
    * @param data the data
    * @return the nd array
    */
   public NDArray create(double[] data) {
      NDArray toReturn = zeros(data.length);
      for (int i = 0; i < data.length; i++) {
         toReturn.set(i, (float) data[i]);
      }
      return toReturn;
   }

   /**
    * Create nd array.
    *
    * @param rows    the rows
    * @param columns the columns
    * @param data    the data
    * @return the nd array
    */
   public NDArray create(int rows, int columns, float[] data) {
      NDArray toReturn = zeros(rows, columns);
      for (int i = 0; i < data.length; i++) {
         toReturn.set(i, data[i]);
      }
      return toReturn;
   }

   /**
    * Create nd array.
    *
    * @param data the data
    * @return the nd array
    */
   public NDArray create(float[] data) {
      NDArray toReturn = zeros(data.length);
      for (int i = 0; i < data.length; i++) {
         toReturn.set(i, data[i]);
      }
      return toReturn;
   }

   /**
    * Empty nd array.
    *
    * @return the nd array
    */
   public NDArray empty() {
      return zeros(0);
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
    * From layers nd array.
    *
    * @param slices the slices
    * @return the nd array
    */
   public NDArray fromLayers(NDArray... slices) {
      return fromLayers(slices.length, 1, slices);
   }

   /**
    * From layers nd array.
    *
    * @param kernels  the kernels
    * @param channels the channels
    * @param slices   the slices
    * @return the nd array
    */
   public abstract NDArray fromLayers(int kernels, int channels, NDArray... slices);

   /**
    * Hstack nd array.
    *
    * @param columns the columns
    * @return the nd array
    */
   public NDArray hstack(Collection<NDArray> columns) {
      return stack(Axis.COLUMN, columns);
   }

   /**
    * Hstack nd array.
    *
    * @param columns the columns
    * @return the nd array
    */
   public NDArray hstack(NDArray... columns) {
      return stack(Axis.COLUMN, columns);
   }

   /**
    * Ones nd array.
    *
    * @param dimensions the dimensions
    * @return the nd array
    */
   public NDArray ones(int... dimensions) {
      return create(NDArrayInitializer.ones, dimensions);
   }

   /**
    * Rand nd array.
    *
    * @param dimensions the dimensions
    * @return the nd array
    */
   public NDArray rand(int... dimensions) {
      return create(NDArrayInitializer.rand, dimensions);
   }

   /**
    * Randn nd array.
    *
    * @param dimensions the dimensions
    * @return the nd array
    */
   public NDArray randn(int... dimensions) {
      return create(NDArrayInitializer.randn, dimensions);
   }

   /**
    * Scalar nd array.
    *
    * @param value the value
    * @return the nd array
    */
   public NDArray scalar(float value) {
      return zeros(1).set(0, value);
   }

   /**
    * Stack nd array.
    *
    * @param axis   the axis
    * @param arrays the arrays
    * @return the nd array
    */
   public NDArray stack(Axis axis, NDArray... arrays) {
      return stack(axis, Arrays.asList(arrays));
   }

   /**
    * Stack nd array.
    *
    * @param axis   the axis
    * @param arrays the arrays
    * @return the nd array
    */
   public NDArray stack(Axis axis, Collection<NDArray> arrays) {
      checkArgument(axis.isRowOrColumn(), "Axis (" + axis + ") is not supported.");
      if (arrays.size() == 0) {
         return empty();
      } else if (arrays.size() == 1) {
         return arrays.iterator().next().copy();
      }
      int[] shape = arrays.iterator().next().shape();
      shape[axis.ordinal] = arrays.stream().mapToInt(n -> n.dimension(axis)).sum();
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
    * Vstack nd array.
    *
    * @param rows the rows
    * @return the nd array
    */
   public NDArray vstack(NDArray... rows) {
      return stack(Axis.ROW, rows);
   }

   /**
    * Vstack nd array.
    *
    * @param rows the rows
    * @return the nd array
    */
   public NDArray vstack(Collection<NDArray> rows) {
      return stack(Axis.ROW, rows);
   }

   /**
    * Zeros nd array.
    *
    * @param dimensions the dimensions
    * @return the nd array
    */
   public abstract NDArray zeros(int... dimensions);

}//END OF NDArrayFactory
