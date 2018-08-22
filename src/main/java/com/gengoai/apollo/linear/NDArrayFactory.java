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
 * @author David B. Bracewell
 */
public enum NDArrayFactory {
   DENSE {
      @Override
      public NDArray zeros(int... dimensions) {
         dimensions = NDArray.ensureCorrectIndices(dimensions);
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
   }, SPARSE {
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
            sliceList.add(Cast.as(slice));
         }
         return new SparseNDArray(shape, sliceList);
      }
   };


   public static NDArray wrap(double[][] matrix) {
      return new DenseNDArray(new DoubleMatrix(matrix));
   }

   public static NDArray wrap(float[] vector) {
      return new DenseNDArray(new FloatMatrix(vector));
   }

   public static NDArray wrap(double[] vector) {
      return new DenseNDArray(new DoubleMatrix(vector));
   }

   public static NDArray wrap(float[][] matrix) {
      return new DenseNDArray(new FloatMatrix(matrix));
   }


   public static NDArrayFactory DEFAULT() {
      return Config.get("ndarray.factory ").as(NDArrayFactory.class, DENSE);
   }

   public NDArray empty() {
      return zeros(0);
   }

   public abstract NDArray zeros(int... dimensions);

   public NDArray fromLayers(NDArray... slices) {
      return fromLayers(slices.length, 1, slices);
   }

   public abstract NDArray fromLayers(int kernels, int channels, NDArray... slices);

   public NDArray ones(int... dimensions) {
      return create(NDArrayInitializer.ones, dimensions);
   }

   public NDArray constant(float value, int... dimensions) {
      return zeros(dimensions).fill(value);
   }

   public NDArray rand(int... dimensions) {
      return create(NDArrayInitializer.rand, dimensions);
   }

   public NDArray randn(int... dimensions) {
      return create(NDArrayInitializer.randn, dimensions);
   }

   public NDArray create(NDArrayInitializer initializer, int... dimensions) {
      NDArray out = zeros(dimensions);
      initializer.accept(out);
      return out;
   }

   public NDArray eye(int n) {
      NDArray toReturn = zeros(n, n);
      for (int i = 0; i < n; i++) {
         toReturn.set(i, i, 1);
      }
      return toReturn;
   }

   public NDArray stack(Axis axis, NDArray... arrays) {
      return stack(axis, Arrays.asList(arrays));
   }

   public NDArray create(int rows, int columns, double[] data) {
      NDArray toReturn = zeros(rows, columns);
      for (int i = 0; i < data.length; i++) {
         toReturn.set(i, (float) data[i]);
      }
      return toReturn;
   }

   public NDArray create(double[] data) {
      NDArray toReturn = zeros(data.length);
      for (int i = 0; i < data.length; i++) {
         toReturn.set(i, (float) data[i]);
      }
      return toReturn;
   }

   public NDArray create(int rows, int columns, float[] data) {
      NDArray toReturn = zeros(rows, columns);
      for (int i = 0; i < data.length; i++) {
         toReturn.set(i, data[i]);
      }
      return toReturn;
   }

   public NDArray scalar(float value) {
      return zeros(1).set(0, value);
   }

   public NDArray create(float[] data) {
      NDArray toReturn = zeros(data.length);
      for (int i = 0; i < data.length; i++) {
         toReturn.set(i, data[i]);
      }
      return toReturn;
   }


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

   public NDArray vstack(NDArray... rows) {
      return stack(Axis.ROW, rows);
   }

   public NDArray hstack(Collection<NDArray> columns) {
      return stack(Axis.COLUMN, columns);
   }

   public NDArray vstack(Collection<NDArray> rows) {
      return stack(Axis.ROW, rows);
   }

   public NDArray hstack(NDArray... columns) {
      return stack(Axis.COLUMN, columns);
   }

}//END OF NDArrayFactory
