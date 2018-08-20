package com.gengoai.apollo.linear.v2;

import com.gengoai.tuple.Tuple2;
import org.jblas.FloatMatrix;

import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;

import static com.gengoai.Validation.checkArgument;
import static com.gengoai.tuple.Tuples.$;

/**
 * @author David B. Bracewell
 */
public enum NDArrayFactory {
   DENSE {
      @Override
      public NDArray zeros(int... dimensions) {
         return create(t -> new FloatMatrix(t.v1, t.v2), dimensions);
      }

      @Override
      public NDArray fromLayers(int kernels, int channels, NDArray... slices) {
         checkArgument(kernels > 0, "Number of kernels must be > 0");
         checkArgument(channels > 0, "Number of channels must be > 0");
         checkArgument(kernels * channels == slices.length,
                       "Number of slices is more than number of kernels * channels");
         int[] shape = new int[]{slices[0].rows(), slices[0].columns(), kernels, channels};
         FloatMatrix[] matrices = Arrays.stream(slices).map(NDArray::toFloatMatrix).toArray(FloatMatrix[]::new);
         return new DenseNDArray(matrices, shape);
      }

      @Override
      public NDArray randn(int... dimensions) {
         return create(t -> FloatMatrix.randn(t.v1, t.v2), dimensions);
      }

      @Override
      public NDArray rand(int... dimensions) {
         return create(t -> FloatMatrix.rand(t.v1, t.v2), dimensions);
      }

      private NDArray create(Function<Tuple2<Integer, Integer>, FloatMatrix> supplier, int... dimensions) {
         dimensions = Util.ensureCorrectIndicies(dimensions);
         FloatMatrix[] matrices = new FloatMatrix[dimensions[2] * dimensions[3]];
         for (int i = 0; i < matrices.length; i++) {
            matrices[i] = supplier.apply($(dimensions[0], dimensions[1]));
         }
         return new DenseNDArray(matrices, dimensions);
      }

   };


   public abstract NDArray zeros(int... dimensions);

   public NDArray fromLayers(NDArray... slices) {
      return fromLayers(slices.length, 1, slices);
   }

   public abstract NDArray fromLayers(int kernels, int channels, NDArray... slices);

   public NDArray ones(int... dimensions) {
      return zeros(dimensions).addi(1);
   }

   public NDArray constant(float value, int... dimensions) {
      return zeros(dimensions).fill(value);
   }

   public NDArray randn(int... dimensions) {
      final Random random = new Random();
      return zeros(dimensions).mapi(v -> random.nextGaussian());
   }

   public NDArray rand(int... dimensions) {
      return zeros(dimensions).mapi(v -> Math.random());
   }

   public NDArray eye(int n){
      NDArray toReturn = zeros(n, n);
      for (int i = 0; i < n; i++) {
         toReturn.set(i, i, 1);
      }
      return toReturn;
   }

}//END OF NDArrayFactory
