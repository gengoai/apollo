package com.gengoai.apollo.linear.v2;

import com.gengoai.Validation;
import com.gengoai.conversion.Cast;
import com.gengoai.json.JsonEntry;
import com.gengoai.tuple.Tuple2;
import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;

import java.util.Arrays;
import java.util.function.BinaryOperator;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static com.gengoai.Validation.checkArgument;
import static com.gengoai.apollo.linear.v2.Axis.*;
import static com.gengoai.apollo.linear.v2.NDArrayFactory.DENSE;
import static com.gengoai.collection.Iterators.zipWithIndex;
import static com.gengoai.tuple.Tuples.$;

/**
 * @author David B. Bracewell
 */
public class DenseNDArray extends NDArray {
   private static final long serialVersionUID = 1L;
   private FloatMatrix[] data;

   protected DenseNDArray(FloatMatrix matrix) {
      super(new int[]{matrix.rows, matrix.columns, 1, 1});
      this.data = new FloatMatrix[]{matrix};
   }

   protected DenseNDArray(FloatMatrix[] matrix, int[] dims) {
      super(dims);
      this.data = matrix;
   }


   protected DenseNDArray(NDArray other) {
      super(other.shape);
      this.data = new FloatMatrix[other.slices()];
      IntStream.range(0, data.length).parallel().forEach(slice -> {
         if (other instanceof DenseNDArray) {
            this.data[slice] = other.slice(slice).toFloatMatrix().dup();
         } else {
            this.data[slice] = other.slice(slice).toFloatMatrix();
         }
      });
   }

   public static NDArray fromJson(JsonEntry entry) {
      int[] shape = Util.ensureCorrectIndicies(entry.getValProperty("shape")
                                                    .asIntegerValueArray());
      FloatMatrix[] matrices = new FloatMatrix[shape[2] * shape[3]];
      JsonEntry array = entry.getProperty("data");
      zipWithIndex(array.elementIterator()).forEachRemaining(e -> {
         matrices[e.getValue()] = new FloatMatrix(e.getKey().getAsArray(Float.class));
         matrices[e.getValue()].reshape(shape[0], shape[1]);
      });
      return new DenseNDArray(matrices, shape);
   }

   public static void main(String[] args) throws Exception {
      NDArray x = DENSE.rand(2, 2, 2, 2);
      NDArray y = DENSE.rand(3, 3);

   }

   @Override
   public NDArray mmul(NDArray other) {
      int[] outShape = new int[]{
         rows(),
         other.columns(),
         Math.max(kernels(), other.kernels()),
         Math.max(channels(), other.channels())
      };

      if (this.order() <= 2) {
         if (other.order() <= 2) {
            return new DenseNDArray(data[0].mmul(other.toFloatMatrix()));
         }
         FloatMatrix[] matrices = new FloatMatrix[other.slices()];
         IntStream.range(0, other.slices())
                  .parallel()
                  .forEach(slice -> matrices[slice] = toFloatMatrix().mmul(other.slice(slice).toFloatMatrix()));
         return new DenseNDArray(matrices, outShape);
      }

      if (other.order() <= 2) {
         return mapFloatMatrix(Cast.as(getFactory().zeros(outShape)),
                               other,
                               FloatMatrix::mmul);
      }

      return mapTensorFloatMatrix(Cast.as(getFactory().zeros(outShape)),
                                  other,
                                  FloatMatrix::mmul);
   }

   @Override
   protected void setSlice(int slice, NDArray other) {
      checkArgument(other.rows() == rows() &&
                       other.columns() == columns(), "Slice size does not match (" +
                                                        rows() + ", " + columns() + ") != (" + other.rows() + ", " + other
                                                                                                                        .columns() + ")"
                   );
      setSlice(slice, other.toFloatMatrix());
   }

   @Override
   public NDArray add(NDArray other) {
      if (other.order() >= 2) {
         return mapDense(newZeroArray(), other, FloatMatrix::add);
      }
      return super.addi(other);
   }

   @Override
   public NDArray addi(NDArray other) {
      if (other.order() >= 2) {
         return mapDense(this, other, FloatMatrix::addi);
      }
      return super.addi(other);
   }

   @Override
   public NDArray copy() {
      return new DenseNDArray(this);
   }

   @Override
   public NDArray div(NDArray other) {
      if (other.order() >= 2) {
         return mapDense(newZeroArray(), other, FloatMatrix::div);
      }
      return super.divi(other);
   }

   @Override
   public NDArray divi(NDArray other) {
      if (other.order() >= 2) {
         return mapDense(this, other, FloatMatrix::divi);
      }
      return super.divi(other);
   }

   @Override
   public float get(int... indices) {
      switch (indices.length) {
         case 1:
            if (order() <= 2) {
               return data[0].get(indices[0]);
            }
            return get(Util.reverseIndex(indices[0], shape[0], shape[1], shape[2], shape[3]));
         case 2:
            return data[0].get(indices[0], indices[1]);
         case 3:
            return data[indices[2]].get(indices[0], indices[1]);
         case 4:
            return data[index(indices[2], indices[3])].get(indices[0], indices[1]);
      }
      throw new IllegalArgumentException("Too many indices");
   }

   @Override
   public NDArrayFactory getFactory() {
      return DENSE;
   }

   private int index(int i, int j) {
      return i + (shape[2] * j);
   }

   @Override
   public boolean isDense() {
      return true;
   }

   private NDArray mapDense(DenseNDArray out, NDArray other, BinaryOperator<FloatMatrix> operator) {
      if (other.order() == 2) {
         checkArgument(order() >= other.order(), "Cannot broadcast "
                                                    + Arrays.toString(other.shape()) + " to " + Arrays.toString(shape));
         return mapFloatMatrix(out, other, operator);
      } else if (other.order() > 2) {
         checkArgument(order() == other.order(), "Cannot broadcast "
                                                    + Arrays.toString(other.shape()) + " to " + Arrays.toString(shape));
         return mapTensorFloatMatrix(this, other, operator);
      }
      throw new IllegalArgumentException();
   }

   private NDArray mapFloatMatrix(DenseNDArray out, NDArray other, BinaryOperator<FloatMatrix> operator) {
      final FloatMatrix fm = other.toFloatMatrix();
      matrixStream().forEach(t -> out.setSlice(t.v1, operator.apply(t.v2, fm)));
      return out;
   }

   private NDArray mapTensorFloatMatrix(DenseNDArray out, NDArray tensor, BinaryOperator<FloatMatrix> operator) {
      checkArgument(slices() == tensor.slices(),
                    "Number of slices does not match. (" + slices() + ") != (" + tensor.slices() + ")");
      IntStream.range(0, data.length).parallel().forEach(slice -> {
         out.setSlice(slice, operator.apply(data[slice], tensor.slice(slice).toFloatMatrix()));
      });
      return out;
   }

   protected Stream<Tuple2<Integer, FloatMatrix>> matrixStream() {
      return IntStream.range(0, data.length)
                      .parallel().mapToObj(i -> $(i, data[i]));
   }

   @Override
   public NDArray mul(NDArray other) {
      if (other.order() >= 2) {
         return mapDense(this, other, FloatMatrix::mul);
      }
      return super.subi(other);
   }

   @Override
   public NDArray muli(NDArray other) {
      if (other.order() >= 2) {
         return mapDense(this, other, FloatMatrix::muli);
      }
      return super.subi(other);
   }

   private DenseNDArray newZeroArray() {
      return Cast.as(getFactory().zeros(shape));
   }

   @Override
   public NDArray rsub(NDArray other) {
      if (other.order() >= 2) {
         return mapDense(newZeroArray(), other, FloatMatrix::rsub);
      }
      return super.subi(other);
   }

   @Override
   public NDArray rsubi(NDArray other) {
      if (other.order() >= 2) {
         return mapDense(this, other, FloatMatrix::rsubi);
      }
      return super.subi(other);
   }

   @Override
   public NDArray set(int row, int column, int kernel, int channel, float value) {
      data[Util.index(kernel, dimension(KERNEL), channel, dimension(CHANNEL))].put(row, column, value);
      return this;
   }

   @Override
   public NDArray set(int row, int column, float value) {
      if (order() <= 2) {
         data[0].put(row, column, value);
         return this;
      }
      throw new IllegalStateException("Must give a slice index or kernel and channel.");
   }

   @Override
   public NDArray set(int index, float value) {
      if (order() <= 2) {
         data[0].put(index, value);
      } else {
         set(Util.reverseIndex(index,
                               dimension(ROW),
                               dimension(COLUMN),
                               dimension(KERNEL),
                               dimension(CHANNEL)), value);
      }
      return this;
   }

   private void setSlice(int slice, FloatMatrix matrix) {
      data[slice] = matrix;
   }


   @Override
   public NDArray slice(int kernel) {
      return new DenseNDArray(data[kernel]);
   }

   @Override
   public NDArray sub(NDArray other) {
      if (other.order() >= 2) {
         return mapDense(newZeroArray(), other, FloatMatrix::sub);
      }
      return super.subi(other);
   }

   @Override
   public NDArray subi(NDArray other) {
      if (other.order() >= 2) {
         return mapDense(this, other, FloatMatrix::subi);
      }
      return super.subi(other);
   }

   @Override
   public DoubleMatrix toDoubleMatrix() {
      Validation.checkState(isMatrix());
      return MatrixFunctions.floatToDouble(data[0]);
   }

   @Override
   public FloatMatrix toFloatMatrix() {
      if (isScalar()) {
         return FloatMatrix.scalar(data[0].get(0));
      }
      Validation.checkState(isMatrix());
      return data[0];
   }

   private String toString(FloatMatrix matrix) {
      return matrix.toString("%f", "[", "]", ", ", "],\n  [");
   }

   @Override
   public String toString() {
      StringBuilder builder = new StringBuilder("[[")
                                 .append(toString(data[0]));
      for (int i = 1; i < data.length; i++) {
         builder.append("]").append(System.lineSeparator())
                .append(" [")
                .append(toString(data[i]));
      }
      return builder.append("]]").toString();
   }


}//END OF DenseFloatNDArray
