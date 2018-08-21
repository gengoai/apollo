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
import static com.gengoai.apollo.linear.v2.Axis.CHANNEL;
import static com.gengoai.apollo.linear.v2.Axis.KERNEL;
import static com.gengoai.apollo.linear.v2.NDArrayFactory.DENSE;
import static com.gengoai.collection.Iterators.zipWithIndex;
import static com.gengoai.tuple.Tuples.$;

/**
 * The type Dense nd array.
 *
 * @author David B. Bracewell
 */
public class DenseNDArray extends NDArray {
   private static final long serialVersionUID = 1L;
   private FloatMatrix[] data;

   /**
    * Instantiates a new Dense nd array.
    *
    * @param matrix the matrix
    */
   DenseNDArray(FloatMatrix matrix) {
      super(new int[]{matrix.rows, matrix.columns, 1, 1});
      this.data = new FloatMatrix[]{matrix};
   }

   /**
    * Instantiates a new Dense nd array.
    *
    * @param matrix the matrix
    * @param dims   the dims
    */
   DenseNDArray(FloatMatrix[] matrix, int[] dims) {
      super(dims);
      this.data = matrix;
   }


   /**
    * Instantiates a new Dense nd array.
    *
    * @param other the other
    */
   DenseNDArray(NDArray other) {
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

   /**
    * From json nd array.
    *
    * @param entry the entry
    * @return the nd array
    */
   public static NDArray fromJson(JsonEntry entry) {
      int[] shape = ensureCorrectIndices(entry.getValProperty("shape").asIntegerValueArray());
      FloatMatrix[] matrices = new FloatMatrix[shape[2] * shape[3]];
      JsonEntry array = entry.getProperty("data");
      zipWithIndex(array.elementIterator()).forEachRemaining(e -> {
         matrices[e.getValue()] = new FloatMatrix(e.getKey().getAsArray(Float.class));
         matrices[e.getValue()].reshape(shape[0], shape[1]);
      });
      return new DenseNDArray(matrices, shape);
   }

   /**
    * The entry point of application.
    *
    * @param args the input arguments
    * @throws Exception the exception
    */
   public static void main(String[] args) throws Exception {
      NDArray x = DENSE.rand(1, 2, 1, 1);
      NDArray y = DENSE.rand(1, 2, 1, 1);
      System.out.printf("x=%s\n\ny=%s\n\n%s\n\n%s",
                        x, y, DENSE.vstack(x, y), DENSE.hstack(x, y)
                       );
   }

   @Override
   public NDArray add(NDArray other) {
      if (other.order() >= 2 || (other.order() == order() && order() < 2)) {
         return mapDense(newZeroArray(), other, FloatMatrix::add);
      }
      return super.add(other);
   }

   @Override
   public NDArray addi(NDArray other) {
      if (other.order() >= 2 || (other.order() == order() && order() < 2)) {
         return mapDense(this, other, FloatMatrix::addi);
      }
      return super.addi(other);
   }

   @Override
   public NDArray copy() {
      return new DenseNDArray(this).setWeight(getWeight())
                                   .setLabel(getLabel())
                                   .setPredicted(getPredicted());
   }

   @Override
   public NDArray div(NDArray other) {
      if (other.order() >= 2 || (other.order() == order() && order() < 2)) {
         return mapDense(newZeroArray(), other, FloatMatrix::div);
      }
      return super.div(other);
   }

   @Override
   public NDArray divi(NDArray other) {
      if (other.order() >= 2 || (other.order() == order() && order() < 2)) {
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
            return get(fromIndex(indices[0], shape));
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
      if (other.order() <= 2) {
         checkArgument(order() >= other.order(), "Cannot broadcast "
                                                    + Arrays.toString(other.shape()) + " to " + Arrays.toString(shape));
         NDArray o = mapFloatMatrix(out, other, operator);
         return o;
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

   /**
    * Matrix stream stream.
    *
    * @return the stream
    */
   protected Stream<Tuple2<Integer, FloatMatrix>> matrixStream() {
      return IntStream.range(0, data.length)
                      .parallel().mapToObj(i -> $(i, data[i]));
   }

   @Override
   public NDArray mmul(NDArray other) {
      int[] outShape = new int[]{
         numRows(),
         other.numCols(),
         Math.max(numKernels(), other.numKernels()),
         Math.max(numChannels(), other.numChannels())
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
   public NDArray mul(NDArray other) {
      if (other.order() >= 2 || (other.order() == order() && order() < 2)) {
         return mapDense(newZeroArray(), other, FloatMatrix::mul);
      }
      return super.mul(other);
   }

   @Override
   public NDArray muli(NDArray other) {
      if (other.order() >= 2 || (other.order() == order() && order() < 2)) {
         return mapDense(this, other, FloatMatrix::muli);
      }
      return super.muli(other);
   }

   private DenseNDArray newZeroArray() {
      return Cast.as(getFactory().zeros(shape));
   }

   @Override
   public NDArray rsub(NDArray other) {
      if (other.order() >= 2 || (other.order() == order() && order() < 2)) {
         return mapDense(newZeroArray(), other, FloatMatrix::rsub);
      }
      return super.rsub(other);
   }

   @Override
   public NDArray rsubi(NDArray other) {
      if (other.order() >= 2 || (other.order() == order() && order() < 2)) {
         return mapDense(newZeroArray(), other, FloatMatrix::rsubi);
      }
      return super.rsubi(other);
   }

   @Override
   public NDArray rdiv(NDArray other) {
      if (other.order() >= 2 || (other.order() == order() && order() < 2)) {
         return mapDense(newZeroArray(), other, FloatMatrix::rdiv);
      }
      return super.rdiv(other);
   }

   @Override
   public NDArray rdivi(NDArray other) {
      if (other.order() >= 2 || (other.order() == order() && order() < 2)) {
         return mapDense(newZeroArray(), other, FloatMatrix::rdivi);
      }
      return super.rdivi(other);
   }

   @Override
   public NDArray set(int row, int column, int kernel, int channel, float value) {
      data[toIndex(kernel, dimension(KERNEL),
                   channel, dimension(CHANNEL))].put(row, column, value);
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
         set(fromIndex(index, shape), value);
      }
      return this;
   }

   @Override
   protected void setSlice(int slice, NDArray newSlice) {
      checkArgument(newSlice.numRows() == numRows() &&
                       newSlice.numCols() == numCols(), "Slice size does not match (" +
                                                           numRows() + ", " + numCols() + ") != (" + newSlice.numRows() + ", " + newSlice
                                                                                                                                    .numCols() + ")"
                   );
      setSlice(slice, newSlice.toFloatMatrix());
   }

   private void setSlice(int slice, FloatMatrix matrix) {
      data[slice] = matrix;
   }


   @Override
   public NDArray slice(int index) {
      return new DenseNDArray(data[index]);
   }

   @Override
   public NDArray sub(NDArray other) {
      if (other.order() >= 2 || (other.order() == order() && order() < 2)) {
         return mapDense(newZeroArray(), other, FloatMatrix::sub);
      }
      return super.sub(other);
   }

   @Override
   public NDArray subi(NDArray other) {
      if (other.order() >= 2 || (other.order() == order() && order() < 2)) {
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
      for (int i = 1; i < Math.min(data.length, 10); i++) {
         builder.append("]").append(System.lineSeparator())
                .append(" [")
                .append(toString(data[i]));
      }

      if (data.length > 10) {
         if (data.length > 11) {
            builder.append("]")
                   .append(System.lineSeparator())
                   .append("  ...")
                   .append(System.lineSeparator());
         }
         builder.append(" [").append(toString(data[10]));
         for (int i = Math.max(11, data.length - 10);
              i < Math.max(Math.min(20, data.length), data.length);
              i++) {
            builder.append("]").append(System.lineSeparator())
                   .append(" [")
                   .append(toString(data[i]));
         }
      }

      return builder.append("]]").toString();
   }


}//END OF DenseFloatNDArray
