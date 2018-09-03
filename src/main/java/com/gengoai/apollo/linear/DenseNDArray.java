package com.gengoai.apollo.linear;

import com.gengoai.Validation;
import com.gengoai.conversion.Cast;
import com.gengoai.tuple.Tuple2;
import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;
import org.jblas.ranges.IntervalRange;

import java.util.Arrays;
import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static com.gengoai.Validation.checkArgument;
import static com.gengoai.Validation.checkElementIndex;
import static com.gengoai.apollo.linear.NDArrayFactory.DENSE;
import static com.gengoai.tuple.Tuples.$;

/**
 * JBLAS FloatMatrix backed NDArray implementation
 *
 * @author David B. Bracewell
 */
public class DenseNDArray extends NDArray {
   private static final long serialVersionUID = 1L;
   private FloatMatrix[] data;

   /***
    * Creates a new DenseNDArray by wrapping a given <code>FloatMatrix</code>
    * @param matrix the matrix to wrap
    */
   public DenseNDArray(FloatMatrix matrix) {
      super(new int[]{matrix.rows, matrix.columns, 1, 1});
      this.data = new FloatMatrix[]{matrix};
   }

   DenseNDArray(DoubleMatrix matrix) {
      this(MatrixFunctions.doubleToFloat(matrix));
   }

   DenseNDArray(FloatMatrix[] matrix, int[] dims) {
      super(dims);
      this.data = matrix;
   }

   DenseNDArray(NDArray other) {
      super(other.shape());
      this.data = new FloatMatrix[other.numSlices()];
      IntStream.range(0, data.length).forEach(slice -> {
         if (other instanceof DenseNDArray) {
            this.data[slice] = other.getSlice(slice).toFloatMatrix().dup();
         } else {
            this.data[slice] = other.getSlice(slice).toFloatMatrix();
         }
      });
   }

   @Override
   public NDArray T() {
      FloatMatrix[] matrices = new FloatMatrix[numSlices()];
      for (int i = 0; i < data.length; i++) {
         matrices[i] = data[i].transpose();
      }
      return new DenseNDArray(matrices, new int[]{numCols(), numRows(), numKernels(), numChannels()});
   }

   @Override
   public NDArray add(NDArray other) {
      if (other.order() <= 2) {
         return mapFloatMatrix(newZeroArray(),
                               other,
                               FloatMatrix::addRowVector,
                               FloatMatrix::addColumnVector,
                               FloatMatrix::add);
      }
      return super.add(other);
   }

   @Override
   public NDArray addi(NDArray other) {
      if (other.order() <= 2) {
         return mapFloatMatrix(this,
                               other,
                               FloatMatrix::addiRowVector,
                               FloatMatrix::addiColumnVector,
                               FloatMatrix::addi);
      }
      return super.addi(other);
   }

   @Override
   public NDArray addi(double value) {
      matrixStream().forEach(t -> t.v2.addi((float) value));
      return this;
   }

   @Override
   public NDArray adjustIndexedValue(int sliceIndex, int matrixIndex, double value) {
      setIndexedValue(sliceIndex, matrixIndex, getIndexedValue(sliceIndex, matrixIndex) + value);
      return this;
   }

   private NDArray broadcastTensor(DenseNDArray out, NDArray other, BiFunction<FloatMatrix, FloatMatrix, FloatMatrix> operator) {
      final FloatMatrix oSlice = other.order() <= 2 ? other.toFloatMatrix() : null;

      forEachSlice((index, slice) -> {
         FloatMatrix otherFloat = oSlice;
         if (other.order() > 2) {
            otherFloat = other.getSlice(index).toFloatMatrix();
         }
         out.setSlice(index, operator.apply(slice.toFloatMatrix(), otherFloat));
      });
      return out;
   }

   @Override
   public NDArray copyData() {
      return new DenseNDArray(this);
   }

   @Override
   public NDArray div(NDArray other) {
      if (other.order() <= 2) {
         return mapFloatMatrix(newZeroArray(),
                               other,
                               FloatMatrix::divRowVector,
                               FloatMatrix::divColumnVector,
                               FloatMatrix::div);
      }
      return super.div(other);
   }

   @Override
   public NDArray divi(NDArray other) {
      if (other.order() <= 2) {
         return mapFloatMatrix(this,
                               other,
                               FloatMatrix::diviRowVector,
                               FloatMatrix::diviColumnVector,
                               FloatMatrix::divi);
      }
      return super.divi(other);
   }

   @Override
   public NDArray divi(double value) {
      matrixStream().forEach(t -> t.v2.divi((float) value));
      return this;
   }

   @Override
   public NDArray dot(NDArray other) {
      if (other instanceof DenseNDArray) {
         NDArray[] out = new NDArray[numSlices()];
         forEachSlice((si, n) -> out[si] = DENSE.scalar(n.toFloatMatrix()
                                                         .dot(other.getSlice(si).toFloatMatrix())));
         return DENSE.fromLayers(numKernels(), numChannels(), out);
      }
      return super.dot(other);
   }

   @Override
   public NDArrayFactory getFactory() {
      return DENSE;
   }

   @Override
   public float getIndexedValue(int sliceIndex, int matrixIndex) {
      return data[sliceIndex].get(matrixIndex);
   }

   @Override
   public NDArray getSlice(int index) {
      if (order() <= 2) {
         return this;
      }
      return new DenseNDArray(data[index]);
   }

   @Override
   public NDArray getVector(int index, Axis axis) {
      checkArgument(axis.isRowOrColumn(), () -> axisNotSupported(axis));
      if (dimension(axis) == 1) {
         return copyData();
      }
      checkElementIndex(index, dimension(axis));
      FloatMatrix[] fm = new FloatMatrix[numSlices()];
      int[] newShape = shape();
      newShape[axis.index] = 1;
      matrixStream().forEach(t -> {
         if (axis == Axis.ROW) {
            fm[t.v1] = t.v2.getRow(index);
         } else {
            fm[t.v1] = t.v2.getColumn(index);
         }
      });
      return new DenseNDArray(fm, newShape);
   }

   @Override
   public boolean isDense() {
      return true;
   }

   private NDArray mapDense(DenseNDArray out, NDArray other, BinaryOperator<FloatMatrix> operator) {
      if (other.order() <= 2) {
         checkArgument(order() >= other.order(), "Cannot broadcast "
                                                    + Arrays.toString(other.shape()) + " to " + Arrays.toString(
            shape()));
         return mapFloatMatrix(out, other, operator);
      } else if (other.order() > 2) {
         checkArgument(order() == other.order(), "Cannot broadcast "
                                                    + Arrays.toString(other.shape()) + " to " + Arrays.toString(
            shape()));
         return mapTensorFloatMatrix(this, other, operator);
      }
      throw new IllegalArgumentException();
   }

   private NDArray mapFloatMatrix(DenseNDArray out, NDArray other, BinaryOperator<FloatMatrix> operator) {
      final FloatMatrix fm = other.toFloatMatrix();
      matrixStream().forEach(t -> out.setSlice(t.v1, operator.apply(t.v2, fm)));
      return out;
   }

   private NDArray mapFloatMatrix(DenseNDArray out,
                                  NDArray other,
                                  BiFunction<FloatMatrix, FloatMatrix, FloatMatrix> rowOp,
                                  BiFunction<FloatMatrix, FloatMatrix, FloatMatrix> colOp,
                                  BiFunction<FloatMatrix, FloatMatrix, FloatMatrix> matrixOp
                                 ) {
      final FloatMatrix fm = other.toFloatMatrix();
      BiFunction<FloatMatrix, FloatMatrix, FloatMatrix> op;
      if (other.isScalar()) {
         op = matrixOp;
      } else if (other.order() == order()) {
         op = matrixOp;
      } else if (other.isRowVector()) {
         op = rowOp;
      } else if (other.isColumnVector()) {
         op = colOp;
      } else {
         throw new IllegalArgumentException("Incompatible: " + other.order() + " : " + order());
      }
      matrixStream().forEach(t -> out.setSlice(t.v1, op.apply(t.v2, fm)));
      return out;
   }

   private NDArray mapTensorFloatMatrix(DenseNDArray out, NDArray tensor, BinaryOperator<FloatMatrix> operator) {
      checkArgument(numSlices() == tensor.numSlices(),
                    "Number of slices does not match. (" + numSlices() + ") != (" + tensor.numSlices() + ")");
      IntStream.range(0, data.length).parallel().forEach(slice -> {
         out.setSlice(slice, operator.apply(data[slice], tensor.getSlice(slice).toFloatMatrix()));
      });
      return out;
   }

   private Stream<Tuple2<Integer, FloatMatrix>> matrixStream() {
      return IntStream.range(0, data.length).mapToObj(i -> $(i, data[i])).parallel();
   }

   @Override
   public NDArray mmul(NDArray other) {
      checkArgument(other.order() <= 2 || other.order() == order(),
                    "Order mismatch, cannot mmul order (" + this.order() + ") and order (" + other.order() + ")");
      int[] outShape = new int[]{
         numRows(),
         other.numCols(),
         Math.max(numKernels(), other.numKernels()),
         Math.max(numChannels(), other.numChannels())
      };
      if (this.order() <= 2) {
         return new DenseNDArray(data[0].mmul(other.toFloatMatrix()));
      }
      return broadcastTensor(Cast.as(getFactory().zeros(outShape)),
                             other,
                             FloatMatrix::mmul);
   }

   @Override
   public NDArray mul(NDArray other) {
      if (other.order() <= 2) {
         return mapFloatMatrix(newZeroArray(),
                               other,
                               FloatMatrix::mulRowVector,
                               FloatMatrix::mulColumnVector,
                               FloatMatrix::mul);
      }
      return super.mul(other);
   }

   @Override
   public NDArray muli(double value) {
      matrixStream().forEach(t -> t.v2.muli((float) value));
      return this;
   }

   @Override
   public NDArray muli(NDArray other) {
      if (other.order() <= 2) {
         return mapFloatMatrix(this,
                               other,
                               FloatMatrix::muliRowVector,
                               FloatMatrix::muliColumnVector,
                               FloatMatrix::muli);
      }
      return super.muli(other);
   }

   private DenseNDArray newZeroArray() {
      return Cast.as(getFactory().zeros(shape()));
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
   public NDArray reshape(int... dims) {
      super.reshape(dims);
      for (int i = 0; i < numSlices(); i++) {
         data[i].reshape(numRows(), numCols());
      }
      return this;
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
   public NDArray setIndexedValue(int sliceIndex, int matrixIndex, double value) {
      data[sliceIndex].put(matrixIndex, (float) value);
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
   public NDArray setVector(int index, Axis axis, NDArray other) {
      checkArgument(axis.isRowOrColumn(), "Axis (" + axis + ") is invalid.");
      matrixStream().forEach(t -> {
         if (axis == Axis.COLUMN) {
            t.v2.putColumn(index, other.getSlice(t.v1).toFloatMatrix());
         } else if (axis == Axis.ROW) {
            t.v2.putRow(index, other.getSlice(t.v1).toFloatMatrix());
         } else {
            throw new IllegalArgumentException("Axis (" + axis + ") is not supported.");
         }
      });
      return this;
   }

   @Override
   public NDArray slice(int iFrom, int iTo, int jFrom, int jTo) {
      FloatMatrix[] matrices = new FloatMatrix[numSlices()];
      matrixStream().forEach(t -> matrices[t.v1] = t.v2.get(new IntervalRange(iFrom, iTo),
                                                            new IntervalRange(jFrom, jTo)));
      return new DenseNDArray(matrices, new int[]{matrices[0].rows, matrices[0].columns, numKernels(), numChannels()});
   }

   @Override
   public NDArray slice(int from, int to) {
      FloatMatrix[] matrices = new FloatMatrix[numSlices()];
      matrixStream().forEach(t -> matrices[t.v1] = new FloatMatrix(Arrays.copyOfRange(t.v2.data, from, to)));
      return new DenseNDArray(matrices, new int[]{matrices[0].rows, matrices[0].columns, numKernels(), numChannels()});
   }

   @Override
   public NDArray sub(NDArray other) {
      if (other.order() <= 2) {
         return mapFloatMatrix(newZeroArray(),
                               other,
                               FloatMatrix::subRowVector,
                               FloatMatrix::subColumnVector,
                               FloatMatrix::sub);
      }
      return super.sub(other);
   }

   @Override
   public NDArray subi(double value) {
      matrixStream().forEach(t -> t.v2.subi((float) value));
      return this;
   }

   @Override
   public NDArray subi(NDArray other) {
      if (other.order() <= 2) {
         return mapFloatMatrix(this,
                               other,
                               FloatMatrix::subiRowVector,
                               FloatMatrix::subiColumnVector,
                               FloatMatrix::subi);
      }
      return super.subi(other);
   }

   @Override
   public DoubleMatrix toDoubleMatrix() {
      if (isScalar()) {
         return DoubleMatrix.scalar(data[0].get(0));
      }
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


}//END OF DenseFloatNDArray
