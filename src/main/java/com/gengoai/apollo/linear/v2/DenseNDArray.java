package com.gengoai.apollo.linear.v2;

import com.gengoai.Validation;
import com.gengoai.conversion.Cast;
import com.gengoai.tuple.Tuple2;
import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;

import java.util.Arrays;
import java.util.function.*;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static com.gengoai.Validation.checkArgument;
import static com.gengoai.tuple.Tuples.$;

/**
 * @author David B. Bracewell
 */
public class DenseNDArray extends NDArray {
   private static final long serialVersionUID = 1L;
   private int[] shape;
   private FloatMatrix[] data;


   public DenseNDArray(FloatMatrix matrix) {
      this.shape = new int[]{matrix.rows, matrix.columns, 1, 1};
      this.data = new FloatMatrix[]{matrix};
   }

   public DenseNDArray(FloatMatrix[] matrix, int[] dims) {
      this.shape = new int[]{1, 1, 1, 1};
      System.arraycopy(dims, 0, this.shape, 0, dims.length);
      this.data = matrix;
   }


   public DenseNDArray(NDArray other) {
      this.shape = other.shape();
      this.data = new FloatMatrix[other.dimension(KERNEL) * other.dimension(CHANNEL)];
      for (int kernel = 0; kernel < dimension(KERNEL); kernel++) {
         for (int channel = 0; channel < dimension(CHANNEL); channel++) {
            this.data[Util.sliceIndex(kernel, channel, dimension(CHANNEL))] = other.slice(kernel, channel)
                                                                                   .toFloatMatrix()
                                                                                   .dup();
         }
      }
   }

   public DenseNDArray(int... dimensions) {
      checkArgument(dimensions.length > 0 &&
                       dimensions.length <= 4, "Only supports 1 to 4 dimensions");
      this.shape = new int[]{1, 1, 1, 1};
      System.arraycopy(dimensions, 0, this.shape, 0, dimensions.length);
      data = new FloatMatrix[this.shape[2] * this.shape[3]];
      for (int i = 0; i < data.length; i++) {
         switch (dimensions.length) {
            case 1:
               data[i] = FloatMatrix.rand(dimensions[0]);
               break;
            default:
               data[i] = FloatMatrix.rand(dimensions[0], dimensions[1]);
         }
      }
   }

   public static void main(String[] args) throws Exception {
      DenseNDArray x = new DenseNDArray(2, 2, 2, 2);
      DenseNDArray y = new DenseNDArray(1, 2);

      System.out.println(x.slice(0));
      System.out.println(y.slice(0));
      System.out.println(x.diviVector(y, COLUMN).slice(0));
   }

   private NDArray mapiTensorFloatMatrix(NDArray tensor, BinaryOperator<FloatMatrix> operator) {
      checkArgument(tensor.sliceLength() == sliceLength(),
                    "Length of each slice is not the same. (" + sliceLength() + ") != (" + tensor.sliceLength() + ")");
      checkArgument(slices() == tensor.slices(),
                    "Number of slices does not match. (" + slices() + ") != (" + tensor.slices() + ")");
      IntStream.range(0, data.length).parallel().forEach(slice -> {
         operator.apply(data[slice], tensor.slice(slice).toFloatMatrix());
      });
      return this;
   }

   @Override
   public NDArray addi(NDArray other, int axis) {
      checkArgument(axis == ROW || axis == COLUMN, "Only ROW and COLUMN are supported");
      checkArgument(other.isVector(), "Only vectors are supported");
      if (axis == ROW) {
         return broadcastiFloatMatrix(other, FloatMatrix::addiRowVector);
      }
      return broadcastiFloatMatrix(other, FloatMatrix::addiColumnVector);
   }

   @Override
   public NDArray addi(float scalar) {
      matrixStream().forEach(e -> e.v2.addi(scalar));
      return this;
   }

   @Override
   public NDArray broadcast(NDArray ndArray, BinaryOperator<NDArray> operator) {
      FloatMatrix[] matrices = new FloatMatrix[slices()];
      for (int slice = 0; slice < slices(); slice++) {
         matrices[slice] = operator.apply(slice(slice), ndArray).toFloatMatrix();
      }
      return new DenseNDArray(matrices, shape);
   }

   private NDArray broadcastFloatMatrix(NDArray other, BinaryOperator<FloatMatrix> operator) {
      FloatMatrix[] matrices = new FloatMatrix[slices()];
      final FloatMatrix fm = other.toFloatMatrix();
      for (int slice = 0; slice < slices(); slice++) {
         matrices[slice] = operator.apply(data[slice], fm);
      }
      return new DenseNDArray(matrices, shape);
   }

   private NDArray broadcastiFloatMatrix(NDArray other, BinaryOperator<FloatMatrix> operator) {
      final FloatMatrix fm = other.toFloatMatrix();
      matrixStream().forEach(e -> operator.apply(e.v2, fm));
      return this;
   }

   @Override
   public NDArray addi(NDArray other) {
      if (other.order() == 2) {
         return broadcastiFloatMatrix(other, FloatMatrix::addi);
      } else if (other.order() > 2) {
         return mapiTensorFloatMatrix(other, FloatMatrix::addi);
      }
      return super.addi(other);
   }

   @Override
   public NDArray divi(float value) {
      matrixStream().forEach(e -> e.v2.divi(value));
      return this;
   }

   @Override
   public NDArray divi(NDArray other) {
      if (other.order() == 2) {
         return broadcastiFloatMatrix(other, FloatMatrix::divi);
      } else if (other.order() > 2) {
         return mapiTensorFloatMatrix(other, FloatMatrix::divi);
      }
      return super.divi(other);
   }

   @Override
   public NDArray diviVector(NDArray other, int axis) {
      checkArgument(axis == ROW || axis == COLUMN, "Only ROW and COLUMN are supported");
      checkArgument(other.isVector(), "Only vectors are supported");
      if (axis == ROW) {
         return broadcastiFloatMatrix(other, FloatMatrix::diviRowVector);
      }
      return broadcastiFloatMatrix(other, FloatMatrix::diviColumnVector);
   }


   private NDArray collapseAndProcess(int iIndex, Function<FloatMatrix, FloatMatrix> function,
                                      BiFunction<FloatMatrix, FloatMatrix, FloatMatrix> biFunction
                                     ) {
      int[] newShape = this.shape();
      newShape[Util.oppositeIndex(iIndex)] = 1;
      DenseNDArray toReturn = new DenseNDArray(newShape);
      switch (iIndex) {
         case ROW:
         case COLUMN:
            IntStream.range(0, data.length)
                     .parallel()
                     .forEach(i -> toReturn.data[i] = function.apply(data[i]));
            break;
         default:
            int jIndex = Util.oppositeIndex(iIndex);
            for (int i = 0; i < dimension(iIndex); i++) {
               toReturn.data[i] = data[index(iIndex, i, jIndex, 0)].dup();
               for (int j = 1; j < dimension(jIndex); j++) {
                  toReturn.data[i] = biFunction.apply(toReturn.data[i], data[index(iIndex, i, jIndex, j)]);
               }
            }
      }
      return toReturn;
   }

   @Override
   public NDArray copy() {
      return new DenseNDArray(this);
   }


   @Override
   public int dimension(int index) {
      return shape[index];
   }

   @Override
   public float get(int... indices) {
      switch (indices.length) {
         case 1:
            return data[0].get(indices[0]);
         case 2:
            return data[0].get(indices[0], indices[1]);
         case 3:
            return data[indices[2]].get(indices[0], indices[1]);
         case 4:
            return data[index(indices[2], indices[3])].get(indices[0], indices[1]);
      }
      throw new IllegalArgumentException("Too many indices");
   }

   private int index(int i, int j) {
      return i + (shape[2] * j);
   }

   private int index(int iIndex, int i, int jIndex, int j) {
      if (iIndex == KERNEL) {
         return index(i, j);
      }
      return index(j, i);
   }

   @Override
   public NDArray mapi(DoubleUnaryOperator operator) {
      IntStream.range(0, data.length).parallel().forEach(i -> {
         FloatMatrix m = data[i];
         for (int j = 0; j < data[i].length; j++) {
            m.put(j, (float) operator.applyAsDouble(m.get(j)));
         }
      });
      return this;
   }

   protected Stream<Tuple2<Integer, FloatMatrix>> matrixStream() {
      return IntStream.range(0, data.length)
                      .parallel().mapToObj(i -> $(i, data[i]));
   }

   @Override
   public NDArray mapi(NDArray other, int axis, DoubleBinaryOperator operator) {
      int oAxis = Util.oppositeIndex(axis);

      if (axis == ROW || axis == COLUMN) {
         matrixStream().forEach(tuple2 -> {
            int i = tuple2.v1;
            FloatMatrix matrix = tuple2.v2;
            for (int r = 0; r < matrix.rows; r++) {
               for (int c = 0; c < matrix.columns; c++) {
                  matrix.put(r, c,
                             (float) operator.applyAsDouble(matrix.get(r, c),
                                                            other.slice(i)
                                                                 .get(Util.selectRowCol(oAxis, r, c))));
               }
            }
         });
      } else if (axis == KERNEL || axis == CHANNEL) {

      } else {
         throw new IllegalArgumentException("Axis (" + axis + ") is invalid.");
      }
      return this;
   }

   @Override
   public float max() {
      return (float) IntStream.range(0, data.length)
                              .parallel()
                              .mapToDouble(i -> data[i].max())
                              .max().orElse(0d);
   }

   @Override
   public NDArray max(int index) {
      return collapseAndProcess(index, m -> index == ROW ? m.rowMaxs() : m.columnMaxs(), FloatMatrix::maxi);
   }

   @Override
   public NDArray min(int index) {
      return collapseAndProcess(index, m -> index == ROW ? m.rowMins() : m.columnMins(), FloatMatrix::mini);
   }

   @Override
   public NDArray muli(float value) {
      for (FloatMatrix aData : data) {
         aData.muli(value);
      }
      return this;
   }

   @Override
   public NDArray muli(NDArray other) {
      DenseNDArray of = Cast.as(other);
      for (int i = 0; i < data.length; i++) {
         data[i].muli(of.data[i]);
      }
      return this;
   }

   private void processMatrices(Consumer<FloatMatrix> consumer) {
      IntStream.range(0, data.length)
               .parallel()
               .forEach(i -> consumer.accept(data[i]));
   }

   @Override
   public NDArray set(int[] indices, float value) {
      switch (indices.length) {
         case 1:
            data[0].put(indices[0], (float) value);
            break;
         case 2:
            data[0].put(indices[0], indices[1], (float) value);
            break;
         case 3:
            data[indices[2]].put(indices[0], indices[1], (float) value);
            break;
         case 4:
            data[index(indices[2], indices[3])].put(indices[0], indices[1], (float) value);
            break;
         default:
            throw new IllegalArgumentException("Too many indices");
      }
      return this;
   }

   @Override
   public int[] shape() {
      return Arrays.copyOf(shape, shape.length);
   }

   @Override
   public NDArray slice(int kernel, int channel) {
      return new DenseNDArray(data[Util.sliceIndex(kernel, channel, shape[2])]);
   }

   @Override
   public NDArray subi(NDArray other) {
      if (other instanceof DenseNDArray) {
         DenseNDArray of = Cast.as(other);
         IntStream.range(0, data.length).parallel().forEach(i -> data[i].subi(of.data[i]));
      } else {
         super.subi(other);
      }
      return this;
   }

   @Override
   public float sum() {
      return (float) IntStream.range(0, data.length)
                              .parallel()
                              .mapToDouble(i -> data[i].max())
                              .sum();
   }

   @Override
   public NDArray sum(int index) {
      return collapseAndProcess(index, m -> index == ROW ? m.rowSums() : m.columnSums(), FloatMatrix::addi);
   }

   @Override
   public DoubleMatrix toDoubleMatrix() {
      Validation.checkState(isMatrix());
      return MatrixFunctions.floatToDouble(data[0]);
   }

   @Override
   public FloatMatrix toFloatMatrix() {
      Validation.checkState(isMatrix());
      return data[0];
   }

   @Override
   public String toString() {
      return Arrays.toString(data);
   }


}//END OF DenseFloatNDArray
