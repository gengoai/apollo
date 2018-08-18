package com.gengoai.apollo.linear.v2;

import com.gengoai.Copyable;
import com.gengoai.collection.Iterators;
import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;

import java.io.Serializable;
import java.util.Iterator;
import java.util.Objects;
import java.util.function.BinaryOperator;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.IntStream;

import static com.gengoai.Validation.checkArgument;

/**
 * @author David B. Bracewell
 */
public abstract class NDArray implements Copyable<NDArray>, Serializable {
   public static final int CHANNEL = 3;
   public static final int COLUMN = 1;
   public static final int KERNEL = 2;
   public static final int ROW = 0;

   public NDArray add(float scalar) {
      return copy().addi(scalar);
   }

   public NDArray add(NDArray other) {
      return copy().addi(other);
   }

   public NDArray add(NDArray other, int axis) {
      return copy().addi(other, axis);
   }

   public NDArray addi(float scalar) {
      if (scalar != 0) {
         denseIterator().forEachRemaining(e -> e.setValue(e.getValue() + scalar));
      }
      return this;
   }

   public NDArray addi(NDArray other) {
      if (other.isScalar()) {
         return addi(other.get(0));
      }
      if (other.isColumnVector()) {
         return addi(other, COLUMN);
      }
      if (other.isRowVector()) {
         return addi(other, ROW);
      }
      if (other.isMatrix()) {
         return broadcasti(other, (m1, m2) -> {
            m2.sparseIterator()
              .forEachRemaining(entry -> m1.increment(entry.getIndicies(), entry.getValue()));
            return m1;
         });
      }
      return mapiTensor(other, (m1, m2) -> {
         m2.sparseIterator()
           .forEachRemaining(entry -> m1.increment(entry.getIndicies(), entry.getValue()));
         return m1;
      });
   }

   public abstract NDArray addi(NDArray other, int axis);

   public abstract NDArray broadcast(NDArray ndArray, BinaryOperator<NDArray> operator);

   public NDArray broadcasti(NDArray ndarray, BinaryOperator<NDArray> operator) {
      checkArgument(ndarray.sliceLength() == sliceLength(),
                    "Length of each slice is not the same. (" + sliceLength() + ") != (" + ndarray.sliceLength() + ")");
      IntStream.range(0, slices()).forEach(slice -> operator.apply(slice(slice), ndarray));
      return this;
   }

   public NDArray decrement(int[] indices, float value) {
      return set(indices, get(indices) - value);
   }

   public NDArray decrement(int row, int column, int kernel, int channel, float value) {
      return decrement(new int[]{row, column, kernel, channel}, value);
   }

   public NDArray decrement(int row, int column, int kernel, float value) {
      return decrement(new int[]{row, column, kernel}, value);
   }

   public NDArray decrement(int row, int column, float value) {
      return decrement(new int[]{row, column}, value);
   }

   public NDArray decrement(int row, float value) {
      return decrement(new int[]{row}, value);
   }

   public Iterator<Entry> denseIterator() {
      return new Iterator<Entry>() {
         int sliceIndex = 0;
         int rowColIndex = 0;

         private boolean advance() {
            while (sliceIndex < slices() && rowColIndex >= sliceLength()) {
               sliceIndex++;
               rowColIndex = 0;
            }
            return sliceIndex < slices();
         }

         @Override
         public boolean hasNext() {
            return advance();
         }

         @Override
         public Entry next() {
            advance();
            int[] kernelChannel = Util.reverseSliceIndex(sliceIndex, dimension(KERNEL));
            int[] rowColumn = Util.reverseRowColumnIndex(rowColIndex, dimension(ROW));
            Entry entry = new Entry(rowColumn[0],
                                    rowColumn[1],
                                    kernelChannel[0],
                                    kernelChannel[1]);
            rowColIndex++;
            return entry;
         }


      };
   }

   public abstract int dimension(int index);

   public NDArray div(NDArray other) {
      return copy().divi(other);
   }

   public NDArray div(float value) {
      return copy().divi(value);
   }

   public NDArray divVector(NDArray other, int axis) {
      return copy().diviVector(other, axis);
   }

   public NDArray divi(float value) {
      if (value == 0f || Float.isNaN(value)) {
         denseIterator().forEachRemaining(e -> e.setValue(Float.NaN));
      } else if (value == Float.NEGATIVE_INFINITY) {
         denseIterator().forEachRemaining(e -> e.setValue(Float.NEGATIVE_INFINITY));
      } else if (value == Float.POSITIVE_INFINITY) {
         denseIterator().forEachRemaining(e -> e.setValue(Float.POSITIVE_INFINITY));
      } else {
         sparseIterator().forEachRemaining(e -> e.setValue(e.getValue() / value));
      }
      return this;
   }

   public NDArray divi(NDArray other) {
      if (other.isScalar()) {
         return divi(other.get(0));
      }
      if (other.isColumnVector()) {
         return diviVector(other, COLUMN);
      }
      if (other.isRowVector()) {
         return diviVector(other, ROW);
      }
      if (other.isMatrix()) {
         return broadcasti(other, (m1, m2) -> {
            m2.sparseIterator()
              .forEachRemaining(entry -> m1.set(entry.getIndicies(), m1.get(entry.getIndicies()) / entry.getValue()));
            return m1;
         });
      }
      return mapiTensor(other, (m1, m2) -> {
         m2.sparseIterator()
           .forEachRemaining(entry -> m1.set(entry.getIndicies(), m1.get(entry.getIndicies()) / entry.getValue()));
         return m1;
      });
   }

   public abstract NDArray diviVector(NDArray other, int axis);

   public abstract float get(int... indices);

   public NDArray increment(int[] indices, float value) {
      return set(indices, get(indices) + value);
   }

   public NDArray increment(int row, int column, int kernel, int i4, float value) {
      return increment(new int[]{row, column, kernel, i4}, value);
   }

   public NDArray increment(int row, int column, int kernel, float value) {
      return increment(new int[]{row, column, kernel}, value);
   }

   public NDArray increment(int row, int column, float value) {
      return increment(new int[]{row, column}, value);
   }

   public NDArray increment(int row, float value) {
      return increment(new int[]{row}, value);
   }

   public boolean isColumnVector() {
      return isMatrix() && dimension(ROW) > 1 && dimension(COLUMN) == 1;
   }

   public boolean isMatrix() {
      return dimension(KERNEL) == 1 && dimension(CHANNEL) == 1;
   }

   public boolean isRowVector() {
      return isMatrix() && dimension(ROW) == 1 && dimension(COLUMN) > 1;
   }

   public boolean isScalar() {
      return dimension(ROW) == 1 && dimension(COLUMN) == 1 &&
                dimension(KERNEL) == 1 && dimension(CHANNEL) == 1;
   }

   public boolean isVector() {
      return isRowVector() || isColumnVector();
   }

   public abstract NDArray mapi(DoubleUnaryOperator operator);

   public abstract NDArray mapi(NDArray other, int axis, DoubleBinaryOperator operator);

   public NDArray mapiTensor(NDArray tensor, BinaryOperator<NDArray> operator) {
      checkArgument(tensor.sliceLength() == sliceLength(),
                    "Length of each slice is not the same. (" + sliceLength() + ") != (" + tensor.sliceLength() + ")");
      checkArgument(slices() == tensor.slices(),
                    "Number of slices does not match. (" + slices() + ") != (" + tensor.slices() + ")");
      IntStream.range(0, slices()).forEach(slice -> operator.apply(slice(slice), tensor.slice(slice)));
      return this;
   }

   public abstract float max();

   public abstract NDArray max(int index);

   public abstract NDArray min(int index);

   public NDArray mul(NDArray other) {
      return copy().muli(other);
   }

   public abstract NDArray muli(float value);

   public abstract NDArray muli(NDArray other);

   public int order() {
      return (dimension(ROW) > 1 ? 1 : 0) +
                (dimension(COLUMN) > 1 ? 1 : 0) +
                (dimension(KERNEL) > 1 ? 1 : 0) +
                (dimension(CHANNEL) > 1 ? 1 : 0);
   }

   public abstract NDArray set(int[] indices, float value);

   public NDArray set(int row, int column, int kernel, int channel, float value) {
      return set(new int[]{row, column, kernel, channel}, value);
   }

   public NDArray set(int row, int column, int kernel, float value) {
      return set(new int[]{row, column, kernel}, value);
   }

   public NDArray set(int row, int column, float value) {
      return set(new int[]{row, column}, value);
   }

   public NDArray set(int row, float value) {
      return set(new int[]{row}, value);
   }

   public abstract int[] shape();

   public abstract NDArray slice(int kernel, int channel);

   public NDArray slice(int kernel) {
      return slice(kernel, 0);
   }

   public int sliceLength() {
      return dimension(ROW) * dimension(COLUMN);
   }

   public int slices() {
      return dimension(KERNEL) * dimension(CHANNEL);
   }

   public Iterator<Entry> sparseIterator() {
      return Iterators.filter(denseIterator(), e -> e.getValue() != 0f);
   }

   public NDArray sub(NDArray other) {
      return copy().subi(other);
   }

   public NDArray subi(NDArray other) {
      other.sparseIterator().forEachRemaining(entry -> decrement(entry.getIndicies(), entry.getValue()));
      return this;
   }

   public abstract float sum();

   public abstract NDArray sum(int index);

   public abstract DoubleMatrix toDoubleMatrix();

   public abstract FloatMatrix toFloatMatrix();

   public class Entry {
      final int row, column, kernel, channel;

      protected Entry(int row, int column, int kernel, int channel) {
         this.row = row;
         this.column = column;
         this.kernel = kernel;
         this.channel = channel;
      }

      @Override
      public boolean equals(Object obj) {
         if (this == obj) {return true;}
         if (obj == null || getClass() != obj.getClass()) {return false;}
         final Entry other = (Entry) obj;
         return Objects.equals(this.row, other.row)
                   && Objects.equals(this.column, other.column)
                   && Objects.equals(this.kernel, other.kernel)
                   && Objects.equals(this.channel, other.channel);
      }

      public int getChannel() {
         return channel;
      }

      public int getColumn() {
         return column;
      }

      public int getKernel() {
         return kernel;
      }

      public int getRow() {
         return row;
      }

      public float getValue() {
         return get(row, column, kernel, channel);
      }

      public void setValue(float value) {
         set(row, column, kernel, channel, value);
      }

      @Override
      public int hashCode() {
         return Objects.hash(row, column, kernel, channel);
      }

      @Override
      public String toString() {
         return "Entry[" +
                   "row=" + row +
                   ", column=" + column +
                   ", kernel=" + kernel +
                   ", channel=" + channel +
                   "]=" + getValue();
      }

      public int getIndex(int axis) {
         switch (axis) {
            case ROW:
               return getRow();
            case COLUMN:
               return getColumn();
            case KERNEL:
               return getKernel();
            case CHANNEL:
               return getChannel();
            default:
               throw new IllegalArgumentException(axis + " is an unkown axis");
         }
      }

      public int[] getIndicies() {
         return new int[]{
            getRow(),
            getColumn(),
            getKernel(),
            getChannel()
         };
      }

   }

}//END OF NDArray
