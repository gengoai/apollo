package com.gengoai.apollo.linear.dense;

import com.gengoai.Validation;
import com.gengoai.apollo.linear.Axis;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.conversion.Cast;
import lombok.NonNull;
import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;
import org.jblas.ranges.IntervalRange;

import java.util.Arrays;
import java.util.Iterator;
import java.util.Objects;

/**
 * @author David B. Bracewell
 */
public class DenseFloatNDArray extends NDArray {
   private static final long serialVersionUID = 1L;
   private FloatMatrix storage;

   public static void main(String[] args) throws Exception {
      DenseFloatNDArray a = new DenseFloatNDArray(FloatMatrix.rand(10, 10));
      a.iterator().forEachRemaining(System.out::println);
   }

   public DenseFloatNDArray(FloatMatrix matrix) {
      this.storage = matrix;
   }

   @Override
   public NDArray T() {
      return new DenseFloatNDArray(storage.transpose());
   }

   @Override
   public NDArray add(double scalar) {
      return new DenseFloatNDArray(storage.add((float) scalar));
   }

   @Override
   public NDArray add(@NonNull NDArray other) {
      return new DenseFloatNDArray(storage.add(other.toFloatMatrix()));
   }

   @Override
   public NDArray add(@NonNull NDArray other, @NonNull Axis axis) {
      if (axis == Axis.ROW) {
         return new DenseFloatNDArray(storage.addRowVector(other.toFloatMatrix()));
      }
      return new DenseFloatNDArray(storage.addColumnVector(other.toFloatMatrix()));
   }

   @Override
   public int size() {
      return length();
   }

   @Override
   public float[] toFloatArray() {
      return storage.data;
   }

   @Override
   public NDArray addi(double scalar) {
      storage.addi((float) scalar);
      return this;
   }

   @Override
   public NDArray addi(@NonNull NDArray other) {
      storage.addi(other.toFloatMatrix());
      return this;
   }

   @Override
   public NDArray addi(@NonNull NDArray other, @NonNull Axis axis) {
      if (axis == Axis.ROW) {
         storage.addiRowVector(other.toFloatMatrix());
      } else {
         storage.addiColumnVector(other.toFloatMatrix());
      }
      return this;
   }


   @Override
   public int[] argMax(Axis axis) {
      if (axis == Axis.ROW) {
         return storage.rowArgmaxs();
      }
      return storage.columnArgmaxs();
   }

   @Override
   public int[] argMin(Axis axis) {
      if (axis == Axis.ROW) {
         return storage.rowArgmins();
      }
      return storage.columnArgmins();
   }

   @Override
   protected NDArray copyData() {
      return new DenseFloatNDArray(storage.dup());
   }

   @Override
   public NDArray div(double scalar) {
      return new DenseFloatNDArray(storage.div((float) scalar));
   }

   @Override
   public NDArray div(@NonNull NDArray other) {
      return new DenseFloatNDArray(storage.div(other.toFloatMatrix()));
   }

   @Override
   public NDArray div(@NonNull NDArray other, @NonNull Axis axis) {
      if (axis == Axis.ROW) {
         return new DenseFloatNDArray(storage.divRowVector(other.toFloatMatrix()));
      }
      return new DenseFloatNDArray(storage.divColumnVector(other.toFloatMatrix()));
   }

   @Override
   public NDArray divi(double scalar) {
      storage.divi((float) scalar);
      return this;
   }

   @Override
   public NDArray divi(@NonNull NDArray other) {
      storage.divi(other.toFloatMatrix());
      return this;
   }

   @Override
   public NDArray divi(@NonNull NDArray other, @NonNull Axis axis) {
      if (axis == Axis.ROW) {
         storage.diviRowVector(other.toFloatMatrix());
      } else {
         storage.diviColumnVector(other.toFloatMatrix());
      }
      return this;
   }

   @Override
   public double dot(@NonNull NDArray other) {
      return storage.dot(other.toFloatMatrix());
   }

   @Override
   public boolean equals(Object o) {
      return o != null
                && o instanceof NDArray
                && length() == Cast.<NDArray>as(o).length()
                && Arrays.equals(Cast.<NDArray>as(o).toArray(), toArray());
   }

   @Override
   public double get(int index) {
      return storage.get(index);
   }

   @Override
   public double get(int i, int j) {
      return storage.get(i, j);
   }

   @Override
   public NDArrayFactory getFactory() {
      return NDArrayFactory.DENSE_FLOAT;
   }

   @Override
   public NDArray getVector(int index, @NonNull Axis axis) {
      return new DenseFloatNDArray(axis == Axis.ROW ? storage.getRow(index) : storage.getColumn(index));
   }

   @Override
   public int hashCode() {
      return Objects.hash(storage);
   }

   @Override
   public boolean isSparse() {
      return false;
   }

   @Override
   public Iterator<Entry> iterator() {
      return new DoubleIterator();
   }

   @Override
   public int length() {
      return storage.length;
   }

   @Override
   public double max() {
      return storage.max();
   }

   @Override
   public NDArray max(@NonNull Axis axis) {
      return new DenseFloatNDArray(axis == Axis.ROW ? storage.rowMaxs() : storage.columnMaxs());
   }

   @Override
   public double mean() {
      return storage.mean();
   }

   @Override
   public NDArray mean(@NonNull Axis axis) {
      return new DenseFloatNDArray(axis == Axis.ROW ? storage.rowMeans() : storage.columnMeans());
   }

   @Override
   public double min() {
      return storage.min();
   }

   @Override
   public NDArray min(@NonNull Axis axis) {
      return new DenseFloatNDArray(axis == Axis.ROW ? storage.rowMins() : storage.columnMins());
   }

   @Override
   public NDArray mmul(@NonNull NDArray other) {
      return new DenseFloatNDArray(storage.mmul(other.toFloatMatrix()));
   }

   @Override
   public NDArray mul(double scalar) {
      return new DenseFloatNDArray(storage.mul((float) scalar));
   }

   @Override
   public NDArray mul(@NonNull NDArray other) {
      return new DenseFloatNDArray(storage.mul(other.toFloatMatrix()));
   }

   @Override
   public NDArray mul(@NonNull NDArray other, @NonNull Axis axis) {
      if (axis == Axis.ROW) {
         return new DenseFloatNDArray(storage.mulRowVector(other.toFloatMatrix()));
      }
      return new DenseFloatNDArray(storage.mulColumnVector(other.toFloatMatrix()));
   }

   @Override
   public NDArray muli(double scalar) {
      storage.muli((float) scalar);
      return this;
   }

   @Override
   public NDArray muli(@NonNull NDArray other) {
      storage.muli(other.toFloatMatrix());
      return this;
   }

   @Override
   public NDArray muli(@NonNull NDArray other, @NonNull Axis axis) {
      if (axis == Axis.ROW) {
         storage.muliRowVector(other.toFloatMatrix());
      } else {
         storage.muliColumnVector(other.toFloatMatrix());
      }
      return this;
   }

   @Override
   public NDArray neg() {
      return new DenseFloatNDArray(storage.neg());
   }

   @Override
   public NDArray negi() {
      storage.negi();
      return this;
   }

   @Override
   public double norm1() {
      return storage.norm1();
   }

   @Override
   public double norm2() {
      return storage.norm2();
   }

   @Override
   public int numCols() {
      return storage.columns;
   }

   @Override
   public int numRows() {
      return storage.rows;
   }

   @Override
   public NDArray rdiv(double scalar) {
      return new DenseFloatNDArray(storage.rdiv((float) scalar));
   }

   @Override
   public NDArray rdiv(@NonNull NDArray other) {
      return new DenseFloatNDArray(storage.rdiv(other.toFloatMatrix()));
   }

   @Override
   public NDArray rdivi(double scalar) {
      storage.rdivi((float) scalar);
      return this;
   }

   @Override
   public NDArray rdivi(@NonNull NDArray other) {
      storage.rdivi(other.toFloatMatrix());
      return this;
   }

   @Override
   public NDArray reshape(int numRows, int numCols) {
      storage.reshape(numRows, numCols);
      return this;
   }

   @Override
   public NDArray rsub(double scalar) {
      return new DenseFloatNDArray(storage.rsub((float) scalar));
   }

   @Override
   public NDArray rsub(@NonNull NDArray other) {
      return new DenseFloatNDArray(storage.rsub(other.toFloatMatrix()));
   }

   @Override
   public NDArray rsubi(double scalar) {
      storage.rsubi((float) scalar);
      return this;
   }

   @Override
   public NDArray rsubi(@NonNull NDArray other) {
      storage.rsubi(other.toFloatMatrix());
      return this;
   }

   @Override
   public NDArray select(@NonNull NDArray predicate) {
      return new DenseFloatNDArray(storage.select(predicate.toFloatMatrix()));
   }

   @Override
   public NDArray selecti(@NonNull NDArray predicate) {
      storage.selecti(predicate.toFloatMatrix());
      return this;
   }

   @Override
   public NDArray set(int index, double value) {
      storage.put(index, (float) value);
      return this;
   }

   @Override
   public NDArray set(int r, int c, double value) {
      storage.put(r, c, (float) value);
      return this;
   }

   @Override
   public NDArray setVector(int index, @NonNull NDArray vector, @NonNull Axis axis) {
      if (axis == Axis.ROW) {
         storage.putRow(index, vector.toFloatMatrix());
      } else {
         storage.putColumn(index, vector.toFloatMatrix());
      }
      return this;
   }

   @Override
   public NDArray slice(int from, int to) {
      return new DenseFloatNDArray(new FloatMatrix(Arrays.copyOfRange(storage.data, from, to)));
   }

   @Override
   public NDArray slice(int iFrom, int iTo, int jFrom, int jTo) {
      return new DenseFloatNDArray(storage.get(new IntervalRange(iFrom, iTo),
                                               new IntervalRange(jFrom, jTo)));
   }

   @Override
   public NDArray slice(@NonNull Axis axis, int... indexes) {
      if (axis == Axis.ROW) {
         return new DenseFloatNDArray(storage.getRows(indexes));
      }
      return new DenseFloatNDArray(storage.getColumns(indexes));
   }

   @Override
   public Iterator<Entry> sparseColumnIterator(int column) {
      return new Iterator<Entry>() {
         int row = 0;

         private boolean advance() {
            while (row < numRows() && get(row, column) == 0) {
               row++;
            }
            return row < numRows();
         }

         @Override
         public boolean hasNext() {
            return advance();
         }

         @Override
         public Entry next() {
            advance();
            DoubleEntry de = new DoubleEntry(toIndex(row, column));
            row++;
            return de;
         }
      };
   }

   @Override
   public Iterator<Entry> sparseRowIterator(int row) {
      return new Iterator<Entry>() {
         int col = 0;

         private boolean advance() {
            while (col < numCols() && get(row, col) == 0) {
               col++;
            }
            return col < numCols();
         }

         @Override
         public boolean hasNext() {
            return advance();
         }

         @Override
         public Entry next() {
            advance();
            DoubleEntry de = new DoubleEntry(toIndex(row, col));
            col++;
            return de;
         }
      };
   }

   @Override
   public NDArray sub(double scalar) {
      return new DenseFloatNDArray(storage.sub((float) scalar));
   }

   @Override
   public NDArray sub(@NonNull NDArray other) {
      return new DenseFloatNDArray(storage.sub(other.toFloatMatrix()));
   }

   @Override
   public NDArray sub(@NonNull NDArray other, @NonNull Axis axis) {
      if (axis == Axis.ROW) {
         return new DenseFloatNDArray(storage.subRowVector(other.toFloatMatrix()));
      }
      return new DenseFloatNDArray(storage.subColumnVector(other.toFloatMatrix()));
   }

   @Override
   public NDArray subi(double scalar) {
      storage.subi((float) scalar);
      return this;
   }

   @Override
   public NDArray subi(@NonNull NDArray other) {
      storage.subi(other.toFloatMatrix());
      return this;
   }

   @Override
   public NDArray subi(@NonNull NDArray other, @NonNull Axis axis) {
      if (axis == Axis.ROW) {
         storage.subiRowVector(other.toFloatMatrix());
      } else {
         storage.subiColumnVector(other.toFloatMatrix());
      }
      return this;
   }

   @Override
   public NDArray sum(@NonNull Axis axis) {
      return axis == Axis.ROW
             ? new DenseFloatNDArray(storage.rowSums())
             : new DenseFloatNDArray(storage.columnSums());
   }

   @Override
   public double sum() {
      return storage.sum();
   }


   @Override
   public boolean[] toBooleanArray() {
      return storage.toBooleanArray();
   }

   @Override
   public DoubleMatrix toDoubleMatrix() {
      return MatrixFunctions.floatToDouble(storage);
   }

   @Override
   public FloatMatrix toFloatMatrix() {
      return storage;
   }

   @Override
   public int[] toIntArray() {
      return storage.toIntArray();
   }

   @Override
   public String toString() {
      return Arrays.toString(storage.data);
   }

   private class DoubleIterator implements Iterator<Entry> {
      int index = 0;

      @Override
      public boolean hasNext() {
         return index < storage.length;
      }

      @Override
      public Entry next() {
         Validation.checkElementIndex(index, storage.length);
         return new DoubleEntry(index++);
      }
   }

   private class DoubleEntry implements Entry {
      private int i;
      private int j;
      private int index;

      @Override
      public String toString() {
         return "DoubleEntry[" + i + "," + j + "]=" + getValue();
      }

      public DoubleEntry(int index) {
         this.index = index;
         this.i = storage.indexRows(index);
         this.j = storage.indexColumns(index);
      }


      @Override
      public int getI() {
         return i;
      }

      @Override
      public int getIndex() {
         return index;
      }

      @Override
      public int getJ() {
         return j;
      }

      @Override
      public double getValue() {
         return storage.get(index);
      }

      @Override
      public void setValue(double value) {
         storage.put(index, (float) value);
      }
   }
}// END OF DenseDoubleNDArray
