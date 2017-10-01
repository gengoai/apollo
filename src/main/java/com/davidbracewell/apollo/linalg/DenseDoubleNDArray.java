package com.davidbracewell.apollo.linalg;

import com.davidbracewell.conversion.Cast;
import com.davidbracewell.guava.common.base.Preconditions;
import lombok.NonNull;
import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;
import org.jblas.ranges.IntervalRange;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Objects;

/**
 * @author David B. Bracewell
 */
public class DenseDoubleNDArray implements NDArray, Serializable {
   private static final long serialVersionUID = 1L;
   private DoubleMatrix storage;

   public DenseDoubleNDArray(DoubleMatrix matrix) {
      this.storage = matrix;
   }

   @Override
   public NDArray T() {
      return new DenseDoubleNDArray(storage.transpose());
   }

   @Override
   public NDArray add(double scalar) {
      return new DenseDoubleNDArray(storage.add(scalar));
   }

   @Override
   public NDArray add(@NonNull NDArray other) {
      shape().checkDimensionMatch(other.shape());
      return new DenseDoubleNDArray(storage.add(other.toDoubleMatrix()));
   }

   @Override
   public NDArray add(@NonNull NDArray other, @NonNull Axis axis) {
      shape().checkDimensionMatch(other.shape(), axis.T());
      if (axis == Axis.ROW) {
         return new DenseDoubleNDArray(storage.addRowVector(other.toDoubleMatrix()));
      }
      return new DenseDoubleNDArray(storage.addColumnVector(other.toDoubleMatrix()));
   }

   @Override
   public NDArray addi(double scalar) {
      storage.addi(scalar);
      return this;
   }

   @Override
   public NDArray addi(@NonNull NDArray other) {
      shape().checkDimensionMatch(other.shape());
      storage.addi(other.toDoubleMatrix());
      return other;
   }

   @Override
   public NDArray addi(@NonNull NDArray other, @NonNull Axis axis) {
      shape().checkDimensionMatch(other.shape(), axis.T());
      if (axis == Axis.ROW) {
         storage.addiRowVector(other.toDoubleMatrix());
      } else {
         storage.addiColumnVector(other.toDoubleMatrix());
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
   public NDArray copy() {
      return new DenseDoubleNDArray(storage.dup());
   }

   @Override
   public NDArray div(double scalar) {
      return new DenseDoubleNDArray(storage.div(scalar));
   }

   @Override
   public NDArray div(@NonNull NDArray other) {
      shape().checkDimensionMatch(other.shape());
      return new DenseDoubleNDArray(storage.div(other.toDoubleMatrix()));
   }

   @Override
   public NDArray div(@NonNull NDArray other, @NonNull Axis axis) {
      shape().checkDimensionMatch(other.shape(), axis.T());
      if (axis == Axis.ROW) {
         return new DenseDoubleNDArray(storage.divRowVector(other.toDoubleMatrix()));
      }
      return new DenseDoubleNDArray(storage.divColumnVector(other.toDoubleMatrix()));
   }

   @Override
   public NDArray divi(double scalar) {
      storage.divi(scalar);
      return this;
   }

   @Override
   public NDArray divi(@NonNull NDArray other) {
      shape().checkDimensionMatch(other.shape());
      storage.divi(other.toDoubleMatrix());
      return this;
   }

   @Override
   public NDArray divi(@NonNull NDArray other, @NonNull Axis axis) {
      shape().checkDimensionMatch(other.shape(), axis.T());
      if (axis == Axis.ROW) {
         storage.diviRowVector(other.toDoubleMatrix());
      } else {
         storage.diviColumnVector(other.toDoubleMatrix());
      }
      return this;
   }

   @Override
   public double dot(@NonNull NDArray other) {
      return storage.dot(other.toDoubleMatrix());
   }

   @Override
   public boolean equals(Object o) {
      return o != null
                && o instanceof NDArray
                && shape().equals(Cast.<NDArray>as(o).shape())
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
      return NDArrayFactory.DENSE_DOUBLE;
   }

   @Override
   public NDArray getVector(int index, @NonNull Axis axis) {
      return new DenseDoubleNDArray(axis == Axis.ROW ? storage.getRow(index) : storage.getColumn(index));
   }

   @Override
   public int hashCode() {
      return Objects.hash(storage);
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
      return new DenseDoubleNDArray(axis == Axis.ROW ? storage.rowMaxs() : storage.columnMaxs());
   }

   @Override
   public double mean() {
      return storage.mean();
   }

   @Override
   public NDArray mean(@NonNull Axis axis) {
      return new DenseDoubleNDArray(axis == Axis.ROW ? storage.rowMeans() : storage.columnMeans());
   }

   @Override
   public double min() {
      return storage.min();
   }

   @Override
   public NDArray min(@NonNull Axis axis) {
      return new DenseDoubleNDArray(axis == Axis.ROW ? storage.rowMins() : storage.columnMins());
   }

   @Override
   public NDArray mmul(@NonNull NDArray other) {
      shape().checkCanMultiply(other.shape());
      return new DenseDoubleNDArray(storage.mmul(other.toDoubleMatrix()));
   }

   @Override
   public NDArray mul(double scalar) {
      return new DenseDoubleNDArray(storage.mul(scalar));
   }

   @Override
   public NDArray mul(@NonNull NDArray other) {
      shape().checkDimensionMatch(other.shape());
      return new DenseDoubleNDArray(storage.mul(other.toDoubleMatrix()));
   }

   @Override
   public NDArray mul(@NonNull NDArray other, @NonNull Axis axis) {
      shape().checkDimensionMatch(other.shape(), axis.T());
      if (axis == Axis.ROW) {
         return new DenseDoubleNDArray(storage.mulRowVector(other.toDoubleMatrix()));
      }
      return new DenseDoubleNDArray(storage.mulColumnVector(other.toDoubleMatrix()));
   }

   @Override
   public NDArray muli(double scalar) {
      storage.muli(scalar);
      return this;
   }

   @Override
   public NDArray muli(@NonNull NDArray other) {
      shape().checkDimensionMatch(other.shape());
      storage.muli(other.toDoubleMatrix());
      return this;
   }

   @Override
   public NDArray muli(@NonNull NDArray other, @NonNull Axis axis) {
      shape().checkDimensionMatch(other.shape(), axis.T());
      if (axis == Axis.ROW) {
         storage.muliRowVector(other.toDoubleMatrix());
      } else {
         storage.muliColumnVector(other.toDoubleMatrix());
      }
      return this;
   }

   @Override
   public NDArray neg() {
      return new DenseDoubleNDArray(storage.neg());
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
   public NDArray rdiv(double scalar) {
      return new DenseDoubleNDArray(storage.rdiv(scalar));
   }

   @Override
   public NDArray rdiv(@NonNull NDArray other) {
      shape().checkDimensionMatch(other.shape());
      return new DenseDoubleNDArray(storage.rdiv(other.toDoubleMatrix()));
   }

   @Override
   public NDArray rdivi(double scalar) {
      storage.rdivi(scalar);
      return this;
   }

   @Override
   public NDArray rdivi(@NonNull NDArray other) {
      shape().checkDimensionMatch(other.shape());
      storage.rdivi(other.toDoubleMatrix());
      return this;
   }

   @Override
   public NDArray rsub(double scalar) {
      return new DenseDoubleNDArray(storage.rsub(scalar));
   }

   @Override
   public NDArray rsub(@NonNull NDArray other) {
      return new DenseDoubleNDArray(storage.rsub(other.toDoubleMatrix()));
   }

   @Override
   public NDArray rsubi(double scalar) {
      storage.rsubi(scalar);
      return this;
   }

   @Override
   public NDArray rsubi(@NonNull NDArray other) {
      shape().checkDimensionMatch(other.shape());
      storage.rsubi(other.toDoubleMatrix());
      return this;
   }

   @Override
   public NDArray select(@NonNull NDArray predicate) {
      shape().checkDimensionMatch(predicate.shape());
      return new DenseDoubleNDArray(storage.select(predicate.toDoubleMatrix()));
   }

   @Override
   public NDArray selecti(@NonNull NDArray predicate) {
      shape().checkDimensionMatch(predicate.shape());
      storage.selecti(predicate.toDoubleMatrix());
      return this;
   }

   @Override
   public NDArray set(int index, double value) {
      storage.put(index, value);
      return this;
   }

   @Override
   public NDArray set(int r, int c, double value) {
      storage.put(r, c, value);
      return this;
   }

   @Override
   public NDArray setVector(int index, @NonNull NDArray vector, @NonNull Axis axis) {
      shape().checkDimensionMatch(vector.shape(), axis.T());
      if (axis == Axis.ROW) {
         storage.putRow(index, vector.toDoubleMatrix());
      } else {
         storage.putColumn(index, vector.toDoubleMatrix());
      }
      return this;
   }

   @Override
   public Shape shape() {
      return new Shape(storage.rows, storage.columns);
   }

   @Override
   public NDArray slice(int from, int to) {
      return new DenseDoubleNDArray(new DoubleMatrix(Arrays.copyOfRange(storage.data, from, to)));
   }

   @Override
   public NDArray slice(int iFrom, int iTo, int jFrom, int jTo) {
      return new DenseDoubleNDArray(storage.get(new IntervalRange(iFrom, iTo),
                                                new IntervalRange(jFrom, jTo)));
   }

   @Override
   public NDArray slice(@NonNull Axis axis, int... indexes) {
      if (axis == Axis.ROW) {
         return new DenseDoubleNDArray(storage.getRows(indexes));
      }
      return new DenseDoubleNDArray(storage.getColumns(indexes));
   }

   @Override
   public NDArray sub(double scalar) {
      return new DenseDoubleNDArray(storage.sub(scalar));
   }

   @Override
   public NDArray sub(@NonNull NDArray other) {
      shape().checkDimensionMatch(other.shape());
      return new DenseDoubleNDArray(storage.sub(other.toDoubleMatrix()));
   }

   @Override
   public NDArray sub(@NonNull NDArray other, @NonNull Axis axis) {
      shape().checkDimensionMatch(other.shape(), axis.T());
      if (axis == Axis.ROW) {
         return new DenseDoubleNDArray(storage.subRowVector(other.toDoubleMatrix()));
      }
      return new DenseDoubleNDArray(storage.subColumnVector(other.toDoubleMatrix()));
   }

   @Override
   public NDArray subi(double scalar) {
      storage.subi(scalar);
      return this;
   }

   @Override
   public NDArray subi(@NonNull NDArray other) {
      shape().checkDimensionMatch(other.shape());
      storage.subi(other.toDoubleMatrix());
      return this;
   }

   @Override
   public NDArray subi(@NonNull NDArray other, @NonNull Axis axis) {
      shape().checkDimensionMatch(other.shape(), axis.T());
      if (axis == Axis.ROW) {
         storage.subiRowVector(other.toDoubleMatrix());
      } else {
         storage.subiColumnVector(other.toDoubleMatrix());
      }
      return this;
   }

   @Override
   public NDArray sum(@NonNull Axis axis) {
      return axis == Axis.ROW
             ? new DenseDoubleNDArray(storage.rowSums())
             : new DenseDoubleNDArray(storage.columnSums());
   }

   @Override
   public double sum() {
      return storage.sum();
   }

   @Override
   public double[][] to2DArray() {
      return storage.toArray2();
   }

   @Override
   public double[] toArray() {
      return storage.toArray();
   }

   @Override
   public boolean[] toBooleanArray() {
      return storage.toBooleanArray();
   }

   @Override
   public DoubleMatrix toDoubleMatrix() {
      return storage;
   }

   @Override
   public FloatMatrix toFloatMatrix() {
      return storage.toFloat();
   }

   @Override
   public int[] toIntArray() {
      return storage.toIntArray();
   }

   @Override
   public String toString() {
      return storage.toString("%f", "[", "]", ", ", ", ");
   }

   private class DoubleIterator implements Iterator<NDArray.Entry> {
      int index = 0;

      @Override
      public boolean hasNext() {
         return index < storage.length;
      }

      @Override
      public Entry next() {
         Preconditions.checkElementIndex(index, storage.length);
         return new DoubleEntry(index++);
      }
   }

   private class DoubleEntry implements NDArray.Entry {
      private int i;
      private int j;
      private int index;

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
         storage.put(index, value);
      }
   }

}// END OF DenseDoubleNDArray
