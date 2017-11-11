package com.davidbracewell.apollo.linear.sparse;

import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.apollo.linear.NDArrayFactory;
import com.davidbracewell.apollo.linear.Shape;
import com.davidbracewell.apollo.linear.Subscript;
import com.davidbracewell.guava.common.base.Preconditions;
import lombok.NonNull;

import java.util.Iterator;
import java.util.Objects;
import java.util.function.Consumer;
import java.util.function.DoubleUnaryOperator;

/**
 * @author David B. Bracewell
 */
public class SparseDoubleNDArray extends NDArray {
   private static final long serialVersionUID = 1L;

   private Sparse2dStorage storage;

   public SparseDoubleNDArray(int nRows, int nCols) {
      this.storage = new Sparse2dStorage(Shape.shape(nRows, nCols));
   }

   public SparseDoubleNDArray(@NonNull Shape shape) {
      this.storage = new Sparse2dStorage(shape);
   }

   public SparseDoubleNDArray(@NonNull Sparse2dStorage array) {
      this.storage = array;
   }

   @Override
   public NDArray compress() {
      this.storage.trimToSize();
      return this;
   }

   @Override
   public NDArray copyData() {
      SparseDoubleNDArray copy = new SparseDoubleNDArray(numRows(), numCols());
      storage.forEachPair((index, value) -> {
         copy.storage.put(index, value);
         return true;
      });
      return copy;
   }

   @Override
   public void forEachSparse(@NonNull Consumer<Entry> consumer) {
      storage.forEach(consumer);
   }

   @Override
   public double get(int dimension) {
      return this.storage.get(dimension);
   }

   @Override
   public double get(int i, int j) {
      return this.storage.get(shape().colMajorIndex(i, j));
   }

   @Override
   public NDArrayFactory getFactory() {
      return NDArrayFactory.SPARSE_DOUBLE;
   }

   @Override
   public int hashCode() {
      return Objects.hash(storage);
   }

   @Override
   public boolean isSparse() {
      return true;
   }

   @Override
   public Iterator<Entry> iterator() {
      return storage.iterator();
   }

   @Override
   public int length() {
      return storage.getShape().length();
   }

   @Override
   public NDArray mapSparse(@NonNull DoubleUnaryOperator operator) {
      NDArray toReturn = getFactory().zeros(shape());
      storage.forEach(entry -> toReturn.set(entry.getIndex(), operator.applyAsDouble(entry.getValue())));
      return toReturn;
   }

   @Override
   public NDArray mmul(@NonNull NDArray other) {
      NDArray toReturn = getFactory().zeros(numRows(), other.numCols());
      storage.forEach(entry -> other.sparseRowIterator(entry.getJ())
                                    .forEachRemaining(e2 -> toReturn.increment(entry.getI(),
                                                                               e2.getJ(),
                                                                               e2.getValue() * entry.getValue())
                                                     ));
      return toReturn;
   }

   @Override
   public int numCols() {
      return shape().j;
   }

   @Override
   public int numRows() {
      return shape().i;
   }

   @Override
   public NDArray set(int index, double value) {
      Preconditions.checkArgument(index >= 0 && index < length(), "Invalid index: " + index + " (" + length() + ")");
      this.storage.put(index, value);
      return this;
   }

   @Override
   public NDArray set(@NonNull Subscript subscript, double value) {
      shape().checkSubscript(subscript);
      this.storage.put(subscript.i, subscript.j, value);
      return this;
   }


   @Override
   public NDArray set(int r, int c, double value) {
      return set(shape().colMajorIndex(r, c), value);
   }

   @Override
   public Shape shape() {
      return storage.getShape();
   }

   @Override
   public int size() {
      return storage.size();
   }

   @Override
   public Iterator<NDArray.Entry> sparseColumnIterator(int column) {
      return storage.sparseColumn(column);
   }

   @Override
   public Iterator<Entry> sparseIterator() {
      return storage.sparseIterator();
   }

   @Override
   public Iterator<Entry> sparseRowIterator(int row) {
      return storage.sparseRow(row);
   }

   @Override
   public double sum() {
      return storage.sum();
   }

   @Override
   public double[][] to2DArray() {
      final double[][] array = new double[shape().i][shape().j];
      storage.forEachPair((index, value) -> {
         Subscript ss = shape().fromColMajorIndex(index);
         array[ss.i][ss.j] = value;
         return true;
      });
      return array;
   }

   @Override
   public double[] toArray() {
      final double[] array = new double[length()];
      storage.forEachPair((index, value) -> {
         array[index] = value;
         return true;
      });
      return array;
   }


   @Override
   public NDArray zero() {
      storage.clear();
      return this;
   }


}// END OF SparseDoubleNDArray
