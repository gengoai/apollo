package com.davidbracewell.apollo.linear.sparse;

import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.apollo.linear.NDArrayFactory;
import com.davidbracewell.guava.common.base.Preconditions;
import lombok.NonNull;

import java.util.Iterator;
import java.util.Objects;
import java.util.function.Consumer;
import java.util.function.DoubleUnaryOperator;

/**
 * @author David B. Bracewell
 */
public class SparseDoubleNDArrayOld extends NDArray {
   private static final long serialVersionUID = 1L;

   private Sparse2dStorage storage;

   public SparseDoubleNDArrayOld(int nRows, int nCols) {
      this.storage = new Sparse2dStorage(nRows, nCols);
   }

   public SparseDoubleNDArrayOld(@NonNull Sparse2dStorage array) {
      this.storage = array;
   }

   @Override
   public NDArray compress() {
      this.storage.trimToSize();
      return this;
   }

   @Override
   public NDArray copyData() {
      SparseDoubleNDArrayOld copy = new SparseDoubleNDArrayOld(numRows(), numCols());
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
      return this.storage.get(toIndex(i,j));
   }

   @Override
   public NDArrayFactory getFactory() {
      return NDArrayFactory.SPARSE_DOUBLE_OLD;
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
   public NDArray mapSparse(@NonNull DoubleUnaryOperator operator) {
      NDArray toReturn = getFactory().zeros(numRows(), numCols());
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
      return storage.numColumns();
   }

   @Override
   public int numRows() {
      return storage.numRows();
   }

   @Override
   public NDArray set(int index, double value) {
      Preconditions.checkArgument(index >= 0 && index < length(), "Invalid index: " + index + " (" + length() + ")");
      this.storage.put(index, value);
      return this;
   }

   @Override
   public NDArray reshape(int numRows, int numCols) {
      storage.reshape(numRows, numCols);
      return this;
   }

   @Override
   public NDArray set(int r, int c, double value) {
      storage.put(r, c, value);
      return this;
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
   public NDArray zero() {
      storage.clear();
      return this;
   }


}// END OF SparseDoubleNDArray
