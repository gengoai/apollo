package com.davidbracewell.apollo.linalg;

import com.davidbracewell.guava.common.base.Preconditions;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import lombok.NonNull;

import java.util.List;

/**
 * @author David B. Bracewell
 */
public class SparseIntMatrix extends BaseMatrix {
   private final Int2DoubleOpenHashMap[] map;
   private final int numberOfRows;
   private final int numberOfColumns;

   public SparseIntMatrix(int numberOfRows, int numberOfColumns) {
      this.numberOfRows = numberOfRows;
      this.numberOfColumns = numberOfColumns;
      this.map = new Int2DoubleOpenHashMap[numberOfRows];
      for (int i = 0; i < numberOfRows; i++) {
         this.map[i] = new Int2DoubleOpenHashMap();
      }
   }

   public static Matrix from(@NonNull List<? extends Vector> vectors) {
      if (vectors.size() == 0) {
         return new SparseIntMatrix(0, 0);
      }
      SparseIntMatrix matrix = new SparseIntMatrix(vectors.size(), vectors.get(0).dimension());
      for (int i = 0; i < vectors.size(); i++) {
         matrix.setRow(i, vectors.get(i));
      }
      return matrix;
   }

   @Override
   public Matrix copy() {
      SparseIntMatrix mm = new SparseIntMatrix(numberOfRows, numberOfColumns);
      for (int i = 0; i < numberOfRows; i++) {
         mm.map[i].putAll(this.map[i]);
      }
      return mm;
   }

   @Override
   protected Matrix createNew(int numRows, int numColumns) {
      return new SparseIntMatrix(numberOfRows, numberOfColumns);
   }

   @Override
   public double get(int row, int column) {
      Preconditions.checkArgument(row < numberOfRows, "Row out of bounds");
      Preconditions.checkArgument(column < numberOfColumns, "Column out of bounds");
      return map[row].getOrDefault(column, 0d);
   }

   @Override
   public Matrix increment(int row, int col, double amount) {
      Preconditions.checkArgument(row < numberOfRows, "Row out of bounds");
      Preconditions.checkArgument(col < numberOfColumns, "Column out of bounds");
      map[row].put(col, amount + map[row].getOrDefault(col, 0d));
      return this;
   }

   @Override
   public int numberOfColumns() {
      return numberOfColumns;
   }

   @Override
   public int numberOfRows() {
      return numberOfRows;
   }

   @Override
   public Matrix set(int row, int column, double value) {
      Preconditions.checkArgument(row < numberOfRows, "Row out of bounds");
      Preconditions.checkArgument(column < numberOfColumns, "Column out of bounds");
      if (value == 0) {
         map[row].remove(column);
      } else {
         map[row].put(column, value);
      }
      return this;
   }
}// END OF SparseIntMatrix
