package com.davidbracewell.apollo.linalg;

import com.davidbracewell.guava.common.base.Preconditions;
import lombok.NonNull;
import org.apache.mahout.math.map.OpenIntIntHashMap;

import java.util.List;

/**
 * @author David B. Bracewell
 */
public class SparseIntMatrix extends BaseMatrix {
   private final OpenIntIntHashMap[] map;
   private final int numberOfRows;
   private final int numberOfColumns;

   public SparseIntMatrix(int numberOfRows, int numberOfColumns) {
      this.numberOfRows = numberOfRows;
      this.numberOfColumns = numberOfColumns;
      this.map = new OpenIntIntHashMap[numberOfRows];
      for (int i = 0; i < numberOfRows; i++) {
         this.map[i] = new OpenIntIntHashMap();
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
         final OpenIntIntHashMap cp = mm.map[i];
         this.map[i].forEachPair(cp::put);
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
      return map[row].get(column);
   }

   @Override
   public Matrix increment(int row, int col, double amount) {
      Preconditions.checkArgument(row < numberOfRows, "Row out of bounds");
      Preconditions.checkArgument(col < numberOfColumns, "Column out of bounds");
      map[row].put(col, (int) amount + map[row].get(col));
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
         map[row].removeKey(column);
      } else {
         map[row].put(column, (int)value);
      }
      return this;
   }
}// END OF SparseIntMatrix
