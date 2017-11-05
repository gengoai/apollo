package com.davidbracewell.apollo.linear;

/**
 * @author David B. Bracewell
 */
public abstract class ColumnMajorOrderNDArray extends NDArray {
   private static final long serialVersionUID = 1L;

   public final int toColumn(int index) {
      return index / numRows();
   }

   public final int toIndex(int i, int j) {
      return i + (numRows() * j);
   }

   public final int toRow(int index) {
      return index % numRows();
   }

}//END OF ColumnMajorOrderNDArray
