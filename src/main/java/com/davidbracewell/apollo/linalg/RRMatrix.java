package com.davidbracewell.apollo.linalg;

import org.apache.commons.math3.exception.NotStrictlyPositiveException;
import org.apache.commons.math3.exception.OutOfRangeException;
import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * @author David B. Bracewell
 */
public class RRMatrix extends AbstractRealMatrix {
   private final Vector[] matrix;

   private final int numberOfRows;
   private final int numberOfColums;

   public RRMatrix(int numberOfRows, int numberOfColumns) {
      this.numberOfRows = numberOfRows;
      this.numberOfColums = numberOfColumns;
      this.matrix = new Vector[numberOfRows];
      for (int i = 0; i < numberOfRows; i++) {
         this.matrix[i] = new SparseVector(numberOfColumns);
      }
   }

   public RRMatrix(RRMatrix other) {
      this.numberOfRows = other.numberOfRows;
      this.numberOfColums = other.numberOfColums;
      this.matrix = new Vector[numberOfRows];
      for (int i = 0; i < numberOfRows; i++) {
         this.matrix[i] = new SparseVector(other.getRow(i));
      }
   }

   @Override
   public RealMatrix copy() {
      return new RRMatrix(this);
   }

   @Override
   public RealMatrix createMatrix(int i, int i1) throws NotStrictlyPositiveException {
      return new RRMatrix(i, i1);
   }

   @Override
   public int getColumnDimension() {
      return numberOfColums;
   }

   @Override
   public double getEntry(int i, int i1) throws OutOfRangeException {
      if (i < 0 || i > numberOfRows) {
         throw new OutOfRangeException(i, 0, numberOfRows);
      }
      if (i1 < 0 || i1 > numberOfColums) {
         throw new OutOfRangeException(i1, 0, numberOfColums);
      }
      return matrix[i].get(i1);
   }

   @Override
   public int getRowDimension() {
      return numberOfRows;
   }

   public Iterator<Matrix.Entry> nonZeroIterator() {

      return new Iterator<Matrix.Entry>() {
         private int row = -1;
         private Iterator<Vector.Entry> colIterator = null;
         private Matrix.Entry next = null;

         private boolean advance() {
            if (next == null) {
               while ((row + 1) < matrix.length &&
                         (colIterator == null || !colIterator.hasNext())) {
                  row++;
                  colIterator = matrix[row].nonZeroIterator();
               }
               if (colIterator.hasNext()) {
                  Vector.Entry e = colIterator.next();
                  next = new Matrix.Entry(row, e.getIndex(), e.getValue());
               } else {
                  return false;
               }
            }
            return true;
         }

         @Override
         public boolean hasNext() {
            return advance();
         }

         @Override
         public Matrix.Entry next() {
            if (!advance()) {
               throw new NoSuchElementException();
            }
            Matrix.Entry toReturn = next;
            next = null;
            return toReturn;
         }
      };
   }

   @Override
   public void setEntry(int i, int i1, double v) throws OutOfRangeException {
      if (i < 0 || i > numberOfRows) {
         throw new OutOfRangeException(i, 0, numberOfRows);
      }
      if (i1 < 0 || i1 > numberOfColums) {
         throw new OutOfRangeException(i1, 0, numberOfColums);
      }
      matrix[i].set(i1, v);
   }
}//END OF RRMatrix
