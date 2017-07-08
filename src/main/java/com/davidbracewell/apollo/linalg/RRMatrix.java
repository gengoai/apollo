package com.davidbracewell.apollo.linalg;

import com.davidbracewell.conversion.Cast;
import it.unimi.dsi.fastutil.ints.Int2ObjectMap;
import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.objects.ObjectIterator;
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
   private Int2ObjectOpenHashMap<Vector> matrix = new Int2ObjectOpenHashMap<>();

   private final int numberOfRows;
   private final int numberOfColums;

   public RRMatrix(int numberOfRows, int numberOfColums) {
      this.numberOfRows = numberOfRows;
      this.numberOfColums = numberOfColums;
   }

   public RRMatrix(RRMatrix other) {
      this.numberOfRows = other.numberOfRows;
      this.numberOfColums = other.numberOfColums;
      this.matrix.putAll(Cast.as(other));
   }


   @Override
   public int getRowDimension() {
      return numberOfRows;
   }

   @Override
   public int getColumnDimension() {
      return numberOfColums;
   }

   @Override
   public RealMatrix createMatrix(int i, int i1) throws NotStrictlyPositiveException {
      return new RRMatrix(i, i1);
   }

   @Override
   public RealMatrix copy() {
      return new RRMatrix(this);
   }

   public Iterator<Matrix.Entry> nonZeroIterator() {

      return new Iterator<Matrix.Entry>() {
         private ObjectIterator<Int2ObjectMap.Entry<Vector>> rowIterator = matrix.int2ObjectEntrySet().fastIterator();
         private Iterator<Vector.Entry> colIterator = null;
         private Matrix.Entry next = null;
         private int currentRow = -1;

         private boolean advance() {
            if (next == null) {
               while (colIterator == null || !colIterator.hasNext()) {
                  if (rowIterator.hasNext()) {
                     Int2ObjectMap.Entry<Vector> e = rowIterator.next();
                     currentRow = e.getIntKey();
                     colIterator = e.getValue().nonZeroIterator();
                  } else {
                     return false;
                  }
               }
               Vector.Entry e = colIterator.next();
               next = new Matrix.Entry(currentRow, e.getIndex(), e.getValue());
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
   public double getEntry(int i, int i1) throws OutOfRangeException {
      if (i < 0 || i > numberOfRows) {
         throw new OutOfRangeException(i, 0, numberOfRows);
      }
      if (i1 < 0 || i1 > numberOfColums) {
         throw new OutOfRangeException(i1, 0, numberOfColums);
      }
      if (matrix.containsKey(i)) {
         return matrix.get(i).get(i1);
      }
      return 0d;
   }

   @Override
   public void setEntry(int i, int i1, double v) throws OutOfRangeException {
      if (i < 0 || i > numberOfRows) {
         throw new OutOfRangeException(i, 0, numberOfRows);
      }
      if (i1 < 0 || i1 > numberOfColums) {
         throw new OutOfRangeException(i1, 0, numberOfColums);
      }
      if (matrix.containsKey(i)) {
         matrix.get(i).set(i1, v);
      } else {
         matrix.put(i, new SparseVector(numberOfColums).set(i1, v));
      }
   }
}//END OF RRMatrix
