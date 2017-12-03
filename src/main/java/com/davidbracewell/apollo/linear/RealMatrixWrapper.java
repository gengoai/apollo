package com.davidbracewell.apollo.linear;

import org.apache.commons.math3.exception.NotStrictlyPositiveException;
import org.apache.commons.math3.exception.OutOfRangeException;
import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class RealMatrixWrapper extends AbstractRealMatrix implements Serializable {
   private static final long serialVersionUID = 1L;
   final NDArray array;

   public RealMatrixWrapper(NDArray array) {
      this.array = array;
   }

   @Override
   public RealMatrix copy() {
      return new RealMatrixWrapper(array.copy());
   }

   @Override
   public RealMatrix createMatrix(int i, int i1) throws NotStrictlyPositiveException {
      return new RealMatrixWrapper(array.getFactory().zeros(i, i1));
   }

   @Override
   public int getColumnDimension() {
      return array.numCols();
   }

   @Override
   public double getEntry(int i, int i1) throws OutOfRangeException {
      return array.get(i, i1);
   }

   @Override
   public int getRowDimension() {
      return array.numRows();
   }

   @Override
   public void setEntry(int i, int i1, double v) throws OutOfRangeException {
      array.set(i, i1, v);
   }
}// END OF RealMatrixWrapper
