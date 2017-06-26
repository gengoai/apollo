package com.davidbracewell.apollo.linalg;

import com.davidbracewell.guava.common.base.Preconditions;
import lombok.NonNull;
import org.jblas.ComplexDouble;
import org.jblas.DoubleMatrix;

/**
 * The type Dense matrix.
 *
 * @author David B. Bracewell
 */
public class DenseMatrix extends BaseMatrix {
   private static final long serialVersionUID = 1L;
   /**
    * The Matrix.
    */
   volatile DoubleMatrix matrix;

   /**
    * Instantiates a new Dense matrix.
    *
    * @param rowDimension    the row dimension
    * @param columnDimension the column dimension
    */
   public DenseMatrix(int rowDimension, int columnDimension) {
      this.matrix = new DoubleMatrix(rowDimension, columnDimension);
   }

   /**
    * Instantiates a new Dense matrix.
    *
    * @param rowDimension    the row dimension
    * @param columnDimension the column dimension
    */
   public DenseMatrix(int rowDimension, int columnDimension, double[] data) {
      this.matrix = new DoubleMatrix(rowDimension, columnDimension);
      for (int r = 0; r < rowDimension; r++) {
         for (int c = 0; c < columnDimension; c++) {
            this.matrix.put(r, c, data[rowDimension * c + r]);
         }
      }
   }


   /**
    * Instantiates a new Dense matrix.
    *
    * @param matrix the matrix
    */
   public DenseMatrix(double[][] matrix) {
      this.matrix = new DoubleMatrix(matrix);
   }

   public DenseMatrix(@NonNull ComplexDouble[][] matrix) {
      this.matrix = new DoubleMatrix(matrix.length, matrix[0].length);
      for (int r = 0; r < matrix.length; r++) {
         for (int c = 0; c < matrix[c].length; c++) {
            this.matrix.put(r, c, matrix[r][c].real());
         }
      }
   }

   /**
    * Instantiates a new Dense matrix.
    *
    * @param matrix the matrix
    */
   public DenseMatrix(@NonNull Matrix matrix) {
      this.matrix = new DoubleMatrix(matrix.numberOfRows(), matrix.numberOfColumns());
      matrix.forEachSparse(e -> this.matrix.put(e.row, e.column, e.value));
   }

   /**
    * Instantiates a new Dense matrix.
    *
    * @param matrix the matrix
    */
   public DenseMatrix(DoubleMatrix matrix) {
      this.matrix = matrix;
   }

   /**
    * Ones matrix.
    *
    * @param rowDimension    the row dimension
    * @param columnDimension the column dimension
    * @return the matrix
    */
   public static Matrix ones(int rowDimension, int columnDimension) {
      return new DenseMatrix(DoubleMatrix.ones(rowDimension, columnDimension));
   }

   public static Matrix random(int numberOfRows, int numberOfColumns) {
      return new DenseMatrix(DoubleMatrix.rand(numberOfRows, numberOfColumns));
   }

   public static Matrix random(int numberOfRows, int numberOfColumns, double min, double max) {
      DenseMatrix m = new DenseMatrix(DoubleMatrix.zeros(numberOfRows, numberOfColumns));
      for (int r = 0; r < numberOfRows; r++) {
         for (int c = 0; c < numberOfColumns; c++) {
            m.set(r, c, min + (max - min) * Math.random());
         }
      }
      return m;
   }

   /**
    * Unit matrix.
    *
    * @param N the n
    * @return the matrix
    */
   public static Matrix unit(int N) {
      return new DenseMatrix(DoubleMatrix.eye(N));
   }

   /**
    * Zeroes matrix.
    *
    * @param rowDimension    the row dimension
    * @param columnDimension the column dimension
    * @return the matrix
    */
   public static Matrix zeroes(int rowDimension, int columnDimension) {
      return new DenseMatrix(DoubleMatrix.zeros(rowDimension, columnDimension));
   }

   @Override
   public Matrix T() {
      return new DenseMatrix(matrix.transpose());
   }

   @Override
   public Matrix add(@NonNull Matrix m) {
      if (m instanceof DenseMatrix) {
         return new DenseMatrix(matrix.add(m.toDense().matrix));
      }
      return super.add(m);
   }

   @Override
   public Matrix addSelf(@NonNull Matrix other) {
      Preconditions.checkArgument(
         other.numberOfColumns() == numberOfColumns() && other.numberOfRows() == numberOfRows(),
         "Dimension Mismatch");
      if (other instanceof DenseMatrix) {
         matrix.addi(other.toDense().matrix);
      } else {
         super.addSelf(other);
      }
      return this;
   }

   public DoubleMatrix asDoubleMatrix() {
      return matrix;
   }

   @Override
   public Matrix copy() {
      return new DenseMatrix(toArray());
   }

   @Override
   protected Matrix createNew(int numRows, int numColumns) {
      return new DenseMatrix(numRows, numColumns);
   }

   @Override
   public double get(int row, int column) {
      return matrix.get(row, column);
   }

   @Override
   public Matrix increment(int row, int col, double amount) {
      set(row, col, get(row, col) + amount);
      return this;
   }

   @Override
   public Matrix incrementSelf(double value) {
      matrix.addi(value);
      return this;
   }


   @Override
   public boolean isDense() {
      return true;
   }

   @Override
   public Matrix multiply(@NonNull Matrix m) {
      Preconditions.checkArgument(numberOfColumns() == m.numberOfRows(), "Dimension Mismatch");
      return new DenseMatrix(matrix.mmul(m.toDense().matrix));
   }

   @Override
   public int numberOfColumns() {
      return matrix.columns;
   }

   @Override
   public int numberOfRows() {
      return matrix.rows;
   }

   @Override
   public Matrix scaleSelf(double value) {
      matrix.muli(value);
      return this;
   }

   @Override
   public Matrix scaleSelf(@NonNull Matrix other) {
      Preconditions.checkArgument(
         other.numberOfColumns() == numberOfColumns() && other.numberOfRows() == numberOfRows(),
         "Dimension Mismatch");
      if (other instanceof DenseMatrix) {
         matrix.muli(other.toDense().matrix);
      } else {
         super.scaleSelf(other);
      }
      return this;
   }

   @Override
   public Matrix set(int row, int column, double value) {
      matrix.put(row, column, value);
      return this;
   }

   @Override
   public Matrix subtract(@NonNull Matrix m) {
      if (m instanceof DenseMatrix) {
         return new DenseMatrix(matrix.sub(m.toDense().matrix));
      }
      return super.subtract(m);
   }

   @Override
   public Matrix subtractSelf(@NonNull Matrix other) {
      Preconditions.checkArgument(
         other.numberOfColumns() == numberOfColumns() && other.numberOfRows() == numberOfRows(),
         "Dimension Mismatch");
      if (other instanceof DenseMatrix) {
         matrix.subi(other.toDense().matrix);
      } else {
         super.subtractSelf(other);
      }
      return this;
   }

   @Override
   public double[][] toArray() {
      return matrix.toArray2();
   }

   @Override
   public DenseMatrix toDense() {
      return this;
   }


}// END OF DenseMatrix
