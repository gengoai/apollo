package com.davidbracewell.apollo.linalg;

/**
 * @author David B. Bracewell
 */
public interface MatrixFactory {

   MatrixFactory DENSE_DOUBLE = new MatrixFactory() {
      @Override
      public Matrix diag(Matrix m) {
         return DenseDoubleMatrix.diag(m);
      }

      @Override
      public Matrix diag(Matrix m, int rows, int columns) {
         return DenseDoubleMatrix.diag(m, rows, columns);
      }

      @Override
      public Matrix eye(int n) {
         return DenseDoubleMatrix.eye(n);
      }

      @Override
      public Matrix ones(int rows, int columns) {
         return DenseDoubleMatrix.ones(rows, columns);
      }

      @Override
      public Matrix ones(int length) {
         return DenseDoubleMatrix.ones(length);
      }

      @Override
      public Matrix rand(int rows, int columns) {
         return DenseDoubleMatrix.rand(rows, columns);
      }

      @Override
      public Matrix rand(int length) {
         return DenseDoubleMatrix.rand(length);
      }

      @Override
      public Matrix randn(int rows, int columns) {
         return DenseDoubleMatrix.randn(rows, columns);
      }

      @Override
      public Matrix randn(int length) {
         return DenseDoubleMatrix.randn(length);
      }

      @Override
      public Matrix scalar(double value) {
         return DenseDoubleMatrix.scalar(value);
      }

      @Override
      public Matrix zeros(int rows, int columns) {
         return DenseDoubleMatrix.zeros(rows, columns);
      }

      @Override
      public Matrix zeros(int length) {
         return DenseDoubleMatrix.zeros(length);
      }

      @Override
      public Matrix empty() {
         return DenseDoubleMatrix.empty();
      }
   };
   MatrixFactory DENSE_FLOAT = new MatrixFactory() {
      @Override
      public Matrix diag(Matrix m) {
         return DenseFloatMatrix.diag(m);
      }

      @Override
      public Matrix diag(Matrix m, int rows, int columns) {
         return DenseFloatMatrix.diag(m, rows, columns);
      }

      @Override
      public Matrix eye(int n) {
         return DenseFloatMatrix.eye(n);
      }

      @Override
      public Matrix ones(int rows, int columns) {
         return DenseFloatMatrix.ones(rows, columns);
      }

      @Override
      public Matrix ones(int length) {
         return DenseFloatMatrix.ones(length);
      }

      @Override
      public Matrix rand(int rows, int columns) {
         return DenseFloatMatrix.rand(rows, columns);
      }

      @Override
      public Matrix rand(int length) {
         return DenseFloatMatrix.rand(length);
      }

      @Override
      public Matrix randn(int rows, int columns) {
         return DenseFloatMatrix.randn(rows, columns);
      }

      @Override
      public Matrix randn(int length) {
         return DenseFloatMatrix.randn(length);
      }

      @Override
      public Matrix scalar(double value) {
         return DenseFloatMatrix.scalar(value);
      }

      @Override
      public Matrix zeros(int rows, int columns) {
         return DenseFloatMatrix.zeros(rows, columns);
      }

      @Override
      public Matrix zeros(int length) {
         return DenseFloatMatrix.zeros(length);
      }

      @Override
      public Matrix empty() {
         return DenseFloatMatrix.empty();
      }
   };

   Matrix diag(Matrix m);

   Matrix diag(Matrix m, int rows, int columns);

   Matrix eye(int n);

   Matrix ones(int rows, int columns);

   Matrix ones(int length);

   Matrix rand(int rows, int columns);

   Matrix rand(int length);

   Matrix randn(int rows, int columns);

   Matrix randn(int length);

   Matrix scalar(double value);

   Matrix zeros(int rows, int columns);

   Matrix zeros(int length);

   Matrix empty();


}// END OF MatrixFactory
