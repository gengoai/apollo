package com.davidbracewell.apollo.linalg;

import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;
import org.jblas.ranges.IntervalRange;

import java.io.Serializable;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoublePredicate;
import java.util.function.DoubleUnaryOperator;

/**
 * @author David B. Bracewell
 */
public class DenseFloatMatrix implements Matrix, Serializable {
   private static final long serialVersionUID = 1L;
   final FloatMatrix matrix;

   public DenseFloatMatrix(FloatMatrix m) {
      this.matrix = m;
   }

   public DenseFloatMatrix(int rows, int columns, float... values) {
      this.matrix = new FloatMatrix(rows, columns, values);
   }

   public DenseFloatMatrix(int rows, int columns) {
      this.matrix = new FloatMatrix(rows, columns);
   }

   public DenseFloatMatrix(int len) {
      this.matrix = new FloatMatrix(len);
   }

   public DenseFloatMatrix(float[] values) {
      this.matrix = new FloatMatrix(values);
   }

   public static DenseFloatMatrix diag(Matrix matrix) {
      return wrap(FloatMatrix.diag(matrix.toFloatMatrix()));
   }

   public static DenseFloatMatrix diag(Matrix matrix, int rows, int columns) {

      return wrap(FloatMatrix.diag(matrix.toFloatMatrix(), rows, columns));
   }

   public static DenseFloatMatrix empty() {
      return wrap(FloatMatrix.EMPTY);
   }

   public static DenseFloatMatrix eye(int n) {
      return wrap(FloatMatrix.eye(n));
   }

   public static DenseFloatMatrix ones(int rows, int columns) {
      return wrap(FloatMatrix.ones(rows, columns));
   }

   public static DenseFloatMatrix ones(int length) {
      return wrap(FloatMatrix.ones(length));
   }

   public static DenseFloatMatrix rand(int rows, int columns) {
      return wrap(FloatMatrix.rand(rows, columns));
   }

   public static DenseFloatMatrix rand(int length) {
      return wrap(FloatMatrix.rand(length));
   }

   public static DenseFloatMatrix randn(int rows, int columns) {
      return wrap(FloatMatrix.randn(rows, columns));
   }

   public static DenseFloatMatrix randn(int length) {
      return wrap(FloatMatrix.randn(length));
   }

   public static DenseFloatMatrix scalar(double value) {
      return wrap(FloatMatrix.scalar((float) value));
   }

   public static DenseFloatMatrix wrap(FloatMatrix fm) {
      return new DenseFloatMatrix(fm);
   }

   public static DenseFloatMatrix zeros(int rows, int columns) {
      return wrap(FloatMatrix.zeros(rows, columns));
   }

   public static DenseFloatMatrix zeros(int length) {
      return wrap(FloatMatrix.zeros(length));
   }

   @Override
   public Matrix add(Matrix other) {
      return wrap(matrix.add(other.toFloatMatrix()));
   }

   @Override
   public Matrix add(double scalar) {
      return wrap(matrix.add((float) scalar));
   }

   @Override
   public Matrix addColumnVector(Matrix columnVector) {
      return wrap(matrix.addColumnVector(columnVector.toFloatMatrix()));
   }

   @Override
   public Matrix addRowVector(Matrix rowVector) {
      return wrap(matrix.addRowVector(rowVector.toFloatMatrix()));
   }

   @Override
   public Matrix addi(Matrix other) {
      matrix.addi(other.toFloatMatrix());
      return this;
   }

   @Override
   public Matrix addi(double scalar) {
      matrix.addi((float) scalar);
      return this;
   }

   @Override
   public Matrix addiColumnVector(Matrix columnVector) {
      matrix.addiColumnVector(columnVector.toFloatMatrix());
      return this;
   }

   @Override
   public Matrix addiRowVector(Matrix rowVector) {
      matrix.addiRowVector(rowVector.toFloatMatrix());
      return this;
   }

   @Override
   public int[] columnArgMaxs() {
      return matrix.columnArgmaxs();
   }

   @Override
   public int[] columnArgMins() {
      return matrix.columnArgmins();
   }

   @Override
   public Matrix columnMaxs() {
      return wrap(matrix.columnMaxs());
   }

   @Override
   public Matrix columnMeans() {
      return wrap(matrix.rowMeans());
   }

   @Override
   public Matrix columnMins() {
      return wrap(matrix.columnMins());
   }

   @Override
   public Matrix columnSums() {
      return wrap(matrix.columnSums());
   }

   @Override
   public Matrix copy() {
      return wrap(matrix.dup());
   }

   @Override
   public Matrix div(Matrix other) {
      return wrap(matrix.div(other.toFloatMatrix()));
   }

   @Override
   public Matrix div(double scalar) {
      return wrap(matrix.div((float) scalar));
   }

   @Override
   public Matrix divColumnVector(Matrix columnVector) {
      return wrap(matrix.divColumnVector(columnVector.toFloatMatrix()));
   }

   @Override
   public Matrix divRowVector(Matrix rowVector) {
      return wrap(matrix.divRowVector(rowVector.toFloatMatrix()));
   }

   @Override
   public Matrix divi(Matrix other) {
      matrix.divi(other.toFloatMatrix());
      return this;
   }

   @Override
   public Matrix divi(double scalar) {
      matrix.divi((float) scalar);
      return this;
   }

   @Override
   public Matrix diviColumnVector(Matrix columnVector) {
      matrix.diviColumnVector(columnVector.toFloatMatrix());
      return this;
   }

   @Override
   public Matrix diviRowVector(Matrix rowVector) {
      matrix.diviRowVector(rowVector.toFloatMatrix());
      return this;
   }

   @Override
   public double dot(Matrix other) {
      return matrix.dot(other.toFloatMatrix());
   }

   @Override
   public Matrix exp() {
      return wrap(MatrixFunctions.exp(matrix));
   }

   @Override
   public double get(int r, int c) {
      return matrix.get(r, c);
   }

   @Override
   public double get(int i) {
      return matrix.get(i);
   }

   @Override
   public Matrix getColumn(int column) {
      return wrap(matrix.getColumn(column));
   }

   @Override
   public Matrix getColumns(int[] cindexes) {
      return wrap(matrix.getColumns(cindexes));
   }

   @Override
   public Matrix getColumns(int from, int to) {
      return wrap(matrix.getColumns(new IntervalRange(from, to)));
   }

   @Override
   public ElementType getElementType() {
      return ElementType.FLOAT;
   }

   @Override
   public MatrixFactory getFactory() {
      return MatrixFactory.DENSE_FLOAT;
   }

   @Override
   public Matrix getRow(int row) {
      return wrap(matrix.getRow(row));
   }

   @Override
   public Matrix getRows(int[] rindexes) {
      return wrap(matrix.getRows(rindexes));
   }

   @Override
   public Matrix getRows(int from, int to) {
      return wrap(matrix.getRows(new IntervalRange(from, to)));
   }

   @Override
   public boolean isColumnVector() {
      return matrix.isColumnVector();
   }

   @Override
   public boolean isEmpty() {
      return matrix.isEmpty();
   }

   @Override
   public Matrix isInfinite() {
      return wrap(matrix.isInfinite());
   }

   @Override
   public Matrix isInfinitei() {
      matrix.isInfinitei();
      return this;
   }

   @Override
   public boolean isLowerTriangular() {
      return matrix.isLowerTriangular();
   }

   @Override
   public Matrix isNaN() {
      return wrap(matrix.isNaN());
   }

   @Override
   public Matrix isNaNi() {
      matrix.isNaNi();
      return this;
   }

   @Override
   public boolean isRowVector() {
      return matrix.isRowVector();
   }

   @Override
   public boolean isScalar() {
      return matrix.isSquare();
   }

   @Override
   public boolean isSquare() {
      return matrix.isSquare();
   }

   @Override
   public boolean isUpperTriangular() {
      return matrix.isUpperTriangular();
   }

   @Override
   public boolean isVector() {
      return matrix.isVector();
   }

   @Override
   public int length() {
      return matrix.length;
   }

   @Override
   public Matrix log() {
      return wrap(MatrixFunctions.log(matrix));
   }

   @Override
   public Matrix map(DoubleUnaryOperator operator) {
      float[] out = new float[matrix.length];
      for (int i = 0; i < out.length; i++) {
         out[i] = (float) operator.applyAsDouble(matrix.data[i]);
      }
      return new DenseFloatMatrix(matrix.rows, matrix.columns, out);
   }

   @Override
   public Matrix map(DoubleBinaryOperator operator, Matrix other) {
      float[] out = new float[matrix.length];
      FloatMatrix om = other.toFloatMatrix();
      for (int i = 0; i < out.length; i++) {
         out[i] = (float) operator.applyAsDouble(matrix.data[i], om.data[i]);
      }
      return new DenseFloatMatrix(matrix.rows, matrix.columns, out);
   }

   @Override
   public Matrix mapi(DoubleUnaryOperator operator) {
      for (int i = 0; i < matrix.data.length; i++) {
         matrix.data[i] = (float) operator.applyAsDouble(matrix.data[i]);
      }
      return this;
   }

   @Override
   public Matrix mapi(DoubleBinaryOperator operator, Matrix other) {
      FloatMatrix om = other.toFloatMatrix();
      for (int i = 0; i < matrix.data.length; i++) {
         matrix.data[i] = (float) operator.applyAsDouble(matrix.data[i], om.data[i]);
      }
      return this;
   }

   @Override
   public Matrix max(double scalar) {
      return wrap(matrix.max((float) scalar));
   }

   @Override
   public Matrix max(Matrix other) {
      return wrap(matrix.max(other.toFloatMatrix()));
   }

   @Override
   public Matrix maxi(double scalar) {
      matrix.maxi((float) scalar);
      return this;
   }

   @Override
   public Matrix maxi(Matrix other) {
      matrix.maxi(other.toFloatMatrix());
      return this;
   }

   @Override
   public double mean() {
      return matrix.mean();
   }

   @Override
   public Matrix mmul(Matrix other) {
      return wrap(matrix.mmul(other.toFloatMatrix()));
   }

   @Override
   public Matrix mmuli(Matrix other) {
      matrix.mmuli(other.toFloatMatrix());
      return this;
   }

   @Override
   public Matrix mul(Matrix other) {
      return wrap(matrix.mul(other.toFloatMatrix()));
   }

   @Override
   public Matrix mul(double scalar) {
      return wrap(matrix.mul((float) scalar));
   }

   @Override
   public Matrix mulColumnVector(Matrix columnVector) {
      return wrap(matrix.mulColumnVector(columnVector.toFloatMatrix()));
   }

   @Override
   public Matrix mulRowVector(Matrix rowVector) {
      return wrap(matrix.mulRowVector(rowVector.toFloatMatrix()));
   }

   @Override
   public Matrix muli(Matrix other) {
      matrix.muli(other.toFloatMatrix());
      return this;
   }

   @Override
   public Matrix muli(double scalar) {
      matrix.muli((float) scalar);
      return this;
   }

   @Override
   public Matrix muliColumnVector(Matrix columnVector) {
      matrix.muliColumnVector(columnVector.toFloatMatrix());
      return this;
   }

   @Override
   public Matrix muliRowVector(Matrix rowVector) {
      matrix.muliRowVector(rowVector.toFloatMatrix());
      return this;
   }

   @Override
   public Matrix neg() {
      return wrap(matrix.neg());
   }

   @Override
   public Matrix negi() {
      matrix.negi();
      return this;
   }

   @Override
   public int numCols() {
      return matrix.columns;
   }

   @Override
   public int numRows() {
      return matrix.rows;
   }

   @Override
   public Matrix predicate(DoublePredicate predicate) {
      float[] out = new float[matrix.length];
      for (int i = 0; i < out.length; i++) {
         out[i] = predicate.test(matrix.data[i]) ? 1.0f : 0.0f;
      }
      return new DenseFloatMatrix(matrix.rows, matrix.columns, out);
   }

   @Override
   public Matrix predicatei(DoublePredicate predicate) {
      for (int i = 0; i < matrix.data.length; i++) {
         matrix.data[i] = predicate.test(matrix.data[i]) ? 1.0f : 0.0f;
      }
      return this;
   }

   @Override
   public Matrix rdiv(double scalar) {
      return wrap(matrix.rdiv((float) scalar));
   }

   @Override
   public Matrix rdivi(Matrix other) {
      return wrap(matrix.rdiv(other.toFloatMatrix()));
   }

   @Override
   public Matrix rdivi(double scalar) {
      matrix.rdivi((float) scalar);
      return this;
   }

   @Override
   public Matrix reshape(int rows, int columns) {
      matrix.reshape(rows, columns);
      return this;
   }

   @Override
   public Matrix resize(int rows, int columns) {
      matrix.resize(rows, columns);
      return this;
   }

   @Override
   public int[] rowArgMaxs() {
      return matrix.rowArgmaxs();
   }

   @Override
   public int[] rowArgMins() {
      return matrix.rowArgmins();
   }

   @Override
   public Matrix rowMaxs() {
      return wrap(matrix.rowMaxs());
   }

   @Override
   public Matrix rowMeans() {
      return wrap(matrix.rowMeans());
   }

   @Override
   public Matrix rowMins() {
      return wrap(matrix.rowMins());
   }

   @Override
   public Matrix rowSums() {
      return wrap(matrix.rowSums());
   }

   @Override
   public Matrix rsub(double scalar) {
      return wrap(matrix.rsub((float) scalar));
   }

   @Override
   public Matrix rsubi(Matrix other) {
      matrix.rsubi(other.toFloatMatrix());
      return this;
   }

   @Override
   public Matrix rsubi(double scalar) {
      matrix.rsubi((float) scalar);
      return this;
   }

   @Override
   public Matrix select(Matrix where) {
      return wrap(matrix.select(where.toFloatMatrix()));
   }

   @Override
   public Matrix selecti(Matrix where) {
      matrix.selecti(where.toFloatMatrix());
      return this;
   }

   @Override
   public Matrix set(int r, int c, double value) {
      matrix.put(r, c, (float) value);
      return this;
   }

   @Override
   public Matrix set(int index, double value) {
      matrix.put(index, (float) value);
      return this;
   }

   @Override
   public Matrix setColumn(int c, Matrix columnVector) {
      matrix.putColumn(c, columnVector.toFloatMatrix());
      return this;
   }

   @Override
   public Matrix setRow(int r, Matrix rowVector) {
      matrix.putRow(r, rowVector.toFloatMatrix());
      return this;
   }

   @Override
   public Matrix sub(Matrix other) {
      return wrap(matrix.sub(other.toFloatMatrix()));
   }

   @Override
   public Matrix sub(double scalar) {
      return wrap(matrix.sub((float) scalar));
   }

   @Override
   public Matrix subColumnVector(Matrix columnVector) {
      return wrap(matrix.subColumnVector(columnVector.toFloatMatrix()));
   }

   @Override
   public Matrix subRowVector(Matrix rowVector) {
      return wrap(matrix.subRowVector(rowVector.toFloatMatrix()));
   }

   @Override
   public Matrix subi(Matrix other) {
      matrix.subi(other.toFloatMatrix());
      return this;
   }

   @Override
   public Matrix subi(double scalar) {
      matrix.subi((float) scalar);
      return this;
   }

   @Override
   public Matrix subiColumnVector(Matrix columnVector) {
      matrix.subiColumnVector(columnVector.toFloatMatrix());
      return this;
   }

   @Override
   public Matrix subiRowVector(Matrix rowVector) {
      matrix.subiRowVector(rowVector.toFloatMatrix());
      return this;
   }

   @Override
   public double sum() {
      return matrix.sum();
   }

   @Override
   public double[] toDoubleArray() {
      double[] out = new double[matrix.length];
      for (int i = 0; i < out.length; i++) {
         out[i] = matrix.data[i];
      }
      return out;
   }

   @Override
   public DoubleMatrix toDoubleMatrix() {
      return MatrixFunctions.floatToDouble(matrix);
   }

   @Override
   public float[] toFloatArray() {
      return matrix.toArray();
   }

   @Override
   public FloatMatrix toFloatMatrix() {
      return matrix;
   }

   @Override
   public String toString() {
      return matrix.toString();
   }

   @Override
   public Matrix transpose() {
      return wrap(matrix.transpose());
   }
}// END OF DenseFloatMatrix
