package com.davidbracewell.apollo.linalg;

import com.davidbracewell.Copyable;
import com.davidbracewell.tuple.Tuple2;
import lombok.NonNull;
import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;

import java.util.function.DoubleBinaryOperator;
import java.util.function.DoublePredicate;
import java.util.function.DoubleUnaryOperator;

import static com.davidbracewell.tuple.Tuples.$;

/**
 * The interface M.
 *
 * @author David B. Bracewell
 */
public interface Matrix extends Copyable<Matrix> {

   /**
    * Add m.
    *
    * @param other the other
    * @return the m
    */
   Matrix add(Matrix other);

   /**
    * Add m.
    *
    * @param scalar the scalar
    * @return the m
    */
   Matrix add(double scalar);

   /**
    * Add column vector m.
    *
    * @param columnVector the column vector
    * @return the m
    */
   Matrix addColumnVector(Matrix columnVector);

   /**
    * Add row vector m.
    *
    * @param rowVector the row vector
    * @return the m
    */
   Matrix addRowVector(Matrix rowVector);

   /**
    * Addi m.
    *
    * @param other the other
    * @return the m
    */
   Matrix addi(Matrix other);

   /**
    * Addi m.
    *
    * @param scalar the scalar
    * @return the m
    */
   Matrix addi(double scalar);

   /**
    * Addi column vector m.
    *
    * @param columnVector the column vector
    * @return the m
    */
   Matrix addiColumnVector(Matrix columnVector);

   /**
    * Addi row vector m.
    *
    * @param rowVector the row vector
    * @return the m
    */
   Matrix addiRowVector(Matrix rowVector);

   int[] columnArgMaxs();

   int[] columnArgMins();

   /**
    * Column maxs m.
    *
    * @return the m
    */
   Matrix columnMaxs();

   Matrix columnMeans();

   /**
    * Column mins m.
    *
    * @return the m
    */
   Matrix columnMins();

   /**
    * Column sums m.
    *
    * @return the m
    */
   Matrix columnSums();

   /**
    * Div m.
    *
    * @param other the other
    * @return the m
    */
   Matrix div(Matrix other);

   /**
    * Div m.
    *
    * @param scalar the scalar
    * @return the m
    */
   Matrix div(double scalar);

   /**
    * Div column vector m.
    *
    * @param columnVector the column vector
    * @return the m
    */
   Matrix divColumnVector(Matrix columnVector);

   /**
    * Div row vector m.
    *
    * @param rowVector the row vector
    * @return the m
    */
   Matrix divRowVector(Matrix rowVector);

   /**
    * Divi m.
    *
    * @param other the other
    * @return the m
    */
   Matrix divi(Matrix other);

   /**
    * Divi m.
    *
    * @param scalar the scalar
    * @return the m
    */
   Matrix divi(double scalar);

   /**
    * Divi column vector m.
    *
    * @param columnVector the column vector
    * @return the m
    */
   Matrix diviColumnVector(Matrix columnVector);

   /**
    * Divi row vector m.
    *
    * @param rowVector the row vector
    * @return the m
    */
   Matrix diviRowVector(Matrix rowVector);

   /**
    * Dot double.
    *
    * @param other the other
    * @return the double
    */
   double dot(Matrix other);

   /**
    * Exp m.
    *
    * @return the m
    */
   Matrix exp();

   double get(int r, int c);

   double get(int i);

   /**
    * Gets column.
    *
    * @param column the column
    * @return the column
    */
   Matrix getColumn(int column);

   Matrix getColumns(int from, int to);

   /**
    * Gets columns.
    *
    * @param cindexes the cindexes
    * @return the columns
    */
   Matrix getColumns(int[] cindexes);

   /**
    * Gets element type.
    *
    * @return the element type
    */
   ElementType getElementType();

   MatrixFactory getFactory();

   /**
    * Gets row.
    *
    * @param row the row
    * @return the row
    */
   Matrix getRow(int row);

   /**
    * Gets rows.
    *
    * @param rindexes the rindexes
    * @return the rows
    */
   Matrix getRows(int[] rindexes);

   Matrix getRows(int from, int to);

   /**
    * Is column vector boolean.
    *
    * @return the boolean
    */
   boolean isColumnVector();

   /**
    * Is empty boolean.
    *
    * @return the boolean
    */
   boolean isEmpty();

   /**
    * Is infinite m.
    *
    * @return the m
    */
   Matrix isInfinite();

   /**
    * Is infinitei m.
    *
    * @return the m
    */
   Matrix isInfinitei();

   /**
    * Is lower triangular boolean.
    *
    * @return the boolean
    */
   boolean isLowerTriangular();

   /**
    * Is na n m.
    *
    * @return the m
    */
   Matrix isNaN();

   /**
    * Is na ni m.
    *
    * @return the m
    */
   Matrix isNaNi();

   /**
    * Is row vector boolean.
    *
    * @return the boolean
    */
   boolean isRowVector();

   /**
    * Is scalar boolean.
    *
    * @return the boolean
    */
   boolean isScalar();

   /**
    * Is square boolean.
    *
    * @return the boolean
    */
   boolean isSquare();

   /**
    * Is upper triangular boolean.
    *
    * @return the boolean
    */
   boolean isUpperTriangular();

   /**
    * Is vector boolean.
    *
    * @return the boolean
    */
   boolean isVector();

   int length();

   /**
    * Log m.
    *
    * @return the m
    */
   Matrix log();

   /**
    * Map m.
    *
    * @param operator the operator
    * @return the m
    */
   Matrix map(DoubleUnaryOperator operator);

   /**
    * Map m.
    *
    * @param operator the operator
    * @param other    the other
    * @return the m
    */
   Matrix map(DoubleBinaryOperator operator, Matrix other);

   /**
    * Mapi m.
    *
    * @param operator the operator
    * @return the m
    */
   Matrix mapi(DoubleUnaryOperator operator);

   /**
    * Mapi m.
    *
    * @param operator the operator
    * @param other    the other
    * @return the m
    */
   Matrix mapi(DoubleBinaryOperator operator, Matrix other);

   /**
    * Max m.
    *
    * @param scalar the scalar
    * @return the m
    */
   Matrix max(double scalar);

   /**
    * Max m.
    *
    * @param other the other
    * @return the m
    */
   Matrix max(Matrix other);

   /**
    * Maxi m.
    *
    * @param scalar the scalar
    * @return the m
    */
   Matrix maxi(double scalar);

   /**
    * Maxi m.
    *
    * @param other the other
    * @return the m
    */
   Matrix maxi(Matrix other);

   /**
    * Mean double.
    *
    * @return the double
    */
   double mean();

   /**
    * Mmul m.
    *
    * @param other the other
    * @return the m
    */
   Matrix mmul(Matrix other);

   /**
    * Mmuli m.
    *
    * @param other the other
    * @return the m
    */
   Matrix mmuli(Matrix other);

   /**
    * Mul m.
    *
    * @param other the other
    * @return the m
    */
   Matrix mul(Matrix other);

   /**
    * Mul m.
    *
    * @param scalar the scalar
    * @return the m
    */
   Matrix mul(double scalar);

   /**
    * Mul column vector m.
    *
    * @param columnVector the column vector
    * @return the m
    */
   Matrix mulColumnVector(Matrix columnVector);

   /**
    * Mul row vector m.
    *
    * @param rowVector the row vector
    * @return the m
    */
   Matrix mulRowVector(Matrix rowVector);

   /**
    * Muli m.
    *
    * @param other the other
    * @return the m
    */
   Matrix muli(Matrix other);

   /**
    * Muli m.
    *
    * @param scalar the scalar
    * @return the m
    */
   Matrix muli(double scalar);

   /**
    * Muli column vector m.
    *
    * @param columnVector the column vector
    * @return the m
    */
   Matrix muliColumnVector(Matrix columnVector);

   /**
    * Muli row vector m.
    *
    * @param rowVector the row vector
    * @return the m
    */
   Matrix muliRowVector(Matrix rowVector);

   /**
    * Neg m.
    *
    * @return the m
    */
   Matrix neg();

   /**
    * Negi m.
    *
    * @return the m
    */
   Matrix negi();

   int numCols();

   int numRows();

   /**
    * Predicate m.
    *
    * @param predicate the predicate
    * @return the m
    */
   Matrix predicate(DoublePredicate predicate);

   /**
    * Predicatei m.
    *
    * @param predicate the predicate
    * @return the m
    */
   Matrix predicatei(DoublePredicate predicate);

   /**
    * Rdiv m.
    *
    * @param other the other
    * @return the m
    */
   default Matrix rdiv(@NonNull Matrix other) {
      return other.div(this);
   }

   /**
    * Rdiv m.
    *
    * @param scalar the scalar
    * @return the m
    */
   Matrix rdiv(double scalar);

   /**
    * Rdivi m.
    *
    * @param other the other
    * @return the m
    */
   Matrix rdivi(Matrix other);

   /**
    * Rdivi m.
    *
    * @param scalar the scalar
    * @return the m
    */
   Matrix rdivi(double scalar);

   Matrix reshape(int rows, int columns);

   Matrix resize(int rows, int columns);

   int[] rowArgMaxs();

   int[] rowArgMins();

   Matrix rowMaxs();

   Matrix rowMeans();

   Matrix rowMins();

   /**
    * Row sums m.
    *
    * @return the m
    */
   Matrix rowSums();

   /**
    * Rsub m.
    *
    * @param other the other
    * @return the m
    */
   default Matrix rsub(@NonNull Matrix other) {
      return other.sub(this);
   }

   /**
    * Rsub m.
    *
    * @param scalar the scalar
    * @return the m
    */
   Matrix rsub(double scalar);

   /**
    * Rsubi m.
    *
    * @param other the other
    * @return the m
    */
   Matrix rsubi(Matrix other);

   /**
    * Rsubi m.
    *
    * @param scalar the scalar
    * @return the m
    */
   Matrix rsubi(double scalar);

   /**
    * Select m.
    *
    * @param where the where
    * @return the m
    */
   Matrix select(Matrix where);

   /**
    * Selecti m.
    *
    * @param where the where
    * @return the m
    */
   Matrix selecti(Matrix where);

   Matrix set(int r, int c, double value);

   Matrix set(int index, double value);

   Matrix setColumn(int c, Matrix columnVector);

   Matrix setRow(int r, Matrix rowVector);

   default Tuple2<Integer, Integer> shape() {
      return $(numRows(), numCols());
   }

   /**
    * Sub m.
    *
    * @param other the other
    * @return the m
    */
   Matrix sub(Matrix other);

   /**
    * Sub m.
    *
    * @param scalar the scalar
    * @return the m
    */
   Matrix sub(double scalar);

   /**
    * Sub column vector m.
    *
    * @param columnVector the column vector
    * @return the m
    */
   Matrix subColumnVector(Matrix columnVector);

   /**
    * Sub row vector m.
    *
    * @param rowVector the row vector
    * @return the m
    */
   Matrix subRowVector(Matrix rowVector);

   /**
    * Subi m.
    *
    * @param other the other
    * @return the m
    */
   Matrix subi(Matrix other);

   /**
    * Subi m.
    *
    * @param scalar the scalar
    * @return the m
    */
   Matrix subi(double scalar);

   /**
    * Subi column vector m.
    *
    * @param columnVector the column vector
    * @return the m
    */
   Matrix subiColumnVector(Matrix columnVector);

   /**
    * Subi row vector m.
    *
    * @param rowVector the row vector
    * @return the m
    */
   Matrix subiRowVector(Matrix rowVector);

   /**
    * Sum double.
    *
    * @return the double
    */
   double sum();

   /**
    * To double array double [ ].
    *
    * @return the double [ ]
    */
   double[] toDoubleArray();

   /**
    * To double matrix double matrix.
    *
    * @return the double matrix
    */
   DoubleMatrix toDoubleMatrix();

   /**
    * To float array float [ ].
    *
    * @return the float [ ]
    */
   float[] toFloatArray();

   /**
    * To float matrix float matrix.
    *
    * @return the float matrix
    */
   FloatMatrix toFloatMatrix();

   /**
    * Transpose m.
    *
    * @return the m
    */
   Matrix transpose();

   /**
    * The enum Element type.
    */
   enum ElementType {
      /**
       * Float element type.
       */
      FLOAT,
      /**
       * Double element type.
       */
      DOUBLE,
      /**
       * Int element type.
       */
      INT
   }

}//END OF M
