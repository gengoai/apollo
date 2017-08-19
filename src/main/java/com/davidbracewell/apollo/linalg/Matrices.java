package com.davidbracewell.apollo.linalg;

import com.davidbracewell.guava.common.base.Preconditions;
import lombok.NonNull;
import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;
import org.jblas.Singular;

import java.util.Arrays;
import java.util.Collection;
import java.util.List;

/**
 * The type Matrix.
 *
 * @author David B. Bracewell
 */
public final class Matrices {

   private Matrices() {
      throw new IllegalAccessError();
   }

   /**
    * Double column matrix double matrix.
    *
    * @param data the data
    * @return the double matrix
    */
   public static DoubleMatrix doubleColumnMatrix(@NonNull double[] data) {
      return new DoubleMatrix(data.length, 1, data);
   }

   /**
    * Double matrix from columns double matrix.
    *
    * @param columns the columns
    * @return the double matrix
    */
   public static DoubleMatrix doubleMatrixFromColumns(Vector... columns) {
      if (columns == null || columns.length == 0) {
         return DoubleMatrix.zeros(0);
      }
      return doubleMatrixFromColumns(Arrays.asList(columns));
   }

   /**
    * Double matrix from columns double matrix.
    *
    * @param columns the columns
    * @return the double matrix
    */
   public static DoubleMatrix doubleMatrixFromColumns(Collection<Vector> columns) {
      if (columns == null || columns.size() == 0) {
         return DoubleMatrix.zeros(0);
      }
      DoubleMatrix matrix = null;
      int column = 0;
      for (Vector c : columns) {
         if (matrix == null) {
            matrix = DoubleMatrix.zeros(c.dimension(), columns.size());
         }
         matrix.putColumn(column, new DoubleMatrix(c.dimension(), 1, c.toArray()));
         column++;
      }
      return matrix;
   }

   /**
    * Double matrix from rows double matrix.
    *
    * @param rows the rows
    * @return the double matrix
    */
   public static DoubleMatrix doubleMatrixFromRows(Vector... rows) {
      if (rows == null || rows.length == 0) {
         return DoubleMatrix.zeros(0);
      }
      return doubleMatrixFromRows(Arrays.asList(rows));
   }

   /**
    * Double matrix from rows double matrix.
    *
    * @param rows the rows
    * @return the double matrix
    */
   public static DoubleMatrix doubleMatrixFromRows(Collection<Vector> rows) {
      if (rows == null || rows.size() == 0) {
         return DoubleMatrix.zeros(0);
      }
      DoubleMatrix matrix = null;
      int row = 0;
      for (Vector r : rows) {
         if (matrix == null) {
            matrix = DoubleMatrix.zeros(rows.size(), r.dimension());
         }
         matrix.putRow(row, new DoubleMatrix(1, r.dimension(), r.toArray()));
         row++;
      }
      return matrix;
   }

   public static FloatMatrix combine(List<FloatMatrix> rows, int[] indices){
      FloatMatrix out = new FloatMatrix(1,1);
      return out;
   }

   /**
    * Double row matrix double matrix.
    *
    * @param data the data
    * @return the double matrix
    */
   public static DoubleMatrix doubleRowMatrix(@NonNull double[] data) {
      return new DoubleMatrix(1, data.length, data);
   }

   /**
    * Float matrix from columns float matrix.
    *
    * @param columns the columns
    * @return the float matrix
    */
   public static FloatMatrix floatMatrixFromColumns(Vector... columns) {
      if (columns == null || columns.length == 0) {
         return FloatMatrix.zeros(0);
      }
      return floatMatrixFromColumns(Arrays.asList(columns));
   }

   /**
    * Float matrix from columns float matrix.
    *
    * @param columns the columns
    * @return the float matrix
    */
   public static FloatMatrix floatMatrixFromColumns(Collection<Vector> columns) {
      if (columns == null || columns.size() == 0) {
         return FloatMatrix.zeros(0);
      }
      FloatMatrix matrix = null;
      int column = 0;
      for (Vector c : columns) {
         if (matrix == null) {
            matrix = FloatMatrix.zeros(c.dimension(), columns.size());
         }
         matrix.putColumn(column, new FloatMatrix(c.dimension(), 1, c.toFloatArray()));
         column++;
      }
      return matrix;
   }

   /**
    * Float matrix from rows float matrix.
    *
    * @param rows the rows
    * @return the float matrix
    */
   public static FloatMatrix floatMatrixFromRows(Collection<Vector> rows) {
      if (rows == null || rows.size() == 0) {
         return FloatMatrix.zeros(0);
      }
      FloatMatrix matrix = null;
      int row = 0;
      for (Vector r : rows) {
         if (matrix == null) {
            matrix = FloatMatrix.zeros(rows.size(), r.dimension());
         }
         matrix.putColumn(row, new FloatMatrix(1, r.dimension(), r.toFloatArray()));
         row++;
      }
      return matrix;
   }

   /**
    * Float matrix from rows float matrix.
    *
    * @param rows the rows
    * @return the float matrix
    */
   public static FloatMatrix floatMatrixFromRows(Vector... rows) {
      if (rows == null || rows.length == 0) {
         return FloatMatrix.zeros(0);
      }
      return floatMatrixFromRows(Arrays.asList(rows));
   }

   /**
    * Svd double matrix [ ].
    *
    * @param matrix the matrix
    * @return the double matrix [ ]
    */
   public static DoubleMatrix[] svd(@NonNull DoubleMatrix matrix) {
      return svd(matrix, true);
   }

   /**
    * Svd double matrix [ ].
    *
    * @param matrix the matrix
    * @param sparse the sparse
    * @return the double matrix [ ]
    */
   public static DoubleMatrix[] svd(@NonNull DoubleMatrix matrix, boolean sparse) {
      if (sparse) {
         return Singular.sparseSVD(matrix);
      }
      return Singular.fullSVD(matrix);
   }

   /**
    * Truncated svd double matrix [ ].
    *
    * @param matrix the matrix
    * @param K      the k
    * @return the double matrix [ ]
    */
   public static DoubleMatrix[] truncatedSVD(@NonNull DoubleMatrix matrix, int K) {
      return truncatedSVD(matrix, K, true);
   }

   /**
    * Truncated svd double matrix [ ].
    *
    * @param matrix the matrix
    * @param K      the k
    * @param sparse the sparse
    * @return the double matrix [ ]
    */
   public static DoubleMatrix[] truncatedSVD(@NonNull DoubleMatrix matrix, int K, boolean sparse) {
      Preconditions.checkArgument(K > 0, "Number of singular values must be greater than zero.");
      DoubleMatrix[] result = svd(matrix, sparse);
      DoubleMatrix u = result[0].getRange(0, result[0].getRows(), 0, K);
      DoubleMatrix s = DoubleMatrix.diag(result[1].getRange(0, K));
      DoubleMatrix v = result[2].getRange(0, result[2].getRows(), 0, K);
      return new DoubleMatrix[]{u, s, v};
   }

}//END OF Matrix
