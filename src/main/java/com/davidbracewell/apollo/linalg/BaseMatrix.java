package com.davidbracewell.apollo.linalg;

import com.davidbracewell.collection.Streams;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.function.SerializableFunction;
import com.davidbracewell.guava.common.base.Preconditions;
import com.davidbracewell.tuple.Tuple2;
import lombok.NonNull;

import java.io.Serializable;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.PrimitiveIterator;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.IntStream;

import static com.davidbracewell.tuple.Tuples.$;

/**
 * @author David B. Bracewell
 */
public abstract class BaseMatrix implements Matrix, Serializable {
   private static final long serialVersionUID = 1L;

   @Override
   public Matrix T() {
      final Matrix transposed = createNew(numberOfRows(), numberOfColumns());
      forEachSparse(e -> transposed.set(e.column, e.row, e.value));
      return transposed;
   }

   @Override
   public Matrix add(@NonNull Matrix other) {
      return copy().addSelf(other);
   }

   @Override
   public Matrix addColumn(@NonNull Vector vector) {
      return copy().addColumnSelf(vector);
   }

   @Override
   public Matrix addColumnSelf(@NonNull Vector vector) {
      forEachColumn(c -> c.addSelf(vector));
      return this;
   }

   @Override
   public Matrix addRow(@NonNull Vector vector) {
      return copy().addRowSelf(vector);
   }

   @Override
   public Matrix addRowSelf(@NonNull Vector vector) {
      forEachRow(r -> r.addSelf(vector));
      return this;
   }

   @Override
   public Matrix addSelf(@NonNull Matrix other) {
      Preconditions.checkArgument(
         other.numberOfColumns() == numberOfColumns() && other.numberOfRows() == numberOfRows(),
         "Dimension Mismatch");
      other.forEachSparse(e -> increment(e.row, e.column, e.value));
      return this;
   }

   @Override
   public Vector column(int column) {
      Preconditions.checkPositionIndex(column, numberOfColumns(), "Column out of index");
      return new ColumnVector(this, column);
   }

   @Override
   public Iterator<Vector> columnIterator() {
      return new Iterator<Vector>() {
         private AtomicInteger c = new AtomicInteger(0);

         @Override
         public boolean hasNext() {
            return c.get() < numberOfColumns();
         }

         @Override
         public Vector next() {
            if (!hasNext()) {
               throw new NoSuchElementException();
            }
            return column(c.getAndIncrement());
         }
      };
   }

   protected abstract Matrix createNew(int numRows, int numColumns);

   @Override
   public Matrix decrement(int row, int col, double amount) {
      return increment(row, col, -amount);
   }

   @Override
   public Matrix decrement(int row, int col) {
      return increment(row, col, -1);
   }


   @Override
   public Matrix diag() {
      final Matrix m = createNew(numberOfRows(), numberOfColumns());
      for (int r = 0; r < numberOfRows() && r < numberOfColumns(); r++) {
         m.set(r, r, get(r, r));
      }
      return m;
   }

   @Override
   public Vector diagVector() {
      Vector diag = new DenseVector(Math.max(numberOfColumns(), numberOfRows()));
      for (int r = 0; r < numberOfRows() && r < numberOfColumns(); r++) {
         diag.set(r, get(r, r));
      }
      return diag;
   }

   @Override
   public Matrix dot(@NonNull Vector vector) {
      Preconditions.checkArgument(vector.dimension() == numberOfColumns(), "Dimension mismatch");
      Matrix dot = createNew(numberOfRows(), 1);
      for (int r = 0; r < numberOfRows(); r++) {
         dot.set(r, 0, row(r).dot(vector));
      }
      return dot;
   }

   @Override
   public Matrix dot(@NonNull Matrix matrix) {
      Preconditions.checkArgument(shape().equals(matrix.shape()), "Dimension mismatch");
      Matrix dot = createNew(numberOfRows(), 1);
      for (int r = 0; r < numberOfRows(); r++) {
         dot.set(r, 0, row(r).dot(matrix.row(r)));
      }
      return dot;
   }

   @Override
   public void forEachColumn(@NonNull Consumer<Vector> consumer) {
      columnIterator().forEachRemaining(consumer);
   }

   @Override
   public void forEachOrderedSparse(@NonNull Consumer<Matrix.Entry> consumer) {
      orderedNonZeroIterator().forEachRemaining(consumer);
   }


   @Override
   public void forEachRow(@NonNull Consumer<Vector> consumer) {
      rowIterator().forEachRemaining(consumer);
   }

   @Override
   public void forEachSparse(@NonNull Consumer<Matrix.Entry> consumer) {
      nonZeroIterator().forEachRemaining(consumer);
   }

   @Override
   public Matrix increment(double value) {
      return copy().incrementSelf(value);
   }


   @Override
   public Matrix increment(int row, int col) {
      return increment(row, col, 1);
   }

   @Override
   public Matrix incrementSelf(double value) {
      forEachRow(row -> row.mapAddSelf(value));
      return this;
   }

   @Override
   public boolean isDense() {
      return false;
   }

   @Override
   public boolean isSparse() {
      return false;
   }

   @Override
   public Iterator<Entry> iterator() {
      return new Iterator<Entry>() {
         private PrimitiveIterator.OfInt rowItr = IntStream
                                                     .range(0, numberOfRows())
                                                     .iterator();
         private int row;
         private PrimitiveIterator.OfInt colItr = null;

         private boolean advance() {
            while (colItr == null || !colItr.hasNext()) {
               if (colItr == null && !rowItr.hasNext()) {
                  return false;
               } else if (colItr == null) {
                  row = rowItr.next();
                  colItr = IntStream
                              .range(0, numberOfColumns())
                              .iterator();
                  return true;
               } else if (!colItr.hasNext()) {
                  colItr = null;
               }
            }
            return colItr != null;
         }

         @Override
         public boolean hasNext() {
            return advance();
         }

         @Override
         public Entry next() {
            if (!advance()) {
               throw new NoSuchElementException();
            }
            int col = colItr.next();
            return new Matrix.Entry(row, col, get(row, col));
         }
      };
   }

   @Override
   public Matrix map(@NonNull DoubleUnaryOperator operator) {
      return copy().mapSelf(operator);
   }

   @Override
   public Matrix mapColumn(@NonNull Vector vector, @NonNull DoubleBinaryOperator operator) {
      return copy().mapColumnSelf(vector, operator);
   }

   @Override
   public Matrix mapColumnSelf(@NonNull Vector vector, @NonNull DoubleBinaryOperator operator) {
      forEachColumn(r -> r.mapSelf(vector, operator));
      return this;
   }

   @Override
   public Matrix mapRow(@NonNull Vector vector, @NonNull DoubleBinaryOperator operator) {
      return copy().mapRowSelf(vector, operator);
   }

   @Override
   public Matrix mapRow(@NonNull SerializableFunction<Vector, Vector> function) {
      Matrix mPrime = copy();
      for (int i = 0; i < mPrime.numberOfRows(); i++) {
         mPrime.setRow(i, function.apply(row(i)));
      }
      return mPrime;
   }

   @Override
   public Matrix mapRowSelf(@NonNull Vector vector, @NonNull DoubleBinaryOperator operator) {
      forEachRow(r -> r.mapSelf(vector, operator));
      return this;
   }

   @Override
   public Matrix mapRowSelf(@NonNull SerializableFunction<Vector, Vector> function) {
      for (int i = 0; i < numberOfRows(); i++) {
         setRow(i, function.apply(row(i)));
      }
      return this;
   }

   @Override
   public Matrix mapSelf(@NonNull DoubleUnaryOperator operator) {
      forEach(entry -> set(entry.row, entry.column, operator.applyAsDouble(entry.value)));
      return this;
   }

   @Override
   public Matrix multiply(@NonNull Matrix m) {
      Preconditions.checkArgument(numberOfColumns() == m.numberOfRows(), "Dimension Mismatch");
      if (m.isSparse() && isSparse()) {
         return new SparseMatrix(Cast.<SparseMatrix>as(this).asRealMatrix()
                                                            .multiply(Cast.<SparseMatrix>as(m).asRealMatrix()));
      }
      final Matrix mprime = createNew(numberOfRows(), m.numberOfColumns());
      IntStream
         .range(0, numberOfRows())
         .parallel()
         .forEach(r -> {
            for (int c = 0; c < m.numberOfColumns(); c++) {
               for (int k = 0; k < numberOfColumns(); k++) {
                  synchronized (mprime) {
                     mprime.increment(r, c, get(r, k) * m.get(k, c));
                  }
               }
            }
         });
      return mprime;
   }

   @Override
   public Matrix multiplyVector(@NonNull Vector v) {
      return copy().multiplyVectorSelf(v);
   }

   @Override
   public Matrix multiplyVectorSelf(@NonNull Vector v) {
      rowIterator().forEachRemaining(r -> r.multiply(v));
      return this;
   }

   @Override
   public Iterator<Matrix.Entry> nonZeroIterator() {
      return orderedNonZeroIterator();
   }

   @Override
   public Iterator<Matrix.Entry> orderedNonZeroIterator() {
      return new Iterator<Entry>() {
         private PrimitiveIterator.OfInt rowItr = IntStream
                                                     .range(0, numberOfRows())
                                                     .iterator();
         private int row;
         private Integer col;
         private PrimitiveIterator.OfInt colItr = null;

         private boolean advance() {
            while (col == null || get(row, col) == 0) {

               if (colItr == null && !rowItr.hasNext()) {
                  return false;
               }

               if (colItr == null) {
                  row = rowItr.next();
                  colItr = IntStream
                              .range(0, numberOfColumns())
                              .iterator();
                  col = colItr.next();
               } else if (!colItr.hasNext()) {
                  colItr = null;
               } else {
                  col = colItr.next();
               }

            }

            return col != null && get(row, col) != 0;
         }

         @Override
         public boolean hasNext() {
            return advance();
         }

         @Override
         public Entry next() {
            if (!advance()) {
               throw new NoSuchElementException();
            }
            int column = col;
            col = null;
            return new Matrix.Entry(row, column, get(row, column));
         }
      };
   }

   @Override
   public Vector row(int row) {
      Preconditions.checkPositionIndex(row, numberOfRows(), "Row out of index");
      return new RowVector(this, row);
   }

   @Override
   public Iterator<Vector> rowIterator() {
      return new Iterator<Vector>() {
         private AtomicInteger r = new AtomicInteger(0);

         @Override
         public boolean hasNext() {
            return r.get() < numberOfRows();
         }

         @Override
         public Vector next() {
            if (!hasNext()) {
               throw new NoSuchElementException();
            }
            return row(r.getAndIncrement());
         }
      };
   }

   @Override
   public Matrix scale(@NonNull Matrix other) {
      return copy().scaleSelf(other);
   }

   @Override
   public Matrix scale(double value) {
      return copy().scaleSelf(value);
   }

   @Override
   public Matrix scale(int r, int c, double amount) {
      set(r, c, get(r, c) * amount);
      return this;
   }

   @Override
   public Matrix scaleSelf(Matrix other) {
      Preconditions.checkArgument(
         other.numberOfColumns() == numberOfColumns() && other.numberOfRows() == numberOfRows(),
         "Dimension Mismatch");
      other.forEachSparse(e -> scale(e.row, e.column, e.value));
      return this;
   }

   @Override
   public Matrix scaleSelf(double value) {
      forEachSparse(e -> set(e.row, e.column, e.value * value));
      return this;
   }

   @Override
   public Matrix setColumn(int column, @NonNull Vector vector) {
      Preconditions.checkElementIndex(column, numberOfColumns());
      Preconditions.checkArgument(vector.dimension() == numberOfRows(), "Dimension Mismatch");
      vector.forEach(e -> set(e.index, column, e.value));
      return this;
   }

   @Override
   public Matrix setRow(int row, Vector vector) {
      Preconditions.checkElementIndex(row, numberOfRows());
      Preconditions.checkArgument(vector.dimension() == numberOfColumns(), "Dimension Mismatch");
      vector.forEach(e -> set(row, e.index, e.value));
      return this;
   }

   @Override
   public Tuple2<Integer, Integer> shape() {
      return $(numberOfRows(), numberOfColumns());
   }

   @Override
   public Matrix subtract(@NonNull Matrix other) {
      return copy().subtractSelf(other);
   }

   @Override
   public Matrix subtractColumn(@NonNull Vector vector) {
      return copy().subtractColumnSelf(vector);
   }

   @Override
   public Matrix subtractColumnSelf(@NonNull Vector vector) {
      forEachColumn(c -> c.subtractSelf(vector));
      return this;
   }

   @Override
   public Matrix subtractRow(@NonNull Vector vector) {
      return copy().subtractRowSelf(vector);
   }

   @Override
   public Matrix subtractRowSelf(@NonNull Vector vector) {
      forEachRow(r -> r.subtractSelf(vector));
      return this;
   }

   @Override
   public Matrix subtractSelf(@NonNull Matrix other) {
      Preconditions.checkArgument(
         other.numberOfColumns() == numberOfColumns() && other.numberOfRows() == numberOfRows(),
         "Dimension Mismatch");
      other.forEachSparse(e -> decrement(e.row, e.column, e.value));
      return this;
   }

   @Override
   public double sum() {
      return Streams
                .asStream(this)
                .mapToDouble(Entry::getValue)
                .sum();
   }

   @Override
   public double[][] toArray() {
      double[][] array = new double[numberOfRows()][numberOfColumns()];
      forEachSparse(e -> array[e.row][e.column] = e.value);
      return array;
   }

   @Override
   public DenseMatrix toDense() {
      return new DenseMatrix(this);
   }

}// END OF BaseMatrix
