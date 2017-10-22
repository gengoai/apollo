package com.davidbracewell.apollo.linear;

import com.davidbracewell.Copyable;
import com.davidbracewell.EnhancedDoubleStatistics;
import com.davidbracewell.Math2;
import com.davidbracewell.collection.Streams;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.guava.common.base.Preconditions;
import lombok.NonNull;
import org.apache.commons.math3.util.FastMath;
import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;

import java.io.PrintStream;
import java.io.PrintWriter;
import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Iterator;
import java.util.function.Consumer;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoublePredicate;
import java.util.function.DoubleUnaryOperator;

/**
 * An n-dimension array of double values used for vectors and matrices.
 *
 * @author David B. Bracewell
 */
public abstract class NDArray implements Serializable, Copyable<NDArray> {
   private static final long serialVersionUID = 1L;

   private Object label;
   private double weight;
   private Object predicted;

   /**
    * Flips the matrix on its diagonal switching the rows and columns
    *
    * @return the transposed array
    */
   public NDArray T() {
      NDArray t = getFactory().zeros(shape().j, shape().i);
      forEachSparse(entry -> t.set(entry.getJ(), entry.getI(), entry.getValue()));
      return t;
   }

   /**
    * Adds a scalar value to each element in the NDArray
    *
    * @param scalar the value to add
    * @return the new NDArray with the scalar value added
    */
   public NDArray add(double scalar) {
      if (scalar == 0) {
         return copy();
      }
      return map(d -> d + scalar);
   }

   /**
    * Adds the values in the other NDArray to this one.
    *
    * @param other the other NDArray whose values will be added
    * @return the new NDArray with the result of this + other
    * @throws IllegalArgumentException If the shape of this NDArray does not match that of the other NDArray
    */
   public NDArray add(@NonNull NDArray other) {
      return map(other, Math2::add);
   }

   /**
    * Performs a column or row vector addition adding the values in the other NDArray to each row or column in this
    * NDArray as specified by the given axis parameter.
    *
    * @param other the other NDArray whose values will be added
    * @param axis  the axis
    * @return the new NDArray with the result of this + other
    * @throws IllegalArgumentException If the row/column shape of this NDArray does not match that of the other NDArray
    */
   public NDArray add(@NonNull NDArray other, @NonNull Axis axis) {
      return map(other, axis, Math2::add);
   }

   /**
    * Adds a scalar value to each element in the NDArray in-place
    *
    * @param scalar the value to add
    * @return this NDArray with the scalar value added
    */
   public NDArray addi(double scalar) {
      if (scalar == 0) {
         return this;
      }
      return mapi(d -> d + scalar);
   }

   /**
    * Adds the values in the other NDArray to this one in-place.
    *
    * @param other the other NDArray whose values will be added
    * @return this NDArray with the result of this + other
    * @throws IllegalArgumentException If the shape of this NDArray does not match that of the other NDArray
    */
   public NDArray addi(@NonNull NDArray other) {
      shape().checkDimensionMatch(other.shape());
      other.forEachSparse(e -> increment(e.getI(), e.getJ(), e.getValue()));
      return this;
   }

   /**
    * Performs a column or row vector addition adding the values in the other NDArray to each row or column in this
    * NDArray as specified by the given axis parameter in-place.
    *
    * @param other the other NDArray whose values will be added
    * @param axis  the axis
    * @return this NDArray with the result of this + other
    * @throws IllegalArgumentException If the row/column shape of this NDArray does not match that of the other NDArray
    */
   public NDArray addi(@NonNull NDArray other, @NonNull Axis axis) {
      return mapi(other, axis, Math2::add);
   }

   public NDArray addiVector(int index, @NonNull NDArray vector, @NonNull Axis axis) {
      shape().checkDimensionMatch(axis.T(), vector.shape(), axis.T());
      Preconditions.checkArgument(vector.isVector(axis));
      if (axis == Axis.ROW) {
         vector.forEachSparse(entry -> increment(index, entry.getJ(), entry.getValue()));
      } else {
         vector.forEachSparse(entry -> increment(entry.getI(), index, entry.getValue()));
      }
      return this;
   }

   /**
    * Calculates the index of the maximum along each row or column based on the given axis.
    *
    * @param axis the axis (row/column) to calculate the max for
    * @return int array of row/column indexes relating to max values
    */
   public int[] argMax(@NonNull Axis axis) {
      NDArray aMax = getFactory().zeros(shape().get(axis), axis);
      NDArray vMax = getFactory().zeros(shape().get(axis), axis);
      vMax.mapi(d -> Double.NEGATIVE_INFINITY);
      forEach(entry -> {
         int index = entry.get(axis);
         if (entry.getValue() > vMax.get(index)) {
            vMax.set(index, entry.getValue());
            aMax.set(index, entry.get(axis.T()));
         }
      });
      int[] toReturn = new int[aMax.length()];
      aMax.forEach(e -> toReturn[e.getIndex()] = (int) e.getValue());
      return toReturn;
   }

   /**
    * Calculates the index of the minimum along each row or column based on the given axis.
    *
    * @param axis the axis (row/column) to calculate the min for
    * @return int array of row/column indexes relating to min values
    */
   public int[] argMin(@NonNull Axis axis) {
      NDArray aMin = getFactory().zeros(shape().get(axis), axis);
      NDArray vMin = getFactory().zeros(shape().get(axis), axis);
      vMin.mapi(d -> Double.POSITIVE_INFINITY);
      forEach(entry -> {
         int index = entry.get(axis);
         if (entry.getValue() < vMin.get(index)) {
            vMin.set(index, entry.getValue());
            aMin.set(index, entry.get(axis.T()));
         }
      });
      int[] toReturn = new int[aMin.length()];
      aMin.forEach(e -> toReturn[e.getIndex()] = (int) e.getValue());
      return toReturn;
   }

   /**
    * Compresses the underneath storage if possible
    *
    * @return this NDArray
    */
   public NDArray compress() {
      return this;
   }

   @Override
   public NDArray copy() {
      return copyData().setLabel(label).setPredicted(predicted).setWeight(weight);
   }

   /**
    * Copy data nd array.
    *
    * @return the nd array
    */
   protected abstract NDArray copyData();

   /**
    * Decrements the value at the given index by 1
    *
    * @param index the index to decrement
    * @return this NDArray
    */
   public NDArray decrement(int index) {
      return decrement(index, 1d);
   }

   /**
    * Decrements the value at the given index by a given amount
    *
    * @param index  the index to decrement
    * @param amount the amount to decrement
    * @return this NDArray
    */
   public NDArray decrement(int index, double amount) {
      set(index, get(index) - amount);
      return this;
   }

   /**
    * Decrements the value at the given subscript by 1
    *
    * @param i the index of the first dimension
    * @param j the index of the second dimension
    * @return this NDArray
    */
   public NDArray decrement(int i, int j) {
      return decrement(i, j, 1d);
   }

   /**
    * Decrements the value at the given subscript by a given amount
    *
    * @param i      the index of the first dimension
    * @param j      the index of the second dimension
    * @param amount the amount to decrement
    * @return this NDArray
    */
   public NDArray decrement(int i, int j, double amount) {
      set(i, j, get(i, j) - amount);
      return this;
   }

   /**
    * Diag nd array.
    *
    * @return the nd array
    */
   public NDArray diag() {
      if (isEmpty()) {
         return new EmptyNDArray();
      } else if (isScalar()) {
         return copy();
      }

      if (isColumnVector()) {
         NDArray toReturn = getFactory().zeros(shape().i, shape().i);
         for (int i = 0; i < shape().i; i++) {
            toReturn.set(i, i, get(i, 0));
         }
         return toReturn;
      } else if (isRowVector()) {
         NDArray toReturn = getFactory().zeros(shape().j, shape().j);
         for (int j = 0; j < shape().j; j++) {
            toReturn.set(j, j, get(0, j));
         }
         return toReturn;
      } else if (isSquare()) {
         NDArray toReturn = getFactory().zeros(shape());
         for (int i = 0; i < shape().i; i++) {
            if (i < shape().j) {
               toReturn.set(i, i, get(i, i));
            }
         }
         return toReturn;
      }

      throw new IllegalStateException("Rectangular matrices are not supported.");
   }

   /**
    * Divides a scalar value to each element in the NDArray
    *
    * @param scalar the value to divided
    * @return the new NDArray with the scalar value divided
    */
   public NDArray div(double scalar) {
      return mapSparse(d -> d / scalar);
   }

   /**
    * Divides the values in the other NDArray to this one element by element.
    *
    * @param other the other NDArray whose values will be divided
    * @return the new NDArray with the result of this / other
    * @throws IllegalArgumentException If the shape of this NDArray does not match that of the other NDArray
    */
   public NDArray div(@NonNull NDArray other) {
      return mapSparse(other, Math2::divide);
   }

   /**
    * Divides a column or row vector element division dividing the values in the other NDArray to each row or column in
    * this NDArray as specified by the given axis parameter.
    *
    * @param other the other NDArray whose values will be divided
    * @param axis  the axis
    * @return the new NDArray with the result of this / other
    * @throws IllegalArgumentException If the row/column shape of this NDArray does not match that of the other NDArray
    */
   public NDArray div(@NonNull NDArray other, @NonNull Axis axis) {
      return mapSparse(other, axis, Math2::divide);
   }

   /**
    * Divides a scalar value to each element in the NDArray in-place.
    *
    * @param scalar the value to divided
    * @return this NDArray with the scalar value divided
    */
   public NDArray divi(double scalar) {
      return mapiSparse(d -> d / scalar);
   }

   /**
    * Divides the values in the other NDArray to this one element by element in-place.
    *
    * @param other the other NDArray whose values will be divided
    * @return this NDArray with the result of this / other
    * @throws IllegalArgumentException If the shape of this NDArray does not match that of the other NDArray
    */
   public NDArray divi(@NonNull NDArray other) {
      return mapiSparse(other, Math2::divide);
   }

   /**
    * Divides a column or row vector element division dividing the values in the other NDArray to each row or column in
    * this NDArray as specified by the given axis parameter in-place.
    *
    * @param other the other NDArray whose values will be divided
    * @param axis  the axis
    * @return this NDArray with the result of this / other
    * @throws IllegalArgumentException If the row/column shape of this NDArray does not match that of the other NDArray
    */
   public NDArray divi(@NonNull NDArray other, @NonNull Axis axis) {
      return mapiSparse(other, axis, Math2::divide);
   }

   public NDArray dot(@NonNull NDArray other, @NonNull Axis axis) {
      NDArray dot = getFactory().zeros(axis, 1, axis.T(), shape().get(axis));
      if (axis == Axis.ROW) {
         for (int i = 0; i < numRows(); i++) {
            double sum = 0d;
            for (Iterator<Entry> itr = sparseRowIterator(i); itr.hasNext(); ) {
               Entry e = itr.next();
               sum += e.getValue() * other.get(axis.T(), e.get(axis), axis, e.get(axis.T()));
            }
            dot.set(i, 0, sum);
         }
      } else {
         for (int i = 0; i < numCols(); i++) {
            double sum = 0d;
            for (Iterator<Entry> itr = sparseRowIterator(i); itr.hasNext(); ) {
               Entry e = itr.next();
               sum += e.getValue() * other.get(axis.T(), e.get(axis), axis, e.get(axis.T()));
            }
            dot.set(0, i, sum);
         }
      }

      return dot;
   }

   /**
    * Calculates the dot product between this  NDArray and a given other
    *
    * @param other The other NDArray
    * @return The dot product
    * @throws IllegalArgumentException If the shapes do not match
    */
   public double dot(@NonNull NDArray other) {
      shape().checkDimensionMatch(other.shape());
      double dot = 0d;
      for (int i = 0; i < shape().i; i++) {
         for (int j = 0; j < shape().j; j++) {
            dot += get(i, j) * other.get(i, j);
         }
      }
      return dot;
   }

   @Override
   public boolean equals(Object o) {
      return o != null
                && o instanceof NDArray
                && shape().equals(Cast.<NDArray>as(o).shape())
                && Arrays.equals(Cast.<NDArray>as(o).toArray(), toArray());
   }

   /**
    * Convenience method for calculating <code>e^x</code> where <code>x</code> is the element value of the NDArray, i.e.
    * <code>Math.exp(x)</code>.
    *
    * @return the new NDArray
    */
   public NDArray exp() {
      NDArray toReturn = getFactory().ones(shape());
      forEachSparse(e -> toReturn.set(e.getIndex(), FastMath.exp(e.getValue())));
      return toReturn;
   }

   /**
    * Convenience method for calculating <code>e^x</code> where <code>x</code> is the element value of the NDArray
    * in-place, i.e. <code>Math.exp(x)</code>.
    *
    * @return this NDArray
    */
   public NDArray expi() {
      return mapi(FastMath::exp);
   }

   /**
    * Sets the values of elements in the NDArray to given value (in-place).
    *
    * @param value the value to assign to all elements.
    * @return this NDArray
    */
   public NDArray fill(double value) {
      mapi(d -> value);
      return this;
   }

   /**
    * Performs the given action for each element in the NDArray.
    *
    * @param consumer the consumer to use for processing the NDArray entries
    */
   public void forEach(@NonNull Consumer<Entry> consumer) {
      iterator().forEachRemaining(consumer);
   }

   /**
    * Processes each sparse entry in this NDArray using the given consumer
    *
    * @param consumer Entry consumer
    */
   public void forEachSparse(@NonNull Consumer<Entry> consumer) {
      sparseIterator().forEachRemaining(consumer);
   }

   /**
    * Processes each sparse entry, in an ordered fashion, in this NDArray using the given consumer
    *
    * @param consumer Entry consumer
    */
   public void forEachSparseOrdered(@NonNull Consumer<Entry> consumer) {
      sparseOrderedIterator().forEachRemaining(consumer);
   }

   /**
    * Gets the value of the NDArray at the given index. This method is useful for vectors and accessing storage
    * directly.
    *
    * @param index the index into the storage
    * @return the value at the given index
    * @throws IndexOutOfBoundsException if the index is invalid
    */
   public abstract double get(int index);

   /**
    * Gets the value of the NDArray at the given subscript <code>(r, c)</code>.
    *
    * @param i the r subscript
    * @param j the c subscript.
    * @return The value at <code>(r, c)</code>
    * @throws IndexOutOfBoundsException if the dimensions are invalid
    */
   public abstract double get(int i, int j);

   /**
    * Gets the value of the NDArray at the given subscript.
    *
    * @param subscript the subscript whose value we want
    * @return The value at <code>(r, c)</code>
    * @throws IndexOutOfBoundsException if the dimensions are invalid
    */
   public double get(@NonNull Subscript subscript) {
      return get(subscript.i, subscript.j);
   }

   /**
    * Gets the value of the NDArray at the given subscript .
    *
    * @param a1   the axis of the first subscript
    * @param dim1 the index of the first axis's dimension
    * @param a2   the axis of the second subscript
    * @param dim2 the index of the second axis's dimension
    * @return the element value
    * @throws IndexOutOfBoundsException if the dimensions are invalid
    */
   public double get(@NonNull Axis a1, int dim1, @NonNull Axis a2, int dim2) {
      int[] dims = {-1, -1};
      dims[a1.index] = dim1;
      dims[a2.index] = dim2;
      return get(dims[0], dims[1]);
   }

   /**
    * Gets the factory object for creating new instances of this type of NDArray.
    *
    * @return the factory
    */
   public abstract NDArrayFactory getFactory();

   /**
    * Gets label.
    *
    * @param <T> the type parameter
    * @return the label
    */
   public <T> T getLabel() {
      return Cast.as(label);
   }

   /**
    * Sets label.
    *
    * @param label the label
    * @return the label
    */
   public NDArray setLabel(Object label) {
      this.label = label;
      return this;
   }

   /**
    * Gets label as double.
    *
    * @return the label as double
    */
   public double getLabelAsDouble() {
      if (label == null) {
         return Double.NaN;
      }
      return Cast.<Number>as(label).doubleValue();
   }

   /**
    * Gets label as nd array.
    *
    * @return the label as nd array
    */
   public NDArray getLabelAsNDArray() {
      if (label == null) {
         return new EmptyNDArray();
      } else if (label instanceof Number) {
         return new ScalarNDArray(Cast.<Number>as(label).doubleValue());
      }
      return Cast.as(label);
   }

   /**
    * Gets label as nd array.
    *
    * @param dimension the dimension
    * @return the label as nd array
    */
   public NDArray getLabelAsNDArray(int dimension) {
      if (label == null) {
         return new EmptyNDArray();
      } else if (label instanceof Number) {
         return getFactory().zeros(dimension)
                            .set(Cast.<Number>as(label).intValue(), 1d);
      }
      return Cast.as(label);
   }

   /**
    * Gets predicted.
    *
    * @param <T> the type parameter
    * @return the predicted
    */
   public <T> T getPredicted() {
      return Cast.as(predicted);
   }

   /**
    * Sets predicted.
    *
    * @param predicted the predicted
    * @return the predicted
    */
   public NDArray setPredicted(Object predicted) {
      this.predicted = predicted;
      return this;
   }

   /**
    * Gets predicted as double.
    *
    * @return the predicted as double
    */
   public double getPredictedAsDouble() {
      if (predicted == null) {
         return Double.NaN;
      }
      return Cast.<Number>as(predicted).doubleValue();
   }

   /**
    * Gets predicted as nd array.
    *
    * @return the predicted as nd array
    */
   public NDArray getPredictedAsNDArray() {
      if (predicted == null) {
         return new EmptyNDArray();
      } else if (predicted instanceof Number) {
         return new ScalarNDArray(Cast.<Number>as(predicted).doubleValue());
      }
      return Cast.as(predicted);
   }

   /**
    * Gets a vector along the given axis at the given index
    *
    * @param index the index of the vector to return
    * @param axis  the axis the index belongs
    * @return An vector NDArray
    */
   public NDArray getVector(int index, @NonNull Axis axis) {
      Preconditions.checkElementIndex(index, shape().get(axis),
                                      "Invalid index " + index + " [0, " + shape().get(axis) + ")");
      NDArray toReturn = getFactory().zeros(shape().get(axis.T()), axis);
      for (int i = 0; i < shape().get(axis.T()); i++) {
         toReturn.set(i, get(axis, index, axis.T(), i));
      }
      return toReturn;
   }

   /**
    * Gets weight.
    *
    * @return the weight
    */
   public double getWeight() {
      return weight;
   }

   /**
    * Sets weight.
    *
    * @param weight the weight
    * @return the weight
    */
   public NDArray setWeight(double weight) {
      this.weight = weight;
      return this;
   }

   /**
    * Has label boolean.
    *
    * @return the boolean
    */
   public boolean hasLabel() {
      return label != null;
   }

   /**
    * Has predicted label boolean.
    *
    * @return the boolean
    */
   public boolean hasPredictedLabel() {
      return predicted != null;
   }

   /**
    * Increments the value of the element at the given index by 1
    *
    * @param index the index whose value will be incremented
    * @return this NDArray
    */
   public NDArray increment(int index) {
      return increment(index, 1d);
   }

   /**
    * Increments the value of the element at the given index by the given amount
    *
    * @param index  the index whose value will be incremented
    * @param amount the amount to increment by
    * @return this NDArray
    */
   public NDArray increment(int index, double amount) {
      set(index, get(index) + amount);
      return this;
   }

   /**
    * Increments the value of the element at the given subscript by 1
    *
    * @param i the index of the first dimension
    * @param j the index of the second dimension
    * @return this NDArray
    */
   public NDArray increment(int i, int j) {
      return increment(i, j, 1d);
   }

   /**
    * Increments the value of the element at the given index by the given amount
    *
    * @param i      the index of the first dimension
    * @param j      the index of the second dimension
    * @param amount the amount to increment by
    * @return this NDArray
    */
   public NDArray increment(int i, int j, double amount) {
      set(i, j, get(i, j) + amount);
      return this;
   }

   /**
    * Checks if this NDArray is a column vector, i.e. has 1 column and multiple rows
    *
    * @return true if column vector, false otherwise
    */
   public boolean isColumnVector() {
      return shape().i > 1 && shape().j == 1;
   }

   /**
    * Checks if the NDArray empty
    *
    * @return True if empty (shape of (0,0)), False if not
    */
   public boolean isEmpty() {
      return shape().i == 0 && shape().j == 0;
   }

   /**
    * Checks if this NDArray is a row vector, i.e. has 1 row and multiple colums
    *
    * @return true if row vector, false otherwise
    */
   public boolean isRowVector() {
      return shape().i == 1 && shape().j > 1;
   }

   /**
    * Checks if this NDArray is a scalar, i.e. 1 row and 1 column
    *
    * @return true if scalar, false otherwise
    */
   public boolean isScalar() {
      return shape().i == 1 && shape().j == 1;
   }

   /**
    * Is sparse boolean.
    *
    * @return the boolean
    */
   public abstract boolean isSparse();

   /**
    * Checks if this NDArray is square, i.e. the number of rows equals  the number of columns
    *
    * @return true if square, false otherwise
    */
   public boolean isSquare() {
      return shape().i == shape().j && shape().i > 1;
   }

   /**
    * Checks if the NDArray is a vector (dimension of one shape is 1)
    *
    * @return True if vector, False otherwise
    */
   public boolean isVector() {
      return isColumnVector() || isRowVector();
   }

   /**
    * Checks if the NDArray is a vector along the given axis
    *
    * @param axis The axis to check
    * @return True if vector along given axis
    */
   public boolean isVector(@NonNull Axis axis) {
      if (axis == Axis.ROW) {
         return isRowVector();
      }
      return isColumnVector();
   }

   /**
    * Iterator over the entries (subscripts, index, and value) of the NDArray
    *
    * @return the iterator
    */
   public abstract Iterator<Entry> iterator();

   /**
    * The single dimension length of the data, i.e. <code>numberOfRows * numberOfColumns</code>
    *
    * @return the length
    */
   public int length() {
      return shape().length();
   }

   /**
    * Applies the log function to each value in the NDArray
    *
    * @return new NDArray with logged values
    */
   public NDArray log() {
      return map(Math2::safeLog);
   }

   /**
    * Applies the log function to each value in the NDArray in-place
    *
    * @return this NDArray
    */
   public NDArray logi() {
      return mapi(Math2::safeLog);
   }

   public Axis majorMode() {
      return Axis.COlUMN;
   }

   /**
    * Applies the given operator to each element in this NDArray creating a new NDArray in the process.
    *
    * @param operator the operator to apply
    * @return the new NDArray with values calculated using the given operator
    */
   public NDArray map(@NonNull DoubleUnaryOperator operator) {
      NDArray toReturn = getFactory().zeros(shape());
      for (int i = 0; i < length(); i++) {
         toReturn.set(i, operator.applyAsDouble(get(i)));
      }
      return toReturn;
   }

   /**
    * Applies the given operator to each element in this NDArray and the given vector along the given axis creating a
    * new NDArray in the process.
    *
    * @param vector   the vector of values to combine with this NDArray
    * @param axis     the axis to apply the operator to
    * @param operator the operator to apply to the elements in this NDArray and the given vector
    * @return the new NDArray
    */
   public NDArray map(@NonNull NDArray vector, @NonNull Axis axis, @NonNull DoubleBinaryOperator operator) {
      shape().checkDimensionMatch(vector.shape(), axis.T());
      Preconditions.checkArgument(vector.isVector(axis), "Not a " + axis + " vector.");
      NDArray toReturn = getFactory().zeros(shape());
      for (int r = 0; r < shape().i; r++) {
         for (int c = 0; c < shape().j; c++) {
            toReturn.set(r, c, operator.applyAsDouble(get(r, c), vector.get(axis.T().select(r, c))));
         }
      }
      return toReturn;
   }

   /**
    * Applies an operation to the elements in this NDArray and given other NDArray using the given operator producing a
    * new NDArray as its outcome.
    *
    * @param other    the other NDArray to perform operation over
    * @param operator the operator to apply
    * @return the new NDArray
    */
   public NDArray map(@NonNull NDArray other, @NonNull DoubleBinaryOperator operator) {
      NDArray toReturn = getFactory().zeros(shape());
      if (majorMode() == other.majorMode()) {
         shape().checkLength(other.shape());
         for (int i = 0; i < length(); i++) {
            toReturn.set(i, operator.applyAsDouble(get(i), other.get(i)));
         }
      } else {
         shape().checkDimensionMatch(other.shape());
         for (int r = 0; r < shape().i; r++) {
            for (int c = 0; c < shape().j; c++) {
               toReturn.set(r, c, operator.applyAsDouble(get(r, c), other.get(r, c)));
            }
         }
      }
      return toReturn;
   }

   /**
    * Applies the given operator to elements in the NDArray if the their values test positive using given the
    * predicate.
    *
    * @param predicate the predicate to use to test values.
    * @param operator  the operator to apply
    * @return the new NDArray
    */
   public NDArray mapIf(@NonNull DoublePredicate predicate, @NonNull DoubleUnaryOperator operator) {
      final NDArray toReturn = getFactory().zeros(shape());
      for (int i = 0; i < length(); i++) {
         if (predicate.test(get(i))) {
            toReturn.set(i, operator.applyAsDouble(get(i)));
         } else {
            toReturn.set(i, get(i));
         }
      }
      return toReturn;
   }

   /**
    * Applies the given operator to each sparse element in this NDArray and the given vector along the given axis
    * creating a new NDArray in the process.
    *
    * @param vector   the vector of values to combine with this NDArray
    * @param axis     the axis to apply the operator to
    * @param operator the operator to apply to the elements in this NDArray and the given vector
    * @return the new NDArray
    */
   public NDArray mapSparse(@NonNull NDArray vector, @NonNull Axis axis, @NonNull DoubleBinaryOperator operator) {
      shape().checkDimensionMatch(vector.shape(), axis.T());
      Preconditions.checkArgument(vector.isVector(axis));
      NDArray toReturn = getFactory().zeros(shape());
      forEachSparse(e ->
                       toReturn.set(e.getI(),
                                    e.getJ(),
                                    operator.applyAsDouble(e.getValue(),
                                                           vector.get(axis.T().select(e.getI(), e.getJ()))))
                   );
      return toReturn;
   }

   /**
    * Applies an operation to the sparse elements in this NDArray and given other NDArray using the given operator
    * producing a new NDArray as its outcome.
    *
    * @param other    the other NDArray to perform operation over
    * @param operator the operator to apply
    * @return the new NDArray
    */
   public NDArray mapSparse(@NonNull NDArray other, @NonNull DoubleBinaryOperator operator) {
      shape().checkDimensionMatch(other.shape());
      NDArray toReturn = getFactory().zeros(shape());
      forEachSparse(entry -> toReturn.set(entry.getI(), entry.getJ(),
                                          operator.applyAsDouble(entry.getValue(),
                                                                 other.get(entry.getI(), entry.getJ()))));
      return toReturn;
   }

   /**
    * Applies the given operator to each sparse element in this NDArray creating a new NDArray in the process.
    *
    * @param operator the operator to apply
    * @return the new NDArray with values calculated using the given operator
    */
   public NDArray mapSparse(@NonNull DoubleUnaryOperator operator) {
      NDArray toReturn = getFactory().zeros(shape());
      forEachSparse(e -> toReturn.set(e.getIndex(), operator.applyAsDouble(e.getValue())));
      return toReturn;
   }

   /**
    * Applies an operation to the elements in this NDArray and given other NDArray using the given operator in-place.
    *
    * @param other    the other NDArray to perform operation over
    * @param operator the operator to apply
    * @return this NDArray
    */
   public NDArray mapi(@NonNull NDArray other, @NonNull DoubleBinaryOperator operator) {
      shape().checkDimensionMatch(other.shape());
      for (int r = 0; r < shape().i; r++) {
         for (int c = 0; c < shape().j; c++) {
            set(r, c, operator.applyAsDouble(get(r, c), other.get(r, c)));
         }
      }
      return this;
   }

   /**
    * Applies the given operator to each element in this NDArray in-place.
    *
    * @param operator the operator to apply
    * @return this NDArray with values calculated using the given operator
    */
   public NDArray mapi(@NonNull DoubleUnaryOperator operator) {
      for (int i = 0; i < length(); i++) {
         set(i, operator.applyAsDouble(get(i)));
      }
      return this;
   }

   /**
    * Applies the given operator to each element in this NDArray and the given vector along the given axis in-place.
    *
    * @param vector   the vector of values to combine with this NDArray
    * @param axis     the axis to apply the operator to
    * @param operator the operator to apply to the elements in this NDArray and the given vector
    * @return this NDArray with operator applied
    */
   public NDArray mapi(@NonNull NDArray vector, @NonNull Axis axis, @NonNull DoubleBinaryOperator operator) {
      shape().checkDimensionMatch(axis.T(), vector.shape(), axis.T());
      Preconditions.checkArgument(vector.isVector(axis));
      for (int r = 0; r < shape().i; r++) {
         for (int c = 0; c < shape().j; c++) {
            set(r, c, operator.applyAsDouble(get(r, c), vector.get(axis.T().select(r, c))));
         }
      }
      return this;
   }

   /**
    * Applies the given operator to elements in the NDArray if the their values test positive using given the predicate
    * in-place.
    *
    * @param predicate the predicate to use to test values.
    * @param operator  the operator to apply
    * @return this NDArray
    */
   public NDArray mapiIf(@NonNull DoublePredicate predicate, @NonNull DoubleUnaryOperator operator) {
      for (int i = 0; i < length(); i++) {
         if (predicate.test(get(i))) {
            set(i, operator.applyAsDouble(get(i)));
         }
      }
      return this;
   }

   /**
    * Applies the given operator to each sparse element in this NDArray and the given vector along the given axis
    * creating a new NDArray in the process.
    *
    * @param vector   the vector of values to combine with this NDArray
    * @param axis     the axis to apply the operator to
    * @param operator the operator to apply to the elements in this NDArray and the given vector
    * @return the new NDArray
    */
   public NDArray mapiSparse(@NonNull NDArray vector, @NonNull Axis axis, @NonNull DoubleBinaryOperator operator) {
      shape().checkDimensionMatch(axis.T(), vector.shape(), axis.T());
      Preconditions.checkArgument(vector.isVector(axis), "Not a " + axis + " vector.");
      forEachSparse(e ->
                       e.setValue(operator.applyAsDouble(e.getValue(), vector.get(axis.T().select(e.getI(), e.getJ()))))
                   );
      return this;
   }

   /**
    * Applies an operation to the sparse elements in this NDArray and given other NDArray using the given operator
    * in-place.
    *
    * @param other    the other NDArray to perform operation over
    * @param operator the operator to apply
    * @return this NDArray
    */
   public NDArray mapiSparse(@NonNull NDArray other, @NonNull DoubleBinaryOperator operator) {
      shape().checkDimensionMatch(other.shape());
      forEachSparse(
         entry -> entry.setValue(operator.applyAsDouble(entry.getValue(), other.get(entry.getI(), entry.getJ()))));
      return this;
   }

   /**
    * Applies the given operator to each sparse element in this NDArray in-place.
    *
    * @param operator the operator to apply
    * @return this NDArray with values calculated using the given operator
    */
   public NDArray mapiSparse(@NonNull DoubleUnaryOperator operator) {
      forEachSparse(entry -> entry.setValue(operator.applyAsDouble(entry.getValue())));
      return this;
   }

   /**
    * Calculates the maximum value in the NDArray
    *
    * @return the maximum value in the NDArray
    */
   public double max() {
      double max = Double.NEGATIVE_INFINITY;
      for (Iterator<Entry> itr = sparseIterator(); itr.hasNext(); ) {
         double v = itr.next().getValue();
         if (v > max) {
            max = v;
         }
      }
      return max;
   }

   /**
    * Calculates the maximum values along each axis
    *
    * @param axis The axis to calculate the max for
    * @return An NDArray of the max values
    */
   public NDArray max(@NonNull Axis axis) {
      NDArray toReturn = getFactory().zeros(shape().get(axis), axis.T());
      toReturn.mapi(d -> Double.NEGATIVE_INFINITY);
      forEach(entry -> {
         if (toReturn.get(entry.get(axis)) < entry.getValue()) {
            toReturn.set(entry.get(axis), entry.getValue());
         }
      });
      return toReturn;
   }

   /**
    * Calculates the mean across all values in the NDArray
    *
    * @return the mean
    */
   public double mean() {
      return sum() / length();
   }

   /**
    * Calculates the mean along each axis
    *
    * @param axis The axis to calculate the mean for
    * @return An NDArray of the mean
    */
   public NDArray mean(@NonNull Axis axis) {
      return sum(axis).divi(shape().get(axis.T()));
   }

   /**
    * Calculates the minimum value in the NDArray
    *
    * @return the minimum value in the NDArray
    */
   public double min() {
      double min = Double.POSITIVE_INFINITY;
      for (Iterator<Entry> itr = sparseIterator(); itr.hasNext(); ) {
         double v = itr.next().getValue();
         if (v < min) {
            min = v;
         }
      }
      return min;
   }

   /**
    * Calculates the minimum values along each axis
    *
    * @param axis The axis to calculate the min for
    * @return An NDArray of the min values
    */
   public NDArray min(@NonNull Axis axis) {
      NDArray toReturn = getFactory().zeros(shape().get(axis), axis);
      toReturn.mapi(d -> Double.POSITIVE_INFINITY);
      forEach(entry -> {
         if (toReturn.get(entry.get(axis)) > entry.getValue()) {
            toReturn.set(entry.get(axis), entry.getValue());
         }
      });
      return toReturn;
   }

   /**
    * Calculates the product of this and the given NDArray (i.e. matrix multiplication).
    *
    * @param other The other NDArray to multiple
    * @return a new NDArray that is the result of this X other
    */
   public NDArray mmul(@NonNull NDArray other) {
      shape().checkCanMultiply(other.shape());
      NDArray toReturn = getFactory().zeros(shape().i, other.shape().j);
      for (int r = 0; r < shape().i; r++) {
         for (int c = 0; c < other.shape().j; c++) {
            Iterator<Entry> ri = sparseRowIterator(r);
            Iterator<Entry> ci = other.sparseColumnIterator(c);
            double sum = 0;
            Entry rE = null;
            Entry cE = null;
            while (ri.hasNext() && ci.hasNext()) {
               if (rE == null) rE = ri.next();
               if (cE == null) cE = ci.next();

               if (rE.getJ() == cE.getI()) {
                  sum += rE.getValue() * cE.getValue();
                  rE = null;
                  cE = null;
               } else if (rE.getJ() < cE.getI()) {
                  while (ri.hasNext() && rE.getJ() < cE.getI()) {
                     rE = ri.next();
                  }
               } else if (rE.getJ() > cE.getI()) {
                  while (ci.hasNext() && rE.getJ() > cE.getI()) {
                     cE = ci.next();
                  }
               }
            }
//            for (int c2 = 0; c2 < shape().j; c2++) {
//               sum += get(r, c2) * other.get(c2, c);
//            }
            toReturn.set(r, c, sum);
         }
      }
      return toReturn;
   }

   /**
    * Multiplies a scalar value to each element in the NDArray
    *
    * @param scalar the value to multiplied
    * @return the new NDArray with the scalar value multiplied
    */
   public NDArray mul(double scalar) {
      return mapSparse(d -> d * scalar);
   }

   /**
    * Multiplies the values in the other NDArray to this one element by element.
    *
    * @param other the other NDArray whose values will be multiplied
    * @return the new NDArray with the result of this * other
    * @throws IllegalArgumentException If the shape of this NDArray does not match that of the other NDArray
    */
   public NDArray mul(@NonNull NDArray other) {
      return mapSparse(other, Math2::multiply);
   }

   /**
    * Performs a column or row vector element multiplication multiplying the values in the other NDArray to each row or
    * column in this NDArray as specified by the given axis parameter.
    *
    * @param other the other NDArray whose values will be multiplied
    * @param axis  the axis
    * @return the new NDArray with the result of this * other
    * @throws IllegalArgumentException If the row/column shape of this NDArray does not match that of the other NDArray
    */
   public NDArray mul(@NonNull NDArray other, @NonNull Axis axis) {
      return mapSparse(other, axis, Math2::multiply);
   }

   /**
    * Multiplies a scalar value to each element in the NDArray in-place
    *
    * @param scalar the value to multiplied
    * @return this NDArray with the scalar value multiplied
    */
   public NDArray muli(double scalar) {
      return mapiSparse(d -> d * scalar);
   }

   /**
    * Multiplies the values in the other NDArray to this one element by element in-place.
    *
    * @param other the other NDArray whose values will be multiplied
    * @return this NDArray with the result of this * other
    * @throws IllegalArgumentException If the shape of this NDArray does not match that of the other NDArray
    */
   public NDArray muli(@NonNull NDArray other) {
      return mapiSparse(other, Math2::multiply);
   }

   /**
    * Performs a column or row vector element multiplication multiplying the values in the other NDArray to each row or
    * column in this NDArray as specified by the given axis parameter in-place.
    *
    * @param other the other NDArray whose values will be multiplied
    * @param axis  the axis
    * @return this NDArray with the result of this * other
    * @throws IllegalArgumentException If the row/column shape of this NDArray does not match that of the other NDArray
    */
   public NDArray muli(@NonNull NDArray other, @NonNull Axis axis) {
      return mapiSparse(other, axis, Math2::multiply);
   }

   public NDArray muliVector(int index, @NonNull NDArray vector, @NonNull Axis axis) {
      shape().checkDimensionMatch(axis.T(), vector.shape(), axis.T());
      Preconditions.checkArgument(vector.isVector(axis));
      if (axis == Axis.ROW) {
         vector.forEachSparse(entry -> {
            set(index, entry.getJ(), get(index, entry.getJ()) * entry.getValue());
         });
      } else {
         vector.forEachSparse(entry -> {
            set(entry.getI(), index, get(entry.getI(), index) * entry.getValue());
         });
      }
      return this;
   }

   /**
    * Negates the values in the NDArray
    *
    * @return the new NDArray with negated values
    */
   public NDArray neg() {
      return map(d -> -d);
   }

   /**
    * Negates the values in the NDArray in-place
    *
    * @return this NDArray
    */
   public NDArray negi() {
      return mapi(d -> -d);
   }

   /**
    * Calculates the L1-norm of the NDArray
    *
    * @return the L1-norm
    */
   public double norm1() {
      return Streams.asStream(sparseIterator())
                    .mapToDouble(e -> Math.abs(e.getValue()))
                    .sum();
   }

   /**
    * Calculates the L2-norm (magnitude) of the NDArray
    *
    * @return the L2-norm
    */
   public double norm2() {
      return Math.sqrt(sumOfSquares());
   }

   /**
    * Num cols int.
    *
    * @return the int
    */
   public int numCols() {
      return shape().j;
   }

   /**
    * Num rows int.
    *
    * @return the int
    */
   public int numRows() {
      return shape().i;
   }

   /**
    * Raises the value of each element in the NDArray by the given power.
    *
    * @param pow the power to raise values to
    * @return the new NDArray
    */
   public NDArray pow(double pow) {
      return mapSparse(d -> d == 0 ? 0d : FastMath.pow(d, pow));
   }

   /**
    * Raises the value of each element in the NDArray by the given power in-place.
    *
    * @param pow the power to raise values to
    * @return this NDArray
    */
   public NDArray powi(double pow) {
      return mapiSparse(d -> d == 0 ? 0d : FastMath.pow(d, pow));
   }

   /**
    * Pretty prints the NDArray
    *
    * @param stream the stream to print the NDArray to
    */
   public void pprint(PrintStream stream) {
      final DecimalFormat df = new DecimalFormat("0.000");
      PrintWriter writer = new PrintWriter(stream);
      writer.print('[');
      for (int i = 0; i < shape().i; i++) {
         if (i > 0) {
            writer.println("],");
            writer.print(" [");
         } else {
            writer.print('[');
         }
         writer.print(df.format(get(i, 0)));
         for (int j = 1; j < shape().j; j++) {
            writer.print(", ");
            writer.print(df.format(get(i, j)));
         }
      }
      writer.println("]]");
      writer.flush();
   }

   /**
    * Divides each element's value from the given scalar (e.g. scalar - element)
    *
    * @param scalar the value to divide
    * @return the new NDArray with the scalar value divided
    */
   public NDArray rdiv(double scalar) {
      return map(d -> scalar / d);
   }

   /**
    * Divides the values in the this NDArray from the other NDArray.
    *
    * @param other the other NDArray whose values will be divided from
    * @return the new NDArray with the result of other / this
    * @throws IllegalArgumentException If the shape of this NDArray does not match that of the other NDArray
    */
   public NDArray rdiv(@NonNull NDArray other) {
      return rmap(other, Math2::divide);
   }

   /**
    * Performs a column or row vector division dividing the values in this NDArray from the other NDArray to each row or
    * column in this NDArray as specified by the given axis parameter.
    *
    * @param other the other NDArray whose values will be divided
    * @param axis  the axis
    * @return the new NDArray with the result of this / other
    * @throws IllegalArgumentException If the row/column shape of this NDArray does not match that of the other NDArray
    */
   public NDArray rdiv(@NonNull NDArray other, @NonNull Axis axis) {
      return rmap(other, axis, Math2::divide);
   }

   /**
    * Divides each element's value from the given scalar (e.g. scalar - element) in place
    *
    * @param scalar the value to divide
    * @return thisNDArray with the scalar value divided
    */
   public NDArray rdivi(double scalar) {
      return mapi(d -> scalar / d);
   }

   /**
    * Divides the values in the this NDArray from the other NDArray in-place.
    *
    * @param other the other NDArray whose values will be divided from
    * @return this NDArray with the result of other / this
    * @throws IllegalArgumentException If the shape of this NDArray does not match that of the other NDArray
    */
   public NDArray rdivi(@NonNull NDArray other) {
      return rmapi(other, Math2::divide);
   }

   /**
    * Performs a column or row vector division dividing the values in this NDArray from the other NDArray to each row or
    * column in this NDArray as specified by the given axis parameter in-place.
    *
    * @param other the other NDArray whose values will be divided
    * @param axis  the axis
    * @return this NDArray with the result of this / other
    * @throws IllegalArgumentException If the row/column shape of this NDArray does not match that of the other NDArray
    */
   public NDArray rdivi(@NonNull NDArray other, @NonNull Axis axis) {
      return rmapi(other, axis, Math2::divide);
   }

   /**
    * Applies an operation to the elements in this NDArray and given other NDArray using the given operator creating a
    * new NDArray along the given axis. Is applied with the other NDArray element's value being the first parameter in
    * the operator call.
    *
    * @param vector   the other NDArray to perform operation over
    * @param axis     the axis to apply the operator over
    * @param operator the operator to apply
    * @return this NDArray
    */
   public NDArray rmap(@NonNull NDArray vector, @NonNull Axis axis, @NonNull DoubleBinaryOperator operator) {
      shape().checkDimensionMatch(axis.T(), vector.shape(), axis.T());
      Preconditions.checkArgument(vector.isVector(axis));
      NDArray toReturn = getFactory().zeros(shape());
      for (int r = 0; r < shape().i; r++) {
         for (int c = 0; c < shape().j; c++) {
            toReturn.set(r, c, operator.applyAsDouble(vector.get(axis.T().select(r, c)), get(r, c)));
         }
      }
      return toReturn;
   }

   /**
    * Applies an operation to the elements in this NDArray and given other NDArray using the given operator creating a
    * new NDArray. Is applied with the other NDArray element's value being the first parameter in the operator call.
    *
    * @param other    the other NDArray to perform operation over
    * @param operator the operator to apply
    * @return the new NDArray
    */
   public NDArray rmap(@NonNull NDArray other, @NonNull DoubleBinaryOperator operator) {
      shape().checkDimensionMatch(other.shape());
      NDArray toReturn = getFactory().zeros(shape());
      for (int r = 0; r < shape().i; r++) {
         for (int c = 0; c < shape().j; c++) {
            toReturn.set(r, c, operator.applyAsDouble(other.get(r, c), get(r, c)));
         }
      }
      return toReturn;
   }

   /**
    * Applies an operation to the elements in this NDArray and given other NDArray using the given operator in-place. Is
    * applied with the other NDArray element's value being the first parameter in the operator call.
    *
    * @param other    the other NDArray to perform operation over
    * @param operator the operator to apply
    * @return this NDArray
    */
   public NDArray rmapi(@NonNull NDArray other, @NonNull DoubleBinaryOperator operator) {
      shape().checkDimensionMatch(other.shape());
      for (int r = 0; r < shape().i; r++) {
         for (int c = 0; c < shape().j; c++) {
            set(r, c, operator.applyAsDouble(other.get(r, c), get(r, c)));
         }
      }
      return this;
   }

   /**
    * Applies an operation to the elements in this NDArray and given other NDArray using the given operator in-place
    * along the given axis. Is applied with the other NDArray element's value being the first parameter in the operator
    * call.
    *
    * @param vector   the other NDArray to perform operation over
    * @param axis     the axis to apply the operator over
    * @param operator the operator to apply
    * @return this NDArray
    */
   public NDArray rmapi(@NonNull NDArray vector, @NonNull Axis axis, @NonNull DoubleBinaryOperator operator) {
      shape().checkDimensionMatch(axis.T(), vector.shape(), axis.T());
      Preconditions.checkArgument(vector.isVector(axis));
      for (int r = 0; r < shape().i; r++) {
         for (int c = 0; c < shape().j; c++) {
            set(r, c, operator.applyAsDouble(vector.get(axis.T().select(r, c)), get(r, c)));
         }
      }
      return this;
   }

   /**
    * Subtracts each element's value from the given scalar (e.g. scalar - element)
    *
    * @param scalar the value to subtract
    * @return the new NDArray with the scalar value subtracted
    */
   public NDArray rsub(double scalar) {
      return map(d -> scalar - d);
   }

   /**
    * Subtracts the values in the this NDArray from the other NDArray.
    *
    * @param other the other NDArray whose values will be subtracted from
    * @return the new NDArray with the result of other - this
    * @throws IllegalArgumentException If the shape of this NDArray does not match that of the other NDArray
    */
   public NDArray rsub(@NonNull NDArray other) {
      return other.map(this, Math2::subtract);
   }

   /**
    * Performs a column or row vector subtraction subtracting the values in this NDArray from the other NDArray to each
    * row or column in this NDArray as specified by the given axis parameter.
    *
    * @param other the other NDArray whose values will be subtracted
    * @param axis  the axis
    * @return the new NDArray with the result of this - other
    * @throws IllegalArgumentException If the row/column shape of this NDArray does not match that of the other NDArray
    */
   public NDArray rsub(@NonNull NDArray other, @NonNull Axis axis) {
      return rmap(other, axis, Math2::subtract);
   }

   /**
    * Subtracts each element's value from the given scalar (e.g. scalar - element)  in-place.
    *
    * @param scalar the value to subtract
    * @return the new NDArray with the scalar value subtracted
    */
   public NDArray rsubi(double scalar) {
      return mapi(d -> scalar - d);
   }

   /**
    * Subtracts the values in the this NDArray from the other NDArray in-place.
    *
    * @param other the other NDArray whose values will be subtracted from
    * @return the new NDArray with the result of other - this
    * @throws IllegalArgumentException If the shape of this NDArray does not match that of the other NDArray
    */
   public NDArray rsubi(@NonNull NDArray other) {
      return rmapi(other, Math2::subtract);
   }

   /**
    * Performs a column or row vector subtraction subtracting the values in this NDArray from the other NDArray to each
    * row or column in this NDArray as specified by the given axis parameter in-place.
    *
    * @param other the other NDArray whose values will be subtracted
    * @param axis  the axis
    * @return the new NDArray with the result of this - other
    * @throws IllegalArgumentException If the row/column shape of this NDArray does not match that of the other NDArray
    */
   public NDArray rsubi(@NonNull NDArray other, @NonNull Axis axis) {
      return rmapi(other, axis, Math2::subtract);
   }

   /**
    * Scalar value double.
    *
    * @return the double
    */
   public double scalarValue() {
      Preconditions.checkState(isScalar(), "NDArray must be scalar");
      return get(0);
   }

   /**
    * Selects all values matching the given predicate
    *
    * @param predicate the predicate to test
    * @return new NDArray with values passing the given predicate and zeros elsewhere
    */
   public NDArray select(@NonNull DoublePredicate predicate) {
      final NDArray toReturn = getFactory().zeros(shape());
      forEach(entry -> {
         if (predicate.test(entry.getValue())) {
            toReturn.set(entry.getI(), entry.getJ(), entry.getValue());
         }
      });
      return toReturn;
   }

   /**
    * Selects all values in this NDArray whose corresponding element in the given predicate NDArray is not zero.
    *
    * @param predicate the predicate NDArray test
    * @return new NDArray with values passing the given predicate and zeros elsewhere
    */
   public NDArray select(@NonNull NDArray predicate) {
      shape().checkDimensionMatch(predicate.shape());
      NDArray toReturn = getFactory().zeros(shape());
      predicate.forEachSparse(entry -> {
         if (entry.getValue() != 0) {
            toReturn.set(entry.getI(), entry.getJ(), entry.getValue());
         }
      });
      return toReturn;
   }

   /**
    * Selects all values matching the given predicate in-place
    *
    * @param predicate the predicate to test
    * @return this NDArray with values passing the given predicate and zeros elsewhere
    */
   public NDArray selecti(@NonNull DoublePredicate predicate) {
      forEach(entry -> {
         if (!predicate.test(entry.getValue())) {
            entry.setValue(0d);
         }
      });
      return this;
   }

   /**
    * Selects all values in this NDArray whose corresponding element in the given predicate NDArray is not zero
    * in-place.
    *
    * @param predicate the predicate NDArray test
    * @return this NDArray with values passing the given predicate and zeros elsewhere
    */
   public NDArray selecti(@NonNull NDArray predicate) {
      shape().checkDimensionMatch(predicate.shape());
      predicate.forEachSparse(entry -> {
         if (entry.getValue() == 0) {
            set(entry.getI(), entry.getJ(), 0d);
         }
      });
      return this;
   }

   /**
    * Sets the value at the given index (useful for vectors and direct storage access)
    *
    * @param index the index to set
    * @param value the new value to set
    * @return this NDArray
    */
   public abstract NDArray set(int index, double value);

   /**
    * Sets the value at the given subscript.
    *
    * @param r     the subscript of the first dimension
    * @param c     the subscript of the second dimension
    * @param value the value to set
    * @return this NDArray
    */
   public abstract NDArray set(int r, int c, double value);

   /**
    * Sets the value at the given subscript
    *
    * @param subscript the subscript of the element value to set
    * @param value     the new value
    * @return this NDArray
    */
   public NDArray set(@NonNull Subscript subscript, double value) {
      return set(subscript.i, subscript.j, value);
   }

   /**
    * Sets the values along the given axis at the given index to those in the given vector in-place.
    *
    * @param index  the index of the row/column
    * @param vector the vector whose values are to replace those in this NDArray
    * @param axis   the axis (row/column) being set
    * @return this NDArray
    */
   public NDArray setVector(int index, @NonNull NDArray vector, @NonNull Axis axis) {
      Preconditions.checkArgument(index >= 0 && index < shape().get(axis), "Invalid index");
      shape().checkDimensionMatch(vector.shape(), axis.T());
      Preconditions.checkArgument(vector.isVector(axis));
      for (int i = 0; i < vector.length(); i++) {
         set(axis.select(index, i), //IF given axis row THEN index ELSE i
             axis.T().select(index, i), //IF given axis == row THEN index ELSE i
             vector.get(i));
      }
      return this;
   }

   /**
    * The shape of the NDArray
    *
    * @return the shape
    */
   public abstract Shape shape();

   /**
    * The sparse size of the NDArray
    *
    * @return the sparse size of the NDArray
    */
   public int size() {
      return length();
   }

   /**
    * Slices vector-based NDArrays using the given range of indexes (inclusive from, exclusive to)
    *
    * @param from the index to start slicing at
    * @param to   the index to slice up to, but not including
    * @return the new sliced NDArray
    * @throws IllegalArgumentException if the NDArrays is not a vector
    */
   public NDArray slice(int from, int to) {
      if (isRowVector()) {
         NDArray toReturn = getFactory().zeros(to - from, Axis.ROW);
         for (int i = from; i < to; i++) {
            toReturn.set(i, get(i));
         }
         return toReturn;
      } else if (isColumnVector()) {
         NDArray toReturn = getFactory().zeros(to - from, Axis.COlUMN);
         for (int i = from; i < to; i++) {
            toReturn.set(i, get(i));
         }
         return toReturn;
      }
      throw new IllegalArgumentException();
   }

   /**
    * Slices the NDArray using the given subscript ranges (inclusive from, exclusive to)
    *
    * @param iFrom the index of the first dimension to start slicing at
    * @param iTo   the index of the first dimension  to slice up to, but not including
    * @param jFrom the index of the second dimension to start slicing at
    * @param jTo   the index of the second dimension  to slice up to, but not including
    * @return the new sliced NDArray
    */
   public NDArray slice(int iFrom, int iTo, int jFrom, int jTo) {
      NDArray toReturn = getFactory().zeros(iTo - iFrom, jTo - jFrom);
      for (int i = iFrom; i < iTo; i++) {
         for (int j = jFrom; j < jTo; j++) {
            toReturn.set(i, j, get(i, j));
         }
      }
      return toReturn;
   }

   /**
    * Slices the NDArray by taking all elements along the given axis for the given indexes
    *
    * @param axis    the axis to slice
    * @param indexes the indexes of the axis to slice
    * @return the sliced NDArray
    */
   public NDArray slice(@NonNull Axis axis, @NonNull int... indexes) {
      NDArray toReturn;
      if (axis == Axis.ROW) {
         toReturn = getFactory().zeros(indexes.length, shape().j);
      } else {
         toReturn = getFactory().zeros(shape().i, indexes.length);
      }
      for (int r = 0; r < indexes.length; r++) {
         toReturn.setVector(r, this.getVector(r, axis), axis);
      }
      return toReturn;
   }

   public Iterator<NDArray.Entry> sparseColumnIterator(int column) {
      return null;
   }

   /**
    * Sparse iterator over the entries in the NDArray (will act like <code>iterator</code> for dense implementations)
    *
    * @return the iterator
    */
   public Iterator<Entry> sparseIterator() {
      return iterator();
   }

   public Iterator<Entry> sparseOrderedColumnIterator(int column) {
      return null;
   }

   /**
    * Sparse iterator over the entries in the NDArray (will act like <code>iterator</code> for dense implementations)
    * ordered by subscript.
    *
    * @return the iterator
    */
   public Iterator<Entry> sparseOrderedIterator() {
      return iterator();
   }

   public Iterator<Entry> sparseOrderedRowIterator(int row) {
      return null;
   }

   public Iterator<Entry> sparseRowIterator(int row) {
      return null;
   }

   /**
    * Statistics enhanced double statistics.
    *
    * @return the enhanced double statistics
    */
   public EnhancedDoubleStatistics statistics() {
      EnhancedDoubleStatistics toReturn = new EnhancedDoubleStatistics();
      forEach(e -> toReturn.accept(e.getValue()));
      return toReturn;
   }

   /**
    * Subtracts a scalar value to each element in the NDArray
    *
    * @param scalar the value to subtract
    * @return the new NDArray with the scalar value subtracted
    */
   public NDArray sub(double scalar) {
      if (scalar == 0) {
         return copy();
      }
      return map(d -> d - scalar);
   }

   /**
    * Subtracts the values in the other NDArray to this one.
    *
    * @param other the other NDArray whose values will be subtracted
    * @return the new NDArray with the result of this - other
    * @throws IllegalArgumentException If the shape of this NDArray does not match that of the other NDArray
    */
   public NDArray sub(@NonNull NDArray other) {
      shape().checkDimensionMatch(other.shape());
      NDArray toReturn = copy();
      other.forEachSparse(e -> toReturn.decrement(e.getI(), e.getJ(), e.getValue()));
      return toReturn;
   }

   /**
    * Performs a column or row vector subtraction subtracting the values in the other NDArray to each row or column in
    * this NDArray as specified by the given axis parameter.
    *
    * @param other the other NDArray whose values will be subtracted
    * @param axis  the axis
    * @return the new NDArray with the result of this - other
    * @throws IllegalArgumentException If the row/column shape of this NDArray does not match that of the other NDArray
    */
   public NDArray sub(@NonNull NDArray other, @NonNull Axis axis) {
      return map(other, axis, Math2::subtract);
   }

   /**
    * Subtracts a scalar value to each element in the NDArray in-place
    *
    * @param scalar the value to subtract
    * @return the new NDArray with the scalar value subtracted
    */
   public NDArray subi(double scalar) {
      if (scalar == 0) {
         return this;
      }
      return mapi(d -> d - scalar);
   }

   /**
    * Subtracts the values in the other NDArray to this one  in-place.
    *
    * @param other the other NDArray whose values will be subtracted
    * @return the new NDArray with the result of this - other
    * @throws IllegalArgumentException If the shape of this NDArray does not match that of the other NDArray
    */
   public NDArray subi(@NonNull NDArray other) {
      shape().checkDimensionMatch(other.shape());
      other.forEachSparse(e -> decrement(e.getI(), e.getJ(), e.getValue()));
      return this;
   }

   /**
    * Performs a column or row vector subtraction subtracting the values in the other NDArray to each row or column in
    * this NDArray as specified by the given axis parameter  in-place.
    *
    * @param other the other NDArray whose values will be subtracted
    * @param axis  the axis
    * @return the new NDArray with the result of this - other
    * @throws IllegalArgumentException If the row/column shape of this NDArray does not match that of the other NDArray
    */
   public NDArray subi(@NonNull NDArray other, @NonNull Axis axis) {
      return mapi(other, axis, Math2::subtract);
   }

   public NDArray subiVector(int index, @NonNull NDArray vector, @NonNull Axis axis) {
      shape().checkDimensionMatch(axis.T(), vector.shape(), axis.T());
      Preconditions.checkArgument(vector.isVector(axis));
      if (axis == Axis.ROW) {
         vector.forEachSparse(entry -> {
            decrement(index, entry.getJ(), entry.getValue());
         });
      } else {
         vector.forEachSparse(entry -> {
            decrement(entry.getI(), index, entry.getValue());
         });
      }
      return this;
   }

   /**
    * Calculates the sum along each axis
    *
    * @param axis The axis to calculate the sum for
    * @return An NDArray of the sum
    */
   public NDArray sum(@NonNull Axis axis) {
      NDArray toReturn = getFactory().zeros(shape().get(axis), axis.T());
      forEachSparse(entry -> toReturn.set(entry.get(axis), toReturn.get(entry.get(axis)) + entry.getValue()));
      return toReturn;
   }

   /**
    * Calculates the sum of all values in the NDArray
    *
    * @return the sum all values
    */
   public double sum() {
      return Streams.asStream(sparseIterator())
                    .mapToDouble(Entry::getValue)
                    .sum();
   }

   /**
    * Sum of squares double.
    *
    * @return the double
    */
   public double sumOfSquares() {
      return Streams.asStream(sparseIterator())
                    .mapToDouble(e -> FastMath.pow(e.getValue(), 2))
                    .sum();
   }

   /**
    * Tests the given predicate on the values in the NDArray returning 1 when TRUE and 0 when FALSE
    *
    * @param predicate the predicate to test
    * @return new NDArray with test results
    */
   public NDArray test(@NonNull DoublePredicate predicate) {
      NDArray toReturn = getFactory().zeros(shape());
      forEach(entry -> {
         if (predicate.test(entry.getValue())) {
            toReturn.set(entry.getI(), entry.getJ(), 1d);
         }
      });
      return toReturn;
   }

   /**
    * Tests the given predicate on the values in the NDArray returning 1 when TRUE and 0 when FALSE in-place
    *
    * @param predicate the predicate to test
    * @return this with test results
    */
   public NDArray testi(@NonNull DoublePredicate predicate) {
      forEach(entry -> {
         if (predicate.test(entry.getValue())) {
            entry.setValue(1d);
         } else {
            entry.setValue(0d);
         }
      });
      return this;
   }

   /**
    * Gets a 2D array view of the NDArray
    *
    * @return 2D array view of the data
    */
   public double[][] to2DArray() {
      final double[][] array = new double[shape().i][shape().j];
      forEachSparse(e -> array[e.getI()][e.getJ()] = e.getValue());
      return array;
   }

   /**
    * The data in the NDArray as a 1d array
    *
    * @return 1d array view of thedata
    */
   public double[] toArray() {
      double[] toReturn = new double[length()];
      forEachSparse(e -> toReturn[e.getIndex()] = e.getValue());
      return toReturn;
   }

   /**
    * Generates a boolean view of the NDArray
    *
    * @return 1d array of boolean values
    */
   public boolean[] toBooleanArray() {
      boolean[] toReturn = new boolean[length()];
      forEachSparse(e -> toReturn[e.getIndex()] = e.getValue() == 1);
      return toReturn;
   }

   /**
    * Generates a JBlas DoubleMatrix view of the data
    *
    * @return the double matrix
    */
   public DoubleMatrix toDoubleMatrix() {
      return new DoubleMatrix(shape().i, shape().j, toArray());
   }

   /**
    * Generates a float view of the NDArray
    *
    * @return 1d array of float values
    */
   public float[] toFloatArray() {
      float[] toReturn = new float[length()];
      forEachSparse(e -> toReturn[e.getIndex()] = (float) e.getValue());
      return toReturn;
   }

   /**
    * Generates a JBlas FloatMatrix view of the data
    *
    * @return the float matrix
    */
   public FloatMatrix toFloatMatrix() {
      return new FloatMatrix(shape().i, shape().j, toFloatArray());
   }

   /**
    * Generates an int view of the NDArray
    *
    * @return 1d array of int values
    */
   public int[] toIntArray() {
      int[] toReturn = new int[length()];
      forEachSparse(e -> toReturn[e.getIndex()] = (int) e.getValue());
      return toReturn;
   }

   @Override
   public String toString() {
      return Arrays.toString(toArray());
   }

   /**
    * To unit vector nd array.
    *
    * @return the nd array
    */
   public NDArray toUnitVector() {
      Preconditions.checkArgument(isVector(), "NDArray must be a vector");
      double mag = norm2();
      return div(mag);
   }

   /**
    * Sets all element values to zero
    *
    * @return this NDArray
    */
   public NDArray zero() {
      return fill(0d);
   }


   /**
    * Defines an entry in the NDArray, which is the dimensions (i and j), the index (vector, direct storage), and
    * value.
    */
   public interface Entry extends Serializable {

      /**
       * Gets the subscript index for the given axis
       *
       * @param axis the axis whose subscript index is wanted
       * @return the subscript index
       */
      default int get(@NonNull Axis axis) {
         return axis == Axis.ROW ? getI() : getJ();
      }

      /**
       * Gets the subscript index of the first dimension.
       *
       * @return the subscript index of the first dimension
       */
      int getI();

      /**
       * Gets the index of the element in the NDArray (useful for vectors)
       *
       * @return the index
       */
      int getIndex();

      /**
       * Gets the subscript index of the second dimension.
       *
       * @return the subscript index of the second dimension
       */
      int getJ();

      /**
       * Gets the value at the current subscript/index
       *
       * @return the value
       */
      double getValue();

      /**
       * Sets the value at the current subscript/index
       *
       * @param value the new value
       */
      void setValue(double value);

   }//END OF Entry

}//END OF NDArray
