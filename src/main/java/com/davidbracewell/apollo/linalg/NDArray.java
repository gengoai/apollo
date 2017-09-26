package com.davidbracewell.apollo.linalg;

import com.davidbracewell.Copyable;
import com.davidbracewell.Math2;
import com.davidbracewell.collection.Streams;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.guava.common.base.Preconditions;
import lombok.NonNull;
import org.apache.commons.math3.util.FastMath;
import org.apache.mahout.math.set.OpenDoubleHashSet;
import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;

import java.io.PrintStream;
import java.io.PrintWriter;
import java.io.Serializable;
import java.text.DecimalFormat;
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
public interface NDArray extends Copyable<NDArray> {

   /**
    * Flips the matrix on its diagonal switching the rows and columns
    *
    * @return the transposed array
    */
   default NDArray T() {
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
   default NDArray add(double scalar) {
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
   default NDArray add(@NonNull NDArray other) {
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
   default NDArray add(@NonNull NDArray other, @NonNull Axis axis) {
      return map(other, axis, Math2::add);
   }

   /**
    * Adds a scalar value to each element in the NDArray in-place
    *
    * @param scalar the value to add
    * @return this NDArray with the scalar value added
    */
   default NDArray addi(double scalar) {
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
   default NDArray addi(@NonNull NDArray other) {
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
   default NDArray addi(@NonNull NDArray other, @NonNull Axis axis) {
      return mapi(other, axis, Math2::add);
   }

   /**
    * Calculates the index of the maximum along each row or column based on the given axis.
    *
    * @param axis the axis (row/column) to calculate the max for
    * @return int array of row/column indexes relating to max values
    */
   default int[] argMax(@NonNull Axis axis) {
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
   default int[] argMin(@NonNull Axis axis) {
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
   default NDArray compress() {
      return this;
   }

   /**
    * Decrements the value at the given index by 1
    *
    * @param index the index to decrement
    * @return this NDArray
    */
   default NDArray decrement(int index) {
      return decrement(index, 1d);
   }

   /**
    * Decrements the value at the given index by a given amount
    *
    * @param index  the index to decrement
    * @param amount the amount to decrement
    * @return this NDArray
    */
   default NDArray decrement(int index, double amount) {
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
   default NDArray decrement(int i, int j) {
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
   default NDArray decrement(int i, int j, double amount) {
      set(i, j, get(i, j) - amount);
      return this;
   }

   default NDArray diag() {
      NDArray toReturn = getFactory().zeros(shape());
      for (int i = 0; i < shape().i; i++) {
         if (i < shape().j) {
            toReturn.set(i, i, get(i, i));
         }
      }
      return toReturn;
   }

   /**
    * Divides a scalar value to each element in the NDArray
    *
    * @param scalar the value to divided
    * @return the new NDArray with the scalar value divided
    */
   default NDArray div(double scalar) {
      return mapSparse(d -> d / scalar);
   }

   /**
    * Divides the values in the other NDArray to this one element by element.
    *
    * @param other the other NDArray whose values will be divided
    * @return the new NDArray with the result of this / other
    * @throws IllegalArgumentException If the shape of this NDArray does not match that of the other NDArray
    */
   default NDArray div(@NonNull NDArray other) {
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
   default NDArray div(@NonNull NDArray other, @NonNull Axis axis) {
      return mapSparse(other, axis, Math2::divide);
   }

   /**
    * Divides a scalar value to each element in the NDArray in-place.
    *
    * @param scalar the value to divided
    * @return this NDArray with the scalar value divided
    */
   default NDArray divi(double scalar) {
      return mapiSparse(d -> d / scalar);
   }

   /**
    * Divides the values in the other NDArray to this one element by element in-place.
    *
    * @param other the other NDArray whose values will be divided
    * @return this NDArray with the result of this / other
    * @throws IllegalArgumentException If the shape of this NDArray does not match that of the other NDArray
    */
   default NDArray divi(@NonNull NDArray other) {
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
   default NDArray divi(@NonNull NDArray other, @NonNull Axis axis) {
      return mapiSparse(other, axis, Math2::divide);
   }

   /**
    * Calculates the dot product between this  NDArray and a given other
    *
    * @param other The other NDArray
    * @return The dot product
    * @throws IllegalArgumentException If the shapes do not match
    */
   default double dot(@NonNull NDArray other) {
      shape().checkDimensionMatch(other.shape());
      if (size() < other.size()) {
         return Streams.asStream(sparseIterator())
                       .mapToDouble(e -> e.getValue() * other.get(e.getI(), e.getJ()))
                       .sum();
      } else {
         return Streams.asStream(other.sparseIterator())
                       .mapToDouble(e -> e.getValue() * get(e.getI(), e.getJ()))
                       .sum();
      }
   }

   /**
    * Convenience method for calculating <code>e^x</code> where <code>x</code> is the element value of the NDArray, i.e.
    * <code>Math.exp(x)</code>.
    *
    * @return the new NDArray
    */
   default NDArray exp() {
      return map(FastMath::exp);
   }

   /**
    * Convenience method for calculating <code>e^x</code> where <code>x</code> is the element value of the NDArray
    * in-place, i.e. <code>Math.exp(x)</code>.
    *
    * @return this NDArray
    */
   default NDArray expi() {
      return mapi(FastMath::exp);
   }

   /**
    * Sets the values of elements in the NDArray to given value (in-place).
    *
    * @param value the value to assign to all elements.
    * @return this NDArray
    */
   default NDArray fill(double value) {
      mapi(d -> value);
      return this;
   }

   /**
    * Performs the given action for each element in the NDArray.
    *
    * @param consumer the consumer to use for processing the NDArray entries
    */
   default void forEach(@NonNull Consumer<Entry> consumer) {
      iterator().forEachRemaining(consumer);
   }

   /**
    * Processes each sparse entry in this NDArray using the given consumer
    *
    * @param consumer Entry consumer
    */
   default void forEachSparse(@NonNull Consumer<Entry> consumer) {
      sparseIterator().forEachRemaining(consumer);
   }

   /**
    * Processes each sparse entry, in an ordered fashion, in this NDArray using the given consumer
    *
    * @param consumer Entry consumer
    */
   default void forEachSparseOrdered(@NonNull Consumer<Entry> consumer) {
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
   double get(int index);

   /**
    * Gets the value of the NDArray at the given subscript <code>(r, c)</code>.
    *
    * @param i the r subscript
    * @param j the c subscript.
    * @return The value at <code>(r, c)</code>
    * @throws IndexOutOfBoundsException if the dimensions are invalid
    */
   double get(int i, int j);

   /**
    * Gets the value of the NDArray at the given subscript.
    *
    * @param subscript the subscript whose value we want
    * @return The value at <code>(r, c)</code>
    * @throws IndexOutOfBoundsException if the dimensions are invalid
    */
   default double get(@NonNull Subscript subscript) {
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
   default double get(@NonNull Axis a1, int dim1, @NonNull Axis a2, int dim2) {
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
   NDArrayFactory getFactory();

   /**
    * Gets the label / key, if any, associated with the NDArray.
    *
    * @param <T> the expected type of the label
    * @return the label
    */
   <T> T getLabel();

   /**
    * Sets the label associated with the NDArray.
    *
    * @param label the new Label
    * @return this NDArray
    */
   NDArray setLabel(Object label);

   /**
    * Gets the label/key, if any, associated with the NDArray as a double value (casting).
    *
    * @return the label as double
    */
   default double getLabelAsDouble() {
      return Cast.as(getLabel());
   }

   /**
    * Gets the label/key, if any, associated with the NDArray as a double set. Will create a new set of size 1 if the
    * label is not already a set.
    *
    * @return the label as double set
    */
   default OpenDoubleHashSet getLabelAsDoubleSet() {
      if (getLabel() instanceof OpenDoubleHashSet) {
         return Cast.as(getLabel());
      }
      OpenDoubleHashSet set = new OpenDoubleHashSet(1);
      set.add(getLabelAsDouble());
      return set;
   }

   /**
    * Gets a vector along the given axis at the given index
    *
    * @param index the index of the vector to return
    * @param axis  the axis the index belongs
    * @return An vector NDArray
    */
   default NDArray getVector(int index, @NonNull Axis axis) {
      Preconditions.checkElementIndex(index, shape().get(axis),
                                      "Invalid index " + index + " [0, " + shape().get(axis) + ")");
      NDArray toReturn = getFactory().zeros(shape().get(axis.T()), axis);
      for (int i = 0; i < shape().get(axis.T()); i++) {
         toReturn.set(i, get(axis, index, axis.T(), i));
      }
      return toReturn;
   }

   /**
    * Checks if the NDArray has a label
    *
    * @return True if the NDArray has a label (non null label value), False otherwise (null label value)
    */
   default boolean hasLabel() {
      return getLabel() != null;
   }

   /**
    * Increments the value of the element at the given index by 1
    *
    * @param index the index whose value will be incremented
    * @return this NDArray
    */
   default NDArray increment(int index) {
      return increment(index, 1d);
   }

   /**
    * Increments the value of the element at the given index by the given amount
    *
    * @param index  the index whose value will be incremented
    * @param amount the amount to increment by
    * @return this NDArray
    */
   default NDArray increment(int index, double amount) {
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
   default NDArray increment(int i, int j) {
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
   default NDArray increment(int i, int j, double amount) {
      set(i, j, get(i, j) + amount);
      return this;
   }

   /**
    * Checks if this NDArray is a column vector, i.e. has 1 column and multiple rows
    *
    * @return true if column vector, false otherwise
    */
   default boolean isColumnVector() {
      return shape().i > 1 && shape().j == 1;
   }

   /**
    * Checks if the NDArray empty
    *
    * @return True if empty (shape of (0,0)), False if not
    */
   default boolean isEmpty() {
      return shape().i == 0 && shape().j == 0;
   }

   /**
    * Checks if this NDArray is a row vector, i.e. has 1 row and multiple colums
    *
    * @return true if row vector, false otherwise
    */
   default boolean isRowVector() {
      return shape().i == 1 && shape().j > 1;
   }

   /**
    * Checks if this NDArray is a scalar, i.e. 1 row and 1 column
    *
    * @return true if scalar, false otherwise
    */
   default boolean isScalar() {
      return shape().i == 1 && shape().j == 1;
   }

   /**
    * Checks if this NDArray is square, i.e. the number of rows equals  the number of columns
    *
    * @return true if square, false otherwise
    */
   default boolean isSquare() {
      return shape().i == shape().j && shape().i > 1;
   }

   /**
    * Checks if the NDArray is a vector (dimension of one shape is 1)
    *
    * @return True if vector, False otherwise
    */
   default boolean isVector() {
      return isColumnVector() || isRowVector();
   }

   /**
    * Checks if the NDArray is a vector along the given axis
    *
    * @param axis The axis to check
    * @return True if vector along given axis
    */
   default boolean isVector(@NonNull Axis axis) {
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
   Iterator<Entry> iterator();

   /**
    * The single dimension length of the data, i.e. <code>numberOfRows * numberOfColumns</code>
    *
    * @return the length
    */
   default int length() {
      return shape().length();
   }

   /**
    * Applies the log function to each value in the NDArray
    *
    * @return new NDArray with logged values
    */
   default NDArray log() {
      return map(Math2::safeLog);
   }

   /**
    * Applies the log function to each value in the NDArray in-place
    *
    * @return this NDArray
    */
   default NDArray logi() {
      return mapi(Math2::safeLog);
   }

   /**
    * Applies the given operator to each element in this NDArray creating a new NDArray in the process.
    *
    * @param operator the operator to apply
    * @return the new NDArray with values calculated using the given operator
    */
   default NDArray map(@NonNull DoubleUnaryOperator operator) {
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
   default NDArray map(@NonNull NDArray vector, @NonNull Axis axis, @NonNull DoubleBinaryOperator operator) {
      shape().checkDimensionMatch(axis.T(), vector.shape(), axis.T());
      Preconditions.checkArgument(vector.isVector(axis));
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
   default NDArray map(@NonNull NDArray other, @NonNull DoubleBinaryOperator operator) {
      shape().checkDimensionMatch(other.shape());
      NDArray toReturn = getFactory().zeros(shape());
      for (int r = 0; r < shape().i; r++) {
         for (int c = 0; c < shape().j; c++) {
            toReturn.set(r, c, operator.applyAsDouble(get(r, c), other.get(r, c)));
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
   default NDArray mapIf(@NonNull DoublePredicate predicate, @NonNull DoubleUnaryOperator operator) {
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
   default NDArray mapSparse(@NonNull NDArray vector, @NonNull Axis axis, @NonNull DoubleBinaryOperator operator) {
      shape().checkDimensionMatch(axis.T(), vector.shape(), axis.T());
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
   default NDArray mapSparse(@NonNull NDArray other, @NonNull DoubleBinaryOperator operator) {
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
   default NDArray mapSparse(@NonNull DoubleUnaryOperator operator) {
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
   default NDArray mapi(@NonNull NDArray other, @NonNull DoubleBinaryOperator operator) {
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
   default NDArray mapi(@NonNull DoubleUnaryOperator operator) {
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
   default NDArray mapi(@NonNull NDArray vector, @NonNull Axis axis, @NonNull DoubleBinaryOperator operator) {
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
   default NDArray mapiIf(@NonNull DoublePredicate predicate, @NonNull DoubleUnaryOperator operator) {
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
   default NDArray mapiSparse(@NonNull NDArray vector, @NonNull Axis axis, @NonNull DoubleBinaryOperator operator) {
      shape().checkDimensionMatch(axis.T(), vector.shape(), axis.T());
      Preconditions.checkArgument(vector.isVector(axis));
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
   default NDArray mapiSparse(@NonNull NDArray other, @NonNull DoubleBinaryOperator operator) {
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
   default NDArray mapiSparse(@NonNull DoubleUnaryOperator operator) {
      forEachSparse(entry -> entry.setValue(operator.applyAsDouble(entry.getValue())));
      return this;
   }

   /**
    * Calculates the maximum value in the NDArray
    *
    * @return the maximum value in the NDArray
    */
   default double max() {
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
   default NDArray max(@NonNull Axis axis) {
      NDArray toReturn = getFactory().zeros(shape().get(axis), axis);
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
   default double mean() {
      return sum() / length();
   }

   /**
    * Calculates the mean along each axis
    *
    * @param axis The axis to calculate the mean for
    * @return An NDArray of the mean
    */
   default NDArray mean(@NonNull Axis axis) {
      return sum(axis).divi(shape().get(axis.T()));
   }

   /**
    * Calculates the minimum value in the NDArray
    *
    * @return the minimum value in the NDArray
    */
   default double min() {
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
   default NDArray min(@NonNull Axis axis) {
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
   default NDArray mmul(@NonNull NDArray other) {
      shape().checkCanMultiply(other.shape());
      NDArray toReturn = getFactory().zeros(shape().i, other.shape().j);
      for (int r = 0; r < shape().i; r++) {
         for (int c = 0; c < other.shape().j; c++) {
            double sum = 0;
            for (int c2 = 0; c2 < shape().j; c2++) {
               sum += get(r, c2) * other.get(c2, c);
            }
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
   default NDArray mul(double scalar) {
      return mapSparse(d -> d * scalar);
   }

   /**
    * Multiplies the values in the other NDArray to this one element by element.
    *
    * @param other the other NDArray whose values will be multiplied
    * @return the new NDArray with the result of this * other
    * @throws IllegalArgumentException If the shape of this NDArray does not match that of the other NDArray
    */
   default NDArray mul(@NonNull NDArray other) {
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
   default NDArray mul(@NonNull NDArray other, @NonNull Axis axis) {
      return mapSparse(other, axis, Math2::multiply);
   }

   /**
    * Multiplies a scalar value to each element in the NDArray in-place
    *
    * @param scalar the value to multiplied
    * @return this NDArray with the scalar value multiplied
    */
   default NDArray muli(double scalar) {
      return mapiSparse(d -> d * scalar);
   }

   /**
    * Multiplies the values in the other NDArray to this one element by element in-place.
    *
    * @param other the other NDArray whose values will be multiplied
    * @return this NDArray with the result of this * other
    * @throws IllegalArgumentException If the shape of this NDArray does not match that of the other NDArray
    */
   default NDArray muli(@NonNull NDArray other) {
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
   default NDArray muli(@NonNull NDArray other, @NonNull Axis axis) {
      return mapiSparse(other, axis, Math2::multiply);
   }

   /**
    * Negates the values in the NDArray
    *
    * @return the new NDArray with negated values
    */
   default NDArray neg() {
      return map(d -> -d);
   }

   /**
    * Negates the values in the NDArray in-place
    *
    * @return this NDArray
    */
   default NDArray negi() {
      return mapi(d -> -d);
   }

   /**
    * Calculates the L1-norm of the NDArray
    *
    * @return the L1-norm
    */
   default double norm1() {
      return Streams.asStream(sparseIterator())
                    .mapToDouble(e -> Math.abs(e.getValue()))
                    .sum();
   }

   /**
    * Calculates the L2-norm (magnitude) of the NDArray
    *
    * @return the L2-norm
    */
   default double norm2() {
      return Math.sqrt(Streams.asStream(sparseIterator())
                              .mapToDouble(e -> FastMath.pow(e.getValue(), 2))
                              .sum());
   }

   /**
    * Raises the value of each element in the NDArray by the given power.
    *
    * @param pow the power to raise values to
    * @return the new NDArray
    */
   default NDArray pow(double pow) {
      return map(d -> FastMath.pow(d, pow));
   }

   /**
    * Raises the value of each element in the NDArray by the given power in-place.
    *
    * @param pow the power to raise values to
    * @return this NDArray
    */
   default NDArray powi(double pow) {
      return mapi(d -> FastMath.pow(d, pow));
   }

   /**
    * Pretty prints the NDArray
    *
    * @param stream the stream to print the NDArray to
    */
   default void pprint(PrintStream stream) {
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
   default NDArray rdiv(double scalar) {
      return map(d -> scalar / d);
   }

   /**
    * Divides the values in the this NDArray from the other NDArray.
    *
    * @param other the other NDArray whose values will be divided from
    * @return the new NDArray with the result of other / this
    * @throws IllegalArgumentException If the shape of this NDArray does not match that of the other NDArray
    */
   default NDArray rdiv(@NonNull NDArray other) {
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
   default NDArray rdiv(@NonNull NDArray other, @NonNull Axis axis) {
      return rmap(other, axis, Math2::divide);
   }

   /**
    * Divides each element's value from the given scalar (e.g. scalar - element) in place
    *
    * @param scalar the value to divide
    * @return thisNDArray with the scalar value divided
    */
   default NDArray rdivi(double scalar) {
      return mapi(d -> scalar / d);
   }

   /**
    * Divides the values in the this NDArray from the other NDArray in-place.
    *
    * @param other the other NDArray whose values will be divided from
    * @return this NDArray with the result of other / this
    * @throws IllegalArgumentException If the shape of this NDArray does not match that of the other NDArray
    */
   default NDArray rdivi(@NonNull NDArray other) {
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
   default NDArray rdivi(@NonNull NDArray other, @NonNull Axis axis) {
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
   default NDArray rmap(@NonNull NDArray vector, @NonNull Axis axis, @NonNull DoubleBinaryOperator operator) {
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
   default NDArray rmap(@NonNull NDArray other, @NonNull DoubleBinaryOperator operator) {
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
   default NDArray rmapi(@NonNull NDArray other, @NonNull DoubleBinaryOperator operator) {
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
   default NDArray rmapi(@NonNull NDArray vector, @NonNull Axis axis, @NonNull DoubleBinaryOperator operator) {
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
   default NDArray rsub(double scalar) {
      return map(d -> scalar - d);
   }

   /**
    * Subtracts the values in the this NDArray from the other NDArray.
    *
    * @param other the other NDArray whose values will be subtracted from
    * @return the new NDArray with the result of other - this
    * @throws IllegalArgumentException If the shape of this NDArray does not match that of the other NDArray
    */
   default NDArray rsub(@NonNull NDArray other) {
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
   default NDArray rsub(@NonNull NDArray other, @NonNull Axis axis) {
      return rmap(other, axis, Math2::subtract);
   }

   /**
    * Subtracts each element's value from the given scalar (e.g. scalar - element)  in-place.
    *
    * @param scalar the value to subtract
    * @return the new NDArray with the scalar value subtracted
    */
   default NDArray rsubi(double scalar) {
      return mapi(d -> scalar - d);
   }

   /**
    * Subtracts the values in the this NDArray from the other NDArray in-place.
    *
    * @param other the other NDArray whose values will be subtracted from
    * @return the new NDArray with the result of other - this
    * @throws IllegalArgumentException If the shape of this NDArray does not match that of the other NDArray
    */
   default NDArray rsubi(@NonNull NDArray other) {
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
   default NDArray rsubi(@NonNull NDArray other, @NonNull Axis axis) {
      return rmapi(other, axis, Math2::subtract);
   }

   /**
    * Selects all values matching the given predicate
    *
    * @param predicate the predicate to test
    * @return new NDArray with values passing the given predicate and zeros elsewhere
    */
   default NDArray select(@NonNull DoublePredicate predicate) {
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
   default NDArray select(@NonNull NDArray predicate) {
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
   default NDArray selecti(@NonNull DoublePredicate predicate) {
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
   default NDArray selecti(@NonNull NDArray predicate) {
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
   NDArray set(int index, double value);

   /**
    * Sets the value at the given subscript.
    *
    * @param r     the subscript of the first dimension
    * @param c     the subscript of the second dimension
    * @param value the value to set
    * @return this NDArray
    */
   NDArray set(int r, int c, double value);

   /**
    * Sets the value at the given subscript
    *
    * @param subscript the subscript of the element value to set
    * @param value     the new value
    * @return this NDArray
    */
   default NDArray set(@NonNull Subscript subscript, double value) {
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
   default NDArray setVector(int index, @NonNull NDArray vector, @NonNull Axis axis) {
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
   Shape shape();

   /**
    * The sparse size of the NDArray
    *
    * @return the sparse size of the NDArray
    */
   default int size() {
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
   default NDArray slice(int from, int to) {
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
   default NDArray slice(int iFrom, int iTo, int jFrom, int jTo) {
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
   default NDArray slice(@NonNull Axis axis, @NonNull int... indexes) {
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

   /**
    * Sparse iterator over the entries in the NDArray (will act like <code>iterator</code> for dense implementations)
    *
    * @return the iterator
    */
   default Iterator<Entry> sparseIterator() {
      return iterator();
   }

   /**
    * Sparse iterator over the entries in the NDArray (will act like <code>iterator</code> for dense implementations)
    * ordered by subscript.
    *
    * @return the iterator
    */
   default Iterator<Entry> sparseOrderedIterator() {
      return iterator();
   }

   /**
    * Subtracts a scalar value to each element in the NDArray
    *
    * @param scalar the value to subtract
    * @return the new NDArray with the scalar value subtracted
    */
   default NDArray sub(double scalar) {
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
   default NDArray sub(@NonNull NDArray other) {
      return map(other, Math2::subtract);
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
   default NDArray sub(@NonNull NDArray other, @NonNull Axis axis) {
      return map(other, axis, Math2::subtract);
   }

   /**
    * Subtracts a scalar value to each element in the NDArray in-place
    *
    * @param scalar the value to subtract
    * @return the new NDArray with the scalar value subtracted
    */
   default NDArray subi(double scalar) {
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
   default NDArray subi(@NonNull NDArray other) {
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
   default NDArray subi(@NonNull NDArray other, @NonNull Axis axis) {
      return mapi(other, axis, Math2::subtract);
   }

   /**
    * Calculates the sum along each axis
    *
    * @param axis The axis to calculate the sum for
    * @return An NDArray of the sum
    */
   default NDArray sum(@NonNull Axis axis) {
      NDArray toReturn = getFactory().zeros(shape().get(axis), axis);
      forEach(entry -> toReturn.set(entry.get(axis), toReturn.get(entry.get(axis)) + entry.getValue()));
      return toReturn;
   }

   /**
    * Calculates the sum of all values in the NDArray
    *
    * @return the sum all values
    */
   default double sum() {
      return Streams.asStream(sparseIterator())
                    .mapToDouble(Entry::getValue)
                    .sum();
   }

   /**
    * Tests the given predicate on the values in the NDArray returning 1 when TRUE and 0 when FALSE
    *
    * @param predicate the predicate to test
    * @return new NDArray with test results
    */
   default NDArray test(@NonNull DoublePredicate predicate) {
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
   default NDArray testi(@NonNull DoublePredicate predicate) {
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
   default double[][] to2DArray() {
      final double[][] array = new double[shape().i][shape().j];
      forEachSparse(e -> array[e.getI()][e.getJ()] = e.getValue());
      return array;
   }

   /**
    * The data in the NDArray as a 1d array
    *
    * @return 1d array view of thedata
    */
   default double[] toArray() {
      double[] toReturn = new double[length()];
      forEachSparse(e -> toReturn[e.getIndex()] = e.getValue());
      return toReturn;
   }

   /**
    * Generates a boolean view of the NDArray
    *
    * @return 1d array of boolean values
    */
   default boolean[] toBooleanArray() {
      boolean[] toReturn = new boolean[length()];
      forEachSparse(e -> toReturn[e.getIndex()] = e.getValue() == 1);
      return toReturn;
   }

   /**
    * Generates a JBlas DoubleMatrix view of the data
    *
    * @return the double matrix
    */
   default DoubleMatrix toDoubleMatrix() {
      return new DoubleMatrix(shape().i, shape().j, toArray());
   }

   /**
    * Generates a float view of the NDArray
    *
    * @return 1d array of float values
    */
   default float[] toFloatArray() {
      float[] toReturn = new float[length()];
      forEachSparse(e -> toReturn[e.getIndex()] = (float) e.getValue());
      return toReturn;
   }

   /**
    * Generates a JBlas FloatMatrix view of the data
    *
    * @return the float matrix
    */
   default FloatMatrix toFloatMatrix() {
      return new FloatMatrix(shape().i, shape().j, toFloatArray());
   }

   /**
    * Generates an int view of the NDArray
    *
    * @return 1d array of int values
    */
   default int[] toIntArray() {
      int[] toReturn = new int[length()];
      forEachSparse(e -> toReturn[e.getIndex()] = (int) e.getValue());
      return toReturn;
   }

   /**
    * Sets all element values to zero
    *
    * @return this NDArray
    */
   default NDArray zero() {
      return fill(0d);
   }


   /**
    * Defines an entry in the NDArray, which is the dimensions (i and j), the index (vector, direct storage), and
    * value.
    */
   interface Entry extends Serializable {

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
