/*
 * (c) 2005 David B. Bracewell
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 *
 */

package com.gengoai.apollo.linear;

import com.gengoai.Copyable;
import com.gengoai.Validation;
import com.gengoai.conversion.Cast;
import org.jblas.DoubleMatrix;

import java.io.Serializable;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoublePredicate;
import java.util.function.DoubleUnaryOperator;

/**
 * An n-dimension array of float values used for vectors, matrices, and tensors.
 *
 * @author David B. Bracewell
 */
public abstract class NDArray implements Serializable, Copyable<NDArray> {
   private Object label = null;
   private Object predicted = null;
   private double weight = 1d;
   /**
    * The Shape.
    */
   protected final Shape shape;

   /**
    * Instantiates a new NDArray.
    *
    * @param shape The shape of the new NDArray
    */
   protected NDArray(Shape shape) {
      Validation.notNull(shape);
      this.shape = shape.copy();
   }

   private double asDouble(Object object) {
      if (object == null) {
         return Double.NaN;
      } else if (object instanceof NDArray) {
         NDArray array = Cast.as(object);
         if (array.shape.isScalar()) {
            return array.scalar();
         }
         return array.argmax();
      }
      return Cast.<Number>as(object).doubleValue();
   }

   private NDArray asNDArray(Object o, int dimension) {
      if (o == null) {
         return com.gengoai.apollo.linear.NDArrayFactory.ND.empty();
      } else if (o instanceof Number) {
         Number numLabel = Cast.as(o);
         if (dimension == 1) {
            return com.gengoai.apollo.linear.NDArrayFactory.ND.scalar(numLabel.floatValue());
         }
         return com.gengoai.apollo.linear.NDArrayFactory.ND.array(dimension).set(numLabel.intValue(), 1f);
      }
      NDArray nd = Cast.as(o, NDArray.class);
      Validation.notNull(nd, "Cannot create NDArray from object.");
      return nd;
   }

   /**
    * Compacts the memory usages of sparse NDArrays.
    *
    * @return this NDArray
    */
   public abstract NDArray compact();

   /**
    * Gets the weight associated with the NDArray.
    *
    * @return the weight
    */
   public double getWeight() {
      return weight;
   }

   /**
    * Sets the weight associated with the NDArray.
    *
    * @param weight the weight
    * @return this NDArray
    */
   public NDArray setWeight(double weight) {
      this.weight = (float) weight;
      return this;
   }

   /**
    * Creates a new NDArray made up of sub-portions of the slices.
    *
    * @param fromRow the index of the row to start slicing from
    * @param toRow   the index of the row to end the slicing at
    * @param fromCol the index of the column to start slicing from
    * @param toCol   the index of the column to end slicing at
    * @return the NDArray
    */
   public abstract NDArray getSubMatrix(int fromRow, int toRow, int fromCol, int toCol);

   /**
    * Gets the label associated with the NDArray as a double value.
    *
    * @return the label as double
    */
   public double getLabelAsDouble() {
      return asDouble(label);
   }

   /**
    * Gets the label associated with the NDArray as an NDArray
    *
    * @return the label as NDArray
    */
   public NDArray getLabelAsNDArray() {
      return getLabelAsNDArray(1);
   }

   /**
    * Gets the label associated with this NDArray as an NDArray (vector) with desired dimension.
    *
    * @param dimension the dimension
    * @return the label as nd array
    */
   public NDArray getLabelAsNDArray(int dimension) {
      return asNDArray(label, dimension);
   }

   /**
    * Gets the predicted label associated with this NDArray.
    *
    * @param <T> the type parameter
    * @return the predicted label
    */
   public <T> T getPredicted() {
      return Cast.as(predicted);
   }

   /**
    * Sets the predicted label for this NDArray.
    *
    * @param predicted the predicted label
    * @return this NDArray
    */
   public NDArray setPredicted(Object predicted) {
      this.predicted = predicted;
      return this;
   }

   /**
    * Gets the predicted label associated with the NDArray as a double value.
    *
    * @return the predicted label as double
    */
   public double getPredictedAsDouble() {
      return asDouble(predicted);
   }

   /**
    * Gets the predicted label associated with the NDArray as an NDArray
    *
    * @return the predicted label as NDArray
    */
   public NDArray getPredictedAsNDArray() {
      return asNDArray(predicted, 1);
   }

   /**
    * Gets the predicted label associated with this NDArray as an NDArray (vector) with desired dimension.
    *
    * @param dimension the dimension
    * @return the predicted label as NDArray
    */
   public NDArray getPredictedAsNDArray(int dimension) {
      return asNDArray(predicted, dimension);
   }


   /**
    * Unitizes the NDArray by dividing the values by L2 Norm (per slice)
    *
    * @return Unitized version of this NDArray
    */
   public abstract NDArray unitize();

   /**
    * Gets the label associated with the NDArray
    *
    * @param <T> the type of the label
    * @return the label
    */
   public <T> T getLabel() {
      return Cast.as(label);
   }

   /**
    * Sets the label associated with the NDArray
    *
    * @param label the label
    * @return This NDArray
    */
   public NDArray setLabel(Object label) {
      this.label = label;
      return this;
   }

   /**
    * Flips the matrix on its diagonal switching the rows and columns. (This is done per slice)
    *
    * @return the transposed array
    */
   public abstract NDArray T();

   /**
    * Adds a scalar value to each element in the NDArray
    *
    * @param value the value to add
    * @return the new NDArray with the scalar value added
    */
   public abstract NDArray add(double value);

   /**
    * Adds the values in the other NDArray to this one.
    *
    * @param rhs the other NDArray whose values will be added
    * @return the new NDArray with the result of this + other
    */
   public abstract NDArray add(NDArray rhs);

   /**
    * Adds the values in the other NDArray to each column in this one.
    *
    * @param rhs the other NDArray whose values will be added
    * @return the new NDArray
    */
   public abstract NDArray addColumnVector(NDArray rhs);

   /**
    * Adds the values in the other NDArray to each row in this one.
    *
    * @param rhs the other NDArray whose values will be added
    * @return the new NDArray
    */
   public abstract NDArray addRowVector(NDArray rhs);

   /**
    * Adds a scalar value to each element in the NDArray in-place
    *
    * @param value the value to add
    * @return this NDArray with the scalar value added
    */
   public abstract NDArray addi(double value);

   /**
    * Adds the values in the other NDArray to this one in-place.
    *
    * @param rhs the other NDArray whose values will be added
    * @return this NDArray with the result of this + other
    */
   public abstract NDArray addi(NDArray rhs);

   /**
    * Performs a column vector addition adding the values in the other NDArray to each column in this NDArray.
    *
    * @param rhs the other NDArray whose values will be added
    * @return this NDArray with the result of this + other
    */
   public abstract NDArray addiColumnVector(NDArray rhs);

   /**
    * Performs a row vector addition adding the values in the other NDArray to each row in this NDArray.
    *
    * @param rhs the other NDArray whose values will be added
    * @return this NDArray with the result of this + other
    */
   public abstract NDArray addiRowVector(NDArray rhs);

   /**
    * Calculates the index in the NDArray with maximum value.
    *
    * @return the index with maximum value
    */
   public abstract long argmax();

   /**
    * Calculates the index in the NDArray with minimum value.
    *
    * @return the index with minimum value
    */
   public abstract long argmin();

   /**
    * Calculates the index of maximum values per column in the NDArray.
    *
    * @return the NDArray of column indexes with maximum value.
    */
   public abstract NDArray columnArgmaxs();

   /**
    * Calculates the index of minimum values per column in the NDArray.
    *
    * @return the NDArray of column indexes with minimum value.
    */
   public abstract NDArray columnArgmins();

   /**
    * Calculates the maximum values per column in the NDArray.
    *
    * @return the NDArray of maximum values per column.
    */
   public abstract NDArray columnMaxs();

   /**
    * Calculates the mean values per column in the NDArray.
    *
    * @return the NDArray of mean values per column.
    */
   public NDArray columnMeans() {
      return columnSums().divi(shape().rows());
   }

   /**
    * Calculates the minimum values per column in the NDArray.
    *
    * @return the NDArray of minimum values per column.
    */
   public abstract NDArray columnMins();

   /**
    * Calculates sums per column in the NDArray.
    *
    * @return the NDArray of sums per column.
    */
   public abstract NDArray columnSums();

   /**
    * Generates a diagonal matrix per slice.
    *
    * @return The NDArray with diagonal slices.
    */
   public abstract NDArray diag();

   /**
    * Divides the values in the other NDArray to this one element by element.
    *
    * @param rhs the other NDArray whose values will be divided
    * @return the new NDArray with the result of this / other
    */
   public abstract NDArray div(NDArray rhs);

   /**
    * Divides a scalar value to each element in the NDArray
    *
    * @param value the value to divide
    * @return the new NDArray with the scalar value divided
    */
   public abstract NDArray div(double value);

   /**
    * Divides a column vector element division dividing the values in the other NDArray to each column in this NDArray.
    *
    * @param rhs the other NDArray whose values will be divided
    * @return the new NDArray with the result of this / other
    */
   public abstract NDArray divColumnVector(NDArray rhs);

   /**
    * Divides a row vector element division dividing the values in the other NDArray to each row in this NDArray.
    *
    * @param rhs the other NDArray whose values will be divided
    * @return the new NDArray with the result of this / other
    */
   public abstract NDArray divRowVector(NDArray rhs);

   /**
    * Divides a scalar value to each element in the NDArray in-place.
    *
    * @param rhs the value to divide
    * @return this NDArray with the scalar value divided
    */
   public abstract NDArray divi(NDArray rhs);

   /**
    * Divides a scalar value to each element in the NDArray in-place.
    *
    * @param value the value to divide
    * @return this NDArray with the scalar value divided
    */
   public abstract NDArray divi(double value);

   /**
    * Divides a column vector element division dividing the values in the other NDArray to each column in this NDArray.
    *
    * @param rhs the other NDArray whose values will be divided
    * @return this NDArray with the result of this / other
    */
   public abstract NDArray diviColumnVector(NDArray rhs);

   /**
    * Divides a row vector element division dividing the values in the other NDArray to each row in this NDArray.
    *
    * @param rhs the other NDArray whose values will be divided
    * @return this NDArray with the result of this / other
    */
   public abstract NDArray diviRowVector(NDArray rhs);

   /**
    * Calculates the dot product between this and the given other NDArray per slice.
    *
    * @param rhs the NDArray to calculate the dot product with
    * @return NDArray of dot products
    */
   public abstract double dot(NDArray rhs);

   /**
    * Creates a new NDArray with elements equal to <code>1.0</code> if its value is equal to the given value.
    *
    * @param value the value test equality for
    * @return the NDArray
    */
   public NDArray eq(double value) {
      return testi(v -> v == value);
   }

   /**
    * Creates a new NDArray with elements equal to <code>1.0</code> if its value is equal to the value in the other
    * NDArray.
    *
    * @param rhs the NDArray whose values to test equality for
    * @return the NDArray
    */
   public NDArray eq(NDArray rhs) {
      return testi(rhs, (v, value) -> v == value);
   }

   /**
    * Updates this NDArray element's to equal <code>1.0</code> if its value is equal to the given value.
    *
    * @param value the value test equality for
    * @return this NDArray
    */
   public NDArray eqi(double value) {
      return testi(v -> v == value);
   }

   /**
    * Updates this NDArray element's to qual  <code>1.0</code> if its value is equal to the value in the other NDArray.
    *
    * @param rhs the NDArray whose values to test equality for
    * @return this NDArray
    */
   public NDArray eqi(NDArray rhs) {
      return testi(rhs, (v, value) -> v == value);
   }

   /**
    * Fills the NDArray with the given value
    *
    * @param value the value to set all cells in the NDArray
    * @return This NDArray
    */
   public abstract NDArray fill(double value);

   /**
    * Creates a new NDArray with elements equal to <code>1.0</code> if its value is greater than or equal to the given
    * value.
    *
    * @param value the value to test
    * @return the NDArray
    */
   public NDArray ge(double value) {
      return test(v -> v >= value);
   }

   /**
    * Creates a new NDArray with elements equal to <code>1.0</code> if its value is greater than or equal to the value
    * in the other NDArray.
    *
    * @param rhs the NDArray whose values to test
    * @return the NDArray
    */
   public NDArray ge(NDArray rhs) {
      return test(rhs, (v, value) -> v >= value);
   }

   /**
    * Updates this NDArray element's to equal <code>1.0</code> if its value is greater than or equal to the given
    * value.
    *
    * @param value the value to test
    * @return this NDArray
    */
   public NDArray gei(double value) {
      return testi(v -> v >= value);
   }

   /**
    * Updates this NDArray element's to equal  <code>1.0</code> if its value is greater than or equal to the value in
    * the other NDArray.
    *
    * @param rhs the NDArray whose values to test
    * @return this NDArray
    */
   public NDArray gei(NDArray rhs) {
      return testi(rhs, (v, value) -> v >= value);
   }

   /**
    * Gets the value at the given index (row/column if vector, entry if other)
    *
    * @param i the index
    * @return the double value
    */
   public abstract double get(long i);

   /**
    * Gets the value of the NDArray at the given row and column. (Assumes channel and kernel are 0)
    *
    * @param row the row index
    * @param col the column index
    * @return the double value
    */
   public abstract double get(int row, int col);

   /**
    * Gets the value of the NDArray at the given channel, row, and column (Assumes kernel is 0)
    *
    * @param channel the channel index
    * @param row     the row index
    * @param col     the column index
    * @return the double value
    */
   public abstract double get(int channel, int row, int col);

   /**
    * Gets the value of the NDArray at the given kernel, channel, row, and column
    *
    * @param kernel  the kernel index
    * @param channel the channel index
    * @param row     the row index
    * @param col     the column index
    * @return the double value
    */
   public abstract double get(int kernel, int channel, int row, int col);

   /**
    * Creates an NDArray made up of the column at the given index for each slice. (Note modifications to the new NDArray
    * do  not effect this one).
    *
    * @param column the column index
    * @return the column NDArray
    */
   public abstract NDArray getColumn(int column);

   /**
    * Creates an NDArray made up of the row at the given index for each slice. (Note modifications to the new NDArray do
    * not effect this one).
    *
    * @param row the row index
    * @return the row NDArray
    */
   public abstract NDArray getRow(int row);

   /**
    * Creates a new NDArray with elements equal to <code>1.0</code> if its value is greater than  to the given value.
    *
    * @param value the value to test
    * @return the NDArray
    */
   public NDArray gt(double value) {
      return test(v -> v > value);
   }

   /**
    * Creates a new NDArray with elements equal to <code>1.0</code> if its value is greater than to the value in the
    * other NDArray.
    *
    * @param rhs the NDArray whose values to test
    * @return the NDArray
    */
   public NDArray gt(NDArray rhs) {
      return test(rhs, (v, value) -> v > value);
   }

   /**
    * Updates this NDArray element's to equal <code>1.0</code> if its value is greater than to the given value.
    *
    * @param value the value to test
    * @return this NDArray
    */
   public NDArray gti(double value) {
      return testi(v -> v > value);
   }

   /**
    * Updates this NDArray element's to equal  <code>1.0</code> if its value is greater than to the value in the other
    * NDArray.
    *
    * @param rhs the NDArray whose values to test
    * @return this NDArray
    */
   public NDArray gti(NDArray rhs) {
      return testi(rhs, (v, value) -> v > value);
   }

   /**
    * Checks if the NDArray is made up of dense slices
    *
    * @return True if the NDArray is made up of dense slices, False otherwise
    */
   public abstract boolean isDense();

   /**
    * Creates a new NDArray with elements equal to <code>1.0</code> if its value is less than or equal to the given
    * value.
    *
    * @param value the value to test
    * @return the NDArray
    */
   public NDArray le(double value) {
      return test(v -> v <= value);
   }

   /**
    * Creates a new NDArray with elements equal to <code>1.0</code> if its value is less than or equal to the value in
    * the other NDArray.
    *
    * @param rhs the NDArray whose values to test
    * @return the NDArray
    */
   public NDArray le(NDArray rhs) {
      return test(rhs, (v, value) -> v <= value);
   }

   /**
    * Updates this NDArray element's to equal <code>1.0</code> if its value is less than or equal to the given value.
    *
    * @param value the value to test
    * @return this NDArray
    */
   public NDArray lei(double value) {
      return testi(v -> v <= value);
   }

   /**
    * Updates this NDArray element's to equal  <code>1.0</code> if its value is less than or equal to the value in the
    * other NDArray.
    *
    * @param rhs the NDArray whose values to test
    * @return this NDArray
    */
   public NDArray lei(NDArray rhs) {
      return testi(rhs, (v, value) -> v <= value);
   }

   /**
    * The total number of elements. (<code>kernels * channels * rows * columns</code>)
    *
    * @return the length (total number) of the elements in the NDArray
    */
   public long length() {
      return shape.sliceLength * shape.matrixLength;
   }

   /**
    * Number of rows in the NDArray
    *
    * @return the number of rows in the NDArray
    */
   public int rows() {
      return shape.rows();
   }

   /**
    * Number of columns in the NDArray
    *
    * @return the number of columns in the NDArray
    */
   public int columns() {
      return shape.columns();
   }

   /**
    * Number of kernels in the NDArray
    *
    * @return the number of kernels in the NDArray
    */
   public int kernels() {
      return shape.kernels();
   }

   /**
    * Number of channels in the NDArray
    *
    * @return the number of channels in the NDArray
    */
   public int channels() {
      return shape.channels();
   }

   /**
    * Creates a new NDArray with elements equal to <code>1.0</code> if its value is less than to the given value.
    *
    * @param value the value to test
    * @return the NDArray
    */
   public NDArray lt(double value) {
      return test(v -> v < value);
   }

   /**
    * Creates a new NDArray with elements equal to <code>1.0</code> if its value is less than to the value in the other
    * NDArray.
    *
    * @param rhs the NDArray whose values to test
    * @return the NDArray
    */
   public NDArray lt(NDArray rhs) {
      return test(rhs, (v, value) -> v < value);
   }

   /**
    * Updates this NDArray element's to equal <code>1.0</code> if its value is less than the given value.
    *
    * @param value the value to test
    * @return this NDArray
    */
   public NDArray lti(double value) {
      return testi(v -> v < value);
   }

   /**
    * Updates this NDArray element's to equal  <code>1.0</code> if its value is less than to the value in the other
    * NDArray.
    *
    * @param rhs the NDArray whose values to test
    * @return this NDArray
    */
   public NDArray lti(NDArray rhs) {
      return testi(rhs, (v, value) -> v < value);
   }

   /**
    * Creates a new NDArray with values from this NDArray evaluated using the given unary operator.
    *
    * @param operator the operation to perform on the values of this NDArray
    * @return the transformed NDArray
    */
   public abstract NDArray map(DoubleUnaryOperator operator);


   /**
    * Creates a new NDArray with values from this NDArray evaluated by the given binary operation with the given value.
    *
    * @param operator the operation to perform on the values of this NDArray and the given value
    * @return the transformed NDArray
    */
   public abstract NDArray map(double value, DoubleBinaryOperator operator);

   /**
    * Creates a new NDArray with values from this NDArray and the given NDArray evaluated using the given  binary
    * operation.
    *
    * @param operator the operation to perform on the values of this NDArray and the given NDArray
    * @return the transformed NDArray
    */
   public abstract NDArray map(NDArray rhs, DoubleBinaryOperator operator);

   /**
    * Creates a new NDArray with values from this NDArray and the given NDArray evaluated using the given  binary
    * operation per column.
    *
    * @param operator the operation to perform on the values of this NDArray and the given NDArray
    * @return the transformed NDArray
    */
   public abstract NDArray mapColumn(NDArray rhs, final DoubleBinaryOperator operator);

   /**
    * Creates a new NDArray with values from this NDArray and the given NDArray evaluated using the given  binary
    * operation per row.
    *
    * @param operator the operation to perform on the values of this NDArray and the given NDArray
    * @return the transformed NDArray
    */
   public abstract NDArray mapRow(NDArray rhs, final DoubleBinaryOperator operator);

   /**
    * Updates the values in this NDArray evaluated using the given unary operator.
    *
    * @param operator the operation to perform on the values of this NDArray
    * @return the transformed NDArray
    */
   public abstract NDArray mapi(DoubleUnaryOperator operator);

   /**
    * Updates the values in this NDArray by performing he given binary operation with the given value.
    *
    * @param operator the operation to perform on the values of this NDArray and the given value
    * @return the transformed NDArray
    */
   public abstract NDArray mapi(double value, DoubleBinaryOperator operator);

   /**
    * Updates the values int this NDArray by performing the given binary operation with the values in the given
    * NDArray.
    *
    * @param operator the operation to perform on the values of this NDArray and the given NDArray
    * @return the transformed NDArray
    */
   public abstract NDArray mapi(NDArray rhs, DoubleBinaryOperator operator);

   /**
    * Updates the values int this NDArray by performing the given binary operation with the values in the given NDArray
    * per column.
    *
    * @param operator the operation to perform on the values of this NDArray and the given NDArray
    * @return the transformed NDArray
    */
   public abstract NDArray mapiColumn(NDArray rhs, final DoubleBinaryOperator operator);

   /**
    * Updates the values int this NDArray by performing the given binary operation with the values in the given NDArray
    * per row.
    *
    * @param operator the operation to perform on the values of this NDArray and the given NDArray
    * @return the transformed NDArray
    */
   public abstract NDArray mapiRow(NDArray rhs, final DoubleBinaryOperator operator);

   /**
    * Calculates the maximum value in the NDArray.
    *
    * @return the maximum value
    */
   public abstract double max();

   /**
    * Calculates the mean value in the NDArray
    *
    * @return the mean value
    */
   public double mean() {
      return sum() / (shape().matrixLength * shape().sliceLength);
   }

   /**
    * Calculates the minimum value in the NDArray
    *
    * @return the minimum value
    */
   public abstract double min();

   /**
    * Creates a new NDArray by multiplying the (matrix) slices of this NDArray with those in the given NDArray.
    *
    * @param rhs the NDArray to multiply
    * @return the resulting NDArray
    */
   public abstract NDArray mmul(NDArray rhs);

   /**
    * Multiplies the values in the other NDArray to this one element by element.
    *
    * @param rhs the other NDArray whose values will be multiplied
    * @return the new NDArray with the result of this * other
    */
   public abstract NDArray mul(NDArray rhs);

   /**
    * Multiplies a scalar value to each element in the NDArray
    *
    * @param value the value to multiplied
    * @return the new NDArray with the scalar value multiplied
    */
   public abstract NDArray mul(double value);

   /**
    * Performs a column vector element multiplication multiplying the values in the other NDArray to each  in this
    * NDArray.
    *
    * @param rhs the other NDArray whose values will be multiplied
    * @return the new NDArray with the result of this * other
    */
   public abstract NDArray mulColumnVector(NDArray rhs);

   /**
    * Performs a row vector element multiplication multiplying the values in the other NDArray to each row  in this
    * NDArray.
    *
    * @param rhs the other NDArray whose values will be multiplied
    * @return the new NDArray with the result of this * other
    */
   public abstract NDArray mulRowVector(NDArray rhs);

   /**
    * Multiplies the values in the other NDArray to this one element by element in-place.
    *
    * @param rhs the other NDArray whose values will be multiplied
    * @return this NDArray with the result of this * other
    */
   public abstract NDArray muli(NDArray rhs);

   /**
    * Multiplies a scalar value to each element in the NDArray in-place.
    *
    * @param value the value to multiplied
    * @return this NDArray with the scalar value multiplied
    */
   public abstract NDArray muli(double value);

   /**
    * Performs a column vector element multiplication multiplying the values in the other NDArray to each  in this
    * NDArray in-place.
    *
    * @param rhs the other NDArray whose values will be multiplied
    * @return the new NDArray with the result of this * other
    */
   public abstract NDArray muliColumnVector(NDArray rhs);

   /**
    * Performs a row vector element multiplication multiplying the values in the other NDArray to each row  in this
    * NDArray in-place.
    *
    * @param rhs the other NDArray whose values will be multiplied
    * @return the new NDArray with the result of this * other
    */
   public abstract NDArray muliRowVector(NDArray rhs);

   /**
    * Creates a new NDArray with elements equal to <code>1.0</code> if its value is not equal to the given value.
    *
    * @param value the value test equality for
    * @return the NDArray
    */
   public NDArray neq(double value) {
      return testi(v -> v != value);
   }

   /**
    * Creates a new NDArray with elements equal to <code>1.0</code> if its value is not equal to the value in the other
    * NDArray.
    *
    * @param rhs the NDArray whose values to test equality for
    * @return the NDArray
    */
   public NDArray neq(NDArray rhs) {
      return testi(rhs, (v, value) -> v != value);
   }

   /**
    * Updates this NDArray element's to equal <code>1.0</code> if its value is not equal to the given value.
    *
    * @param value the value test equality for
    * @return this NDArray
    */
   public NDArray neqi(double value) {
      return testi(v -> v != value);
   }

   /**
    * Updates this NDArray element's to equal  <code>1.0</code> if its value is not equal to the value in the other
    * NDArray.
    *
    * @param rhs the NDArray whose values to test equality for
    * @return this NDArray
    */
   public NDArray neqi(NDArray rhs) {
      return testi(rhs, (v, value) -> v != value);
   }

   /**
    * Calculates the L1 norm of the NDArray
    *
    * @return the L1 norm of the NDArray
    */
   public abstract double norm1();


   /**
    * Calculates the L2 norm of the NDArray
    *
    * @return the L2 norm of the NDArray
    */
   public abstract double norm2();

   /**
    * Calculates the pivot elements for this square matrix. Will calculate per slice.
    *
    * @return A NDArray of 1's and 0's representing pivot elements.
    */
   public abstract NDArray pivot();

   /**
    * Divides the values in the this NDArray from the other NDArray.
    *
    * @param rhs the other NDArray whose values will be divided from
    * @return the new NDArray with the result of other / this
    */
   public abstract NDArray rdiv(NDArray rhs);

   /**
    * Divides each element's value from the given scalar (e.g. scalar - element)
    *
    * @param value the value to divide
    * @return the new NDArray with the scalar value divided
    */
   public abstract NDArray rdiv(double value);

   /**
    * Performs a column vector division dividing the values in this NDArray from the other NDArray to each column in
    * this NDArray.
    *
    * @param rhs the other NDArray whose values will be divided
    * @return the new NDArray with the result of this / other
    */
   public abstract NDArray rdivColumnVector(NDArray rhs);

   /**
    * Performs a row vector division dividing the values in this NDArray from the other NDArray to each row in this
    * NDArray.
    *
    * @param rhs the other NDArray whose values will be divided
    * @return the new NDArray with the result of this / other
    */
   public abstract NDArray rdivRowVector(NDArray rhs);

   /**
    * Divides the values in the this NDArray from the other NDArray in-place.
    *
    * @param rhs the other NDArray whose values will be divided from
    * @return the new NDArray with the result of other / this
    */
   public abstract NDArray rdivi(NDArray rhs);

   /**
    * Divides each element's value from the given scalar (e.g. scalar - element) in-place
    *
    * @param value the value to divide
    * @return the new NDArray with the scalar value divided
    */
   public abstract NDArray rdivi(double value);

   /**
    * Performs a column vector division dividing the values in this NDArray from the other NDArray to each column in
    * this NDArray in-place.
    *
    * @param rhs the other NDArray whose values will be divided
    * @return the new NDArray with the result of this / other
    */
   public abstract NDArray rdiviColumnVector(NDArray rhs);

   /**
    * Performs a row vector division dividing the values in this NDArray from the other NDArray to each row in this
    * NDArray in-place.
    *
    * @param rhs the other NDArray whose values will be divided
    * @return the new NDArray with the result of this / other
    */
   public abstract NDArray rdiviRowVector(NDArray rhs);

   /**
    * Reshapes the NDArray
    *
    * @param dims the new dimensions of the NDArray
    * @return this NDArray with new shape
    */
   public abstract NDArray reshape(int... dims);


   /**
    * Calculates the index of maximum values per row in the NDArray.
    *
    * @return the NDArray of row indexes with maximum value.
    */
   public abstract NDArray rowArgmaxs();

   /**
    * Calculates the index of minimum values per row in the NDArray.
    *
    * @return the NDArray of row indexes with minimum value.
    */
   public abstract NDArray rowArgmins();

   /**
    * Calculates the maximum values per row in the NDArray.
    *
    * @return the NDArray of maximum values per row.
    */
   public abstract NDArray rowMaxs();

   /**
    * Calculates the mean values per row in the NDArray.
    *
    * @return the NDArray of mean values per row.
    */
   public NDArray rowMeans() {
      return rowSums().divi(shape().columns());
   }

   /**
    * Calculates the minimum values per row in the NDArray.
    *
    * @return the NDArray of minimum values per row.
    */
   public abstract NDArray rowMins();

   /**
    * Calculates the sum per row in the NDArray.
    *
    * @return the NDArray of sum per row.
    */
   public abstract NDArray rowSums();

   /**
    * Subtracts the values in the this NDArray from the other NDArray.
    *
    * @param rhs the other NDArray whose values will be subtracted from
    * @return the new NDArray with the result of other - this
    */
   public abstract NDArray rsub(NDArray rhs);

   /**
    * Subtracts each element's value from the given scalar (e.g. scalar - element)
    *
    * @param value the value to subtract
    * @return the new NDArray with the scalar value subtracted
    */
   public abstract NDArray rsub(double value);

   /**
    * Performs a column vector subtraction subtracting the values in this NDArray from the other NDArray to each column
    * in this NDArray.
    *
    * @param rhs the other NDArray whose values will be subtracted
    * @return the new NDArray with the result of this - other
    */
   public abstract NDArray rsubColumnVector(NDArray rhs);

   /**
    * Performs a row vector subtraction subtracting the values in this NDArray from the other NDArray to each row in
    * this NDArray.
    *
    * @param rhs the other NDArray whose values will be subtracted
    * @return the new NDArray with the result of this - other
    */
   public abstract NDArray rsubRowVector(NDArray rhs);

   /**
    * Subtracts the values in the this NDArray from the other NDArray in-place.
    *
    * @param rhs the other NDArray whose values will be subtracted from
    * @return the new NDArray with the result of other - this
    */
   public abstract NDArray rsubi(NDArray rhs);

   /**
    * Subtracts each element's value from the given scalar (e.g. scalar - element) in-place
    *
    * @param value the value to subtract
    * @return the new NDArray with the scalar value subtracted
    */
   public abstract NDArray rsubi(double value);

   /**
    * Performs a column vector subtraction subtracting the values in this NDArray from the other NDArray to each column
    * in this NDArray in-place.
    *
    * @param rhs the other NDArray whose values will be subtracted
    * @return the new NDArray with the result of this - other
    */
   public abstract NDArray rsubiColumnVector(NDArray rhs);

   /**
    * Performs a row vector subtraction subtracting the values in this NDArray from the other NDArray to each row in
    * this NDArray in-place.
    *
    * @param rhs the other NDArray whose values will be subtracted
    * @return the new NDArray with the result of this - other
    */
   public abstract NDArray rsubiRowVector(NDArray rhs);

   /**
    * Returns the scalar value of this NDArray (value at <code>(0,0,0,0)</code>)
    *
    * @return the scalar value
    */
   public double scalar() {
      return get(0);
   }


   /**
    * Selects all values matching the given predicate.
    *
    * @param predicate the predicate to test
    * @return new NDArray with values passing the given predicate and zeros elsewhere
    */
   public abstract NDArray select(DoublePredicate predicate);

   /**
    * Selects all values matching the given predicate in-place.
    *
    * @param predicate the predicate to test
    * @return this NDArray with values passing the given predicate and zeros elsewhere
    */
   public abstract NDArray selecti(DoublePredicate predicate);

   /**
    * Selects all values for which the corresponding element in the given NDArray has a value of <code>1.0</code>.
    *
    * @param rhs the NDArray used to determine which values are selected
    * @return the selected NDArray
    */
   public abstract NDArray select(NDArray rhs);

   /**
    * Selects all values for which the corresponding element in the given NDArray has a value of <code>1.0</code>
    * in-place.
    *
    * @param rhs the NDArray used to determine which values are selected
    * @return the selected NDArray
    */
   public abstract NDArray selecti(NDArray rhs);

   /**
    * Sets the value of the element at the given index. (row/column if vector, entry if other)
    *
    * @param i     the index
    * @param value the value
    * @return this NDArray
    */
   public abstract NDArray set(long i, double value);

   /**
    * Sets the value of the element at the given row and column (assumes kernel and channel are 0).
    *
    * @param row   the row index
    * @param col   the column index
    * @param value the value
    * @return this NDArray
    */
   public abstract NDArray set(int row, int col, double value);

   /**
    * Sets the value of the element at the given channel, row, and column (assumes kernel is 0).
    *
    * @param channel the channel index
    * @param row     the row index
    * @param col     the column index
    * @param value   the value
    * @return this NDArray
    */
   public abstract NDArray set(int channel, int row, int col, double value);

   /**
    * Sets the value of the element at the given kernel, channel, row, and column
    *
    * @param kernel  the kernel index
    * @param channel the channel index
    * @param row     the row index
    * @param col     the column index
    * @param value   the value
    * @return this NDArray
    */
   public abstract NDArray set(int kernel, int channel, int row, int col, double value);

   /**
    * Sets the values of the <code>ith</code> column to those in the given NDArray.
    *
    * @param i     the column index
    * @param array the array of new column values
    * @return this NDArray
    */
   public abstract NDArray setColumn(int i, NDArray array);

   /**
    * Sets the values of the <code>ith</code> row to those in the given NDArray.
    *
    * @param i     the row index
    * @param array the array of new row values
    * @return this NDArray
    */
   public abstract NDArray setRow(int i, NDArray array);

   /**
    * Gets the shape of the NDArray.
    *
    * @return the shape
    */
   public abstract Shape shape();

   /**
    * Then number of sparse entries (dense NDArray will have <code>size()=length()</code>)
    *
    * @return the number of sparse entries.
    */
   public abstract long size();

   /**
    * Sets the slice at the given index.
    *
    * @param slice the slice index
    * @param array the NDArray of values for the new slice
    * @return this NDArray
    */
   public abstract NDArray setSlice(int slice, NDArray array);

   /**
    * Sets the matrix associated with the given kernel and channel.
    *
    * @param kernel  the kernel index
    * @param channel the channel index
    * @param array   the matrix
    * @return this NDArray
    */
   public NDArray setMatrix(int kernel, int channel, NDArray array) {
      return setSlice(shape.sliceIndex(kernel, channel), array);
   }


   /**
    * Returns a  view of a single slice of this NDArray. Note that changes to the slice will effect this NDArray.
    *
    * @param slice the slice index
    * @return the NDArray for the slice
    */
   public abstract NDArray slice(int slice);

   /**
    * Calculates the index of the per-slice maximum values.
    *
    * @return the per-slice argmax
    */
   public abstract NDArray sliceArgmaxs();

   /**
    * Calculates the index of the per-slice minimum values.
    *
    * @return the per-slice argmin
    */
   public abstract NDArray sliceArgmins();

   /**
    * Calculates the dot product between each slice of this and the given NDArray
    *
    * @param rhs the NDArray to calculate the dot product with
    * @return the per-slice dot product
    */
   public abstract NDArray sliceDot(NDArray rhs);

   /**
    * Calculates the  per-slice maximum values.
    *
    * @return the per-slice maximum values
    */
   public abstract NDArray sliceMaxs();

   /**
    * Calculates the  per-slice mean values.
    *
    * @return the per-slice mean values
    */
   public abstract NDArray sliceMeans();

   /**
    * Calculates the  per-slice minimum values.
    *
    * @return the per-slice minimum values
    */
   public abstract NDArray sliceMins();

   /**
    * Calculates the  per-slice L1 norm values.
    *
    * @return the per-slice L1 norm  values
    */
   public abstract NDArray sliceNorm1();

   /**
    * Calculates the  per-slice L2 norm values.
    *
    * @return the per-slice L2 norm  values
    */
   public abstract NDArray sliceNorm2();

   /**
    * Calculates the  per-slice sum of square values.
    *
    * @return the per-slice sum of square  values
    */
   public abstract NDArray sliceSumOfSquares();

   /**
    * Calculates the  per-slice sums.
    *
    * @return the per-slice sums
    */
   public abstract NDArray sliceSums();

   /**
    * Sub nd array.
    *
    * @param rhs the rhs
    * @return the nd array
    */
   public abstract NDArray sub(NDArray rhs);

   /**
    * Sub nd array.
    *
    * @param value the value
    * @return the nd array
    */
   public abstract NDArray sub(double value);

   /**
    * Sub column vector nd array.
    *
    * @param rhs the rhs
    * @return the nd array
    */
   public abstract NDArray subColumnVector(NDArray rhs);

   /**
    * Sub row vector nd array.
    *
    * @param rhs the rhs
    * @return the nd array
    */
   public abstract NDArray subRowVector(NDArray rhs);

   /**
    * Subi nd array.
    *
    * @param rhs the rhs
    * @return the nd array
    */
   public abstract NDArray subi(NDArray rhs);

   /**
    * Subi nd array.
    *
    * @param value the value
    * @return the nd array
    */
   public abstract NDArray subi(double value);

   /**
    * Subi column vector nd array.
    *
    * @param rhs the rhs
    * @return the nd array
    */
   public abstract NDArray subiColumnVector(NDArray rhs);

   /**
    * Subi row vector nd array.
    *
    * @param rhs the rhs
    * @return the nd array
    */
   public abstract NDArray subiRowVector(NDArray rhs);

   /**
    * Sum double.
    *
    * @return the double
    */
   public abstract double sum();

   /**
    * Sum of squares double.
    *
    * @return the double
    */
   public abstract double sumOfSquares();

   /**
    * Test nd array.
    *
    * @param predicate the predicate
    * @return the nd array
    */
   public abstract NDArray test(DoublePredicate predicate);

   /**
    * Test nd array.
    *
    * @param rhs       the rhs
    * @param predicate the predicate
    * @return the nd array
    */
   public abstract NDArray test(NDArray rhs, DoubleBinaryPredicate predicate);

   /**
    * Testi nd array.
    *
    * @param predicate the predicate
    * @return the nd array
    */
   public abstract NDArray testi(DoublePredicate predicate);

   /**
    * Testi nd array.
    *
    * @param rhs       the rhs
    * @param predicate the predicate
    * @return the nd array
    */
   public abstract NDArray testi(NDArray rhs, DoubleBinaryPredicate predicate);

   /**
    * To double matrix double matrix [ ].
    *
    * @return the double matrix [ ]
    */
   public abstract DoubleMatrix[] toDoubleMatrix();

   /**
    * Zero nd array.
    *
    * @return the nd array
    */
   public NDArray zero() {
      return fill(0d);
   }

   /**
    * Zero like nd array.
    *
    * @return the nd array
    */
   public abstract NDArray zeroLike();

   /**
    * Gets channels.
    *
    * @param from the from
    * @param to   the to
    * @return the channels
    */
   public abstract NDArray getChannels(int from, int to);

   /**
    * Gets channels.
    *
    * @param channels the channels
    * @return the channels
    */
   public abstract NDArray getChannels(int[] channels);

   /**
    * Gets kernels.
    *
    * @param from the from
    * @param to   the to
    * @return the kernels
    */
   public abstract NDArray getKernels(int from, int to);

   /**
    * Gets kernels.
    *
    * @param kernels the kernels
    * @return the kernels
    */
   public abstract NDArray getKernels(int[] kernels);

   /**
    * Gets rows.
    *
    * @param rows the rows
    * @return the rows
    */
   public abstract NDArray getRows(int[] rows);

   /**
    * Gets columns.
    *
    * @param columns the columns
    * @return the columns
    */
   public abstract NDArray getColumns(int[] columns);

   /**
    * Gets rows.
    *
    * @param from the from
    * @param to   the to
    * @return the rows
    */
   public abstract NDArray getRows(int from, int to);

   /**
    * Gets columns.
    *
    * @param from the from
    * @param to   the to
    * @return the columns
    */
   public abstract NDArray getColumns(int from, int to);


   /**
    * The interface Double binary predicate.
    */
   @FunctionalInterface
   interface DoubleBinaryPredicate {

      /**
       * Test boolean.
       *
       * @param v1 the v 1
       * @param v2 the v 2
       * @return the boolean
       */
      boolean test(double v1, double v2);
   }

   /**
    * The interface Entry consumer.
    */
   @FunctionalInterface
   public interface EntryConsumer {

      /**
       * Apply.
       *
       * @param index the index
       * @param value the value
       */
      void apply(long index, double value);

   }


   /**
    * To double array double [ ].
    *
    * @return the double [ ]
    */
   public abstract double[] toDoubleArray();

   /**
    * For each sparse.
    *
    * @param consumer the consumer
    */
   public abstract void forEachSparse(EntryConsumer consumer);
}//END OF NDArray
