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
 * @author David B. Bracewell
 */
public abstract class NDArray implements Serializable, Copyable<NDArray> {
   private Object label = null;
   private Object predicted = null;
   private double weight = 1d;
   protected final Shape shape;

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

   public abstract NDArray trimToSize();

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

   public abstract NDArray T();

   public abstract NDArray add(double value);

   public abstract NDArray add(NDArray rhs);

   public abstract NDArray addColumnVector(NDArray rhs);

   public abstract NDArray addRowVector(NDArray rhs);

   public abstract NDArray addi(double value);

   public abstract NDArray addi(NDArray rhs);

   public abstract NDArray addiColumnVector(NDArray rhs);

   public abstract NDArray addiRowVector(NDArray rhs);

   public abstract double argmax();

   public abstract double argmin();

   public abstract NDArray columnArgmaxs();

   public abstract NDArray columnArgmins();

   public abstract NDArray columnMaxs();

   public NDArray columnMeans() {
      return columnSums().divi(shape().rows());
   }

   public abstract NDArray columnMins();

   public abstract NDArray columnSums();

   public abstract NDArray diag();

   public abstract NDArray div(NDArray rhs);

   public abstract NDArray div(double value);

   public abstract NDArray divColumnVector(NDArray rhs);

   public abstract NDArray divRowVector(NDArray rhs);

   public abstract NDArray divi(NDArray rhs);

   public abstract NDArray divi(double value);

   public abstract NDArray diviColumnVector(NDArray rhs);

   public abstract NDArray diviRowVector(NDArray rhs);

   public abstract double dot(NDArray rhs);

   public NDArray eq(double value) {
      return testi(v -> v == value);
   }

   public NDArray eq(NDArray rhs) {
      return testi(rhs, (v, value) -> v == value);
   }

   public NDArray eqi(double value) {
      return testi(v -> v == value);
   }

   public NDArray eqi(NDArray rhs) {
      return testi(rhs, (v, value) -> v == value);
   }

   public abstract NDArray fill(double value);

   public NDArray ge(double value) {
      return test(v -> v >= value);
   }

   public NDArray ge(NDArray rhs) {
      return test(rhs, (v, value) -> v >= value);
   }

   public NDArray gei(double value) {
      return testi(v -> v >= value);
   }

   public NDArray gei(NDArray rhs) {
      return testi(rhs, (v, value) -> v >= value);
   }

   public abstract double get(long i);

   public abstract double get(int row, int col);

   public abstract double get(int channel, int row, int col);

   public abstract double get(int kernel, int channel, int row, int col);

   public abstract NDArray getColumn(int column);

   public abstract NDArray getRow(int row);

   public NDArray gt(double value) {
      return test(v -> v > value);
   }

   public NDArray gt(NDArray rhs) {
      return test(rhs, (v, value) -> v > value);
   }

   public NDArray gti(double value) {
      return testi(v -> v > value);
   }

   public NDArray gti(NDArray rhs) {
      return testi(rhs, (v, value) -> v > value);
   }

   public abstract boolean isDense();

   public NDArray le(double value) {
      return test(v -> v <= value);
   }

   public NDArray le(NDArray rhs) {
      return test(rhs, (v, value) -> v <= value);
   }

   public NDArray lei(double value) {
      return testi(v -> v <= value);
   }

   public NDArray lei(NDArray rhs) {
      return testi(rhs, (v, value) -> v <= value);
   }

   public long length() {
      return shape.sliceLength * shape.matrixLength;
   }

   public int rows() {
      return shape.rows();
   }

   public int columns() {
      return shape.columns();
   }

   public int kernels() {
      return shape.kernels();
   }

   public int channels() {
      return shape.channels();
   }

   public NDArray lt(double value) {
      return test(v -> v < value);
   }

   public NDArray lt(NDArray rhs) {
      return test(rhs, (v, value) -> v < value);
   }

   public NDArray lti(double value) {
      return testi(v -> v < value);
   }

   public NDArray lti(NDArray rhs) {
      return testi(rhs, (v, value) -> v < value);
   }

   public abstract NDArray map(DoubleUnaryOperator operator);

   public abstract NDArray map(double value, DoubleBinaryOperator operator);

   public abstract NDArray map(NDArray rhs, DoubleBinaryOperator operator);

   public abstract NDArray mapColumn(NDArray rhs, final DoubleBinaryOperator operator);

   public abstract NDArray mapRow(NDArray rhs, final DoubleBinaryOperator operator);

   public abstract NDArray mapi(DoubleUnaryOperator operator);

   public abstract NDArray mapi(double value, DoubleBinaryOperator operator);

   public abstract NDArray mapi(NDArray rhs, DoubleBinaryOperator operator);

   public abstract NDArray mapiColumn(NDArray rhs, final DoubleBinaryOperator operator);

   public abstract NDArray mapiRow(NDArray rhs, final DoubleBinaryOperator operator);

   public abstract double max();

   public double mean() {
      return sum() / (shape().matrixLength * shape().sliceLength);
   }

   public abstract double min();

   public abstract NDArray mmul(NDArray rhs);

   public abstract NDArray mul(NDArray rhs);

   public abstract NDArray mul(double value);

   public abstract NDArray mulColumnVector(NDArray rhs);

   public abstract NDArray mulRowVector(NDArray rhs);

   public abstract NDArray muli(NDArray rhs);

   public abstract NDArray muli(double value);

   public abstract NDArray muliColumnVector(NDArray rhs);

   public abstract NDArray muliRowVector(NDArray rhs);

   public NDArray neq(double value) {
      return testi(v -> v != value);
   }

   public NDArray neq(NDArray rhs) {
      return testi(rhs, (v, value) -> v != value);
   }

   public NDArray neqi(double value) {
      return testi(v -> v != value);
   }

   public NDArray neqi(NDArray rhs) {
      return testi(rhs, (v, value) -> v != value);
   }

   public abstract double norm1();

   public abstract double norm2();

   public abstract NDArray pivot();

   public abstract NDArray rdiv(NDArray rhs);

   public abstract NDArray rdiv(double value);

   public abstract NDArray rdivColumnVector(NDArray rhs);

   public abstract NDArray rdivRowVector(NDArray rhs);

   public abstract NDArray rdivi(NDArray rhs);

   public abstract NDArray rdivi(double value);

   public abstract NDArray rdiviColumnVector(NDArray rhs);

   public abstract NDArray rdiviRowVector(NDArray rhs);

   public abstract NDArray reshape(int... dims);

   public abstract NDArray rowArgmaxs();

   public abstract NDArray rowArgmins();

   public abstract NDArray rowMaxs();

   public NDArray rowMeans() {
      return rowSums().divi(shape().columns());
   }

   public abstract NDArray rowMins();

   public abstract NDArray rowSums();

   public abstract NDArray rsub(NDArray rhs);

   public abstract NDArray rsub(double value);

   public abstract NDArray rsubColumnVector(NDArray rhs);

   public abstract NDArray rsubRowVector(NDArray rhs);

   public abstract NDArray rsubi(NDArray rhs);

   public abstract NDArray rsubi(double value);

   public abstract NDArray rsubiColumnVector(NDArray rhs);

   public abstract NDArray rsubiRowVector(NDArray rhs);

   public double scalar() {
      return get(0);
   }

   public abstract NDArray select(NDArray rhs);

   public abstract NDArray selecti(NDArray rhs);

   public abstract NDArray set(long i, double value);

   public abstract NDArray set(int row, int col, double value);

   public abstract NDArray set(int channel, int row, int col, double value);

   public abstract NDArray set(int kernel, int channel, int row, int col, double value);

   public abstract NDArray setColumn(int column, NDArray array);

   public abstract NDArray setRow(int row, NDArray array);

   public abstract Shape shape();

   public abstract long size();

   public abstract NDArray slice(int slice);

   public abstract NDArray sliceArgmaxs();

   public abstract NDArray sliceArgmins();

   public abstract NDArray sliceDot(NDArray rhs);

   public abstract NDArray sliceMaxs();

   public abstract NDArray sliceMeans();

   public abstract NDArray sliceMins();

   public abstract NDArray sliceNorm1();

   public abstract NDArray sliceNorm2();

   public abstract NDArray sliceSumOfSquares();

   public abstract NDArray sliceSums();

   public abstract NDArray sub(NDArray rhs);

   public abstract NDArray sub(double value);

   public abstract NDArray subColumnVector(NDArray rhs);

   public abstract NDArray subRowVector(NDArray rhs);

   public abstract NDArray subi(NDArray rhs);

   public abstract NDArray subi(double value);

   public abstract NDArray subiColumnVector(NDArray rhs);

   public abstract NDArray subiRowVector(NDArray rhs);

   public abstract double sum();

   public abstract double sumOfSquares();

   public abstract NDArray test(DoublePredicate predicate);

   public abstract NDArray test(NDArray rhs, DoubleBinaryPredicate predicate);

   public abstract NDArray testi(DoublePredicate predicate);

   public abstract NDArray testi(NDArray rhs, DoubleBinaryPredicate predicate);

   public abstract DoubleMatrix[] toDoubleMatrix();

   public NDArray zero() {
      return fill(0d);
   }

   public abstract NDArray zeroLike();

   public abstract NDArray getChannels(int from, int to);

   public abstract NDArray getChannels(int[] channels);

   public abstract NDArray getKernels(int from, int to);

   public abstract NDArray getKernels(int[] kernels);

   public abstract NDArray getRows(int[] rows);

   public abstract NDArray getColumns(int[] columns);

   public abstract NDArray getRows(int from, int to);

   public abstract NDArray getColumns(int from, int to);


   @FunctionalInterface
   interface DoubleBinaryPredicate {

      boolean test(double v1, double v2);
   }

   @FunctionalInterface
   public interface EntryConsumer {

      void apply(long index, double value);

   }


   public abstract double[] toDoubleArray();

   public abstract void forEachSparse(EntryConsumer consumer);
}//END OF NDArray
