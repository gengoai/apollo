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

package com.gengoai.apollo.linear.p2;

import com.gengoai.Copyable;
import com.gengoai.apollo.linear.Shape;
import org.jblas.DoubleMatrix;

import java.io.Serializable;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoublePredicate;
import java.util.function.DoubleUnaryOperator;

/**
 * @author David B. Bracewell
 */
public interface NDArray extends Serializable, Copyable<NDArray> {

   NDArray T();

   NDArray add(double value);

   NDArray add(NDArray rhs);

   NDArray addColumnVector(NDArray rhs);

   NDArray addRowVector(NDArray rhs);

   NDArray addi(double value);

   NDArray addi(NDArray rhs);

   NDArray addiColumnVector(NDArray rhs);

   NDArray addiRowVector(NDArray rhs);

   double argmax();

   double argmin();

   NDArray columnArgmaxs();

   NDArray columnArgmins();

   NDArray columnMaxs();

   default NDArray columnMeans() {
      return columnSums().divi(shape().rows());
   }

   NDArray columnMins();

   NDArray columnSums();

   NDArray diag();

   NDArray div(NDArray rhs);

   NDArray div(double value);

   NDArray divColumnVector(NDArray rhs);

   NDArray divRowVector(NDArray rhs);

   NDArray divi(NDArray rhs);

   NDArray divi(double value);

   NDArray diviColumnVector(NDArray rhs);

   NDArray diviRowVector(NDArray rhs);

   double dot(NDArray rhs);

   default NDArray eq(double value) {
      return testi(v -> v == value);
   }

   default NDArray eq(NDArray rhs) {
      return testi(rhs, (v, value) -> v == value);
   }

   default NDArray eqi(double value) {
      return testi(v -> v == value);
   }

   default NDArray eqi(NDArray rhs) {
      return testi(rhs, (v, value) -> v == value);
   }

   NDArray fill(double value);

   default NDArray ge(double value) {
      return test(v -> v >= value);
   }

   default NDArray ge(NDArray rhs) {
      return test(rhs, (v, value) -> v >= value);
   }

   default NDArray gei(double value) {
      return testi(v -> v >= value);
   }

   default NDArray gei(NDArray rhs) {
      return testi(rhs, (v, value) -> v >= value);
   }

   double get(long i);

   double get(int row, int col);

   double get(int channel, int row, int col);

   double get(int kernel, int channel, int row, int col);

   NDArray getColumn(int column);

   NDArray getRow(int row);

   default NDArray gt(double value) {
      return test(v -> v > value);
   }

   default NDArray gt(NDArray rhs) {
      return test(rhs, (v, value) -> v > value);
   }

   default NDArray gti(double value) {
      return testi(v -> v > value);
   }

   default NDArray gti(NDArray rhs) {
      return testi(rhs, (v, value) -> v > value);
   }

   boolean isDense();

   default NDArray le(double value) {
      return test(v -> v <= value);
   }

   default NDArray le(NDArray rhs) {
      return test(rhs, (v, value) -> v <= value);
   }

   default NDArray lei(double value) {
      return testi(v -> v <= value);
   }

   default NDArray lei(NDArray rhs) {
      return testi(rhs, (v, value) -> v <= value);
   }

   long length();

   default NDArray lt(double value) {
      return test(v -> v < value);
   }

   default NDArray lt(NDArray rhs) {
      return test(rhs, (v, value) -> v < value);
   }

   default NDArray lti(double value) {
      return testi(v -> v < value);
   }

   default NDArray lti(NDArray rhs) {
      return testi(rhs, (v, value) -> v < value);
   }

   NDArray map(DoubleUnaryOperator operator);

   NDArray map(double value, DoubleBinaryOperator operator);

   NDArray map(NDArray rhs, DoubleBinaryOperator operator);

   NDArray mapColumn(NDArray rhs, final DoubleBinaryOperator operator);

   NDArray mapRow(NDArray rhs, final DoubleBinaryOperator operator);

   NDArray mapi(DoubleUnaryOperator operator);

   NDArray mapi(double value, DoubleBinaryOperator operator);

   NDArray mapi(NDArray rhs, DoubleBinaryOperator operator);

   NDArray mapiColumn(NDArray rhs, final DoubleBinaryOperator operator);

   NDArray mapiRow(NDArray rhs, final DoubleBinaryOperator operator);

   double max();

   default double mean() {
      return sum() / (shape().matrixLength * shape().sliceLength);
   }

   double min();

   NDArray mmul(NDArray rhs);

   NDArray mul(NDArray rhs);

   NDArray mul(double value);

   NDArray mulColumnVector(NDArray rhs);

   NDArray mulRowVector(NDArray rhs);

   NDArray muli(NDArray rhs);

   NDArray muli(double value);

   NDArray muliColumnVector(NDArray rhs);

   NDArray muliRowVector(NDArray rhs);

   default NDArray neq(double value) {
      return testi(v -> v != value);
   }

   default NDArray neq(NDArray rhs) {
      return testi(rhs, (v, value) -> v != value);
   }

   default NDArray neqi(double value) {
      return testi(v -> v != value);
   }

   default NDArray neqi(NDArray rhs) {
      return testi(rhs, (v, value) -> v != value);
   }

   double norm1();

   double norm2();

   NDArray pivot();

   NDArray rdiv(NDArray rhs);

   NDArray rdiv(double value);

   NDArray rdivColumnVector(NDArray rhs);

   NDArray rdivRowVector(NDArray rhs);

   NDArray rdivi(NDArray rhs);

   NDArray rdivi(double value);

   NDArray rdiviColumnVector(NDArray rhs);

   NDArray rdiviRowVector(NDArray rhs);

   NDArray reshape(int... dims);

   NDArray rowArgmaxs();

   NDArray rowArgmins();

   NDArray rowMaxs();

   default NDArray rowMeans() {
      return rowSums().divi(shape().columns());
   }

   NDArray rowMins();

   NDArray rowSums();

   NDArray rsub(NDArray rhs);

   NDArray rsub(double value);

   NDArray rsubColumnVector(NDArray rhs);

   NDArray rsubRowVector(NDArray rhs);

   NDArray rsubi(NDArray rhs);

   NDArray rsubi(double value);

   NDArray rsubiColumnVector(NDArray rhs);

   NDArray rsubiRowVector(NDArray rhs);

   default double scalar() {
      return get(0);
   }

   NDArray select(NDArray rhs);

   NDArray selecti(NDArray rhs);

   void set(long i, double value);

   void set(int row, int col, double value);

   void set(int channel, int row, int col, double value);

   void set(int kernel, int channel, int row, int col, double value);

   NDArray setColumn(int column, NDArray array);

   NDArray setRow(int row, NDArray array);

   Shape shape();

   long size();

   NDArray slice(int slice);

   NDArray sliceArgmaxs();

   NDArray sliceArgmins();

   NDArray sliceDot(NDArray rhs);

   NDArray sliceMaxs();

   NDArray sliceMeans();

   NDArray sliceMins();

   NDArray sliceNorm1();

   NDArray sliceNorm2();

   NDArray sliceSumOfSquares();

   NDArray sliceSums();

   NDArray sub(NDArray rhs);

   NDArray sub(double value);

   NDArray subColumnVector(NDArray rhs);

   NDArray subRowVector(NDArray rhs);

   NDArray subi(NDArray rhs);

   NDArray subi(double value);

   NDArray subiColumnVector(NDArray rhs);

   NDArray subiRowVector(NDArray rhs);

   double sum();

   double sumOfSquares();

   NDArray test(DoublePredicate predicate);

   NDArray test(NDArray rhs, DoubleBinaryPredicate predicate);

   NDArray testi(DoublePredicate predicate);

   NDArray testi(NDArray rhs, DoubleBinaryPredicate predicate);

   DoubleMatrix[] toDoubleMatrix();

   default NDArray zero() {
      return fill(0d);
   }

   NDArray zeroLike();

   NDArray getChannels(int from, int to);

   NDArray getChannels(int[] channels);

   NDArray getKernels(int from, int to);

   NDArray getKernels(int[] kernels);

   NDArray getRows(int[] rows);

   NDArray getColumns(int[] columns);

   NDArray getRows(int from, int to);

   NDArray getColumns(int from, int to);


   @FunctionalInterface
   interface DoubleBinaryPredicate {

      boolean test(double v1, double v2);
   }


}//END OF NDArray
