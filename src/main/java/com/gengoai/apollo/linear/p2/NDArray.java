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

   NDArray div(NDArray rhs);

   NDArray divColumnVector(NDArray rhs);

   NDArray divRowVector(NDArray rhs);

   NDArray divi(NDArray rhs);

   NDArray diviColumnVector(NDArray rhs);

   NDArray diviRowVector(NDArray rhs);

   NDArray fill(double value);

   double get(long i);

   double get(int row, int col);

   double get(int channel, int row, int col);

   double get(int kernel, int channel, int row, int col);

   boolean isDense();

   NDArray map(DoubleUnaryOperator operator);

   NDArray mapi(DoubleUnaryOperator operator);

   NDArray map(double value, DoubleBinaryOperator operator);

   NDArray map(NDArray rhs, DoubleBinaryOperator operator);

   NDArray mapColumn(NDArray rhs, final DoubleBinaryOperator operator);

   NDArray mapRow(NDArray rhs, final DoubleBinaryOperator operator);

   NDArray mapi(double value, DoubleBinaryOperator operator);

   NDArray mapi(NDArray rhs, DoubleBinaryOperator operator);

   NDArray mapiColumn(NDArray rhs, final DoubleBinaryOperator operator);

   NDArray mapiRow(NDArray rhs, final DoubleBinaryOperator operator);

   NDArray mul(NDArray rhs);

   NDArray mulColumnVector(NDArray rhs);

   NDArray mulRowVector(NDArray rhs);

   NDArray muli(NDArray rhs);

   NDArray muliColumnVector(NDArray rhs);

   NDArray muliRowVector(NDArray rhs);

   NDArray rdiv(NDArray rhs);

   NDArray rdivColumnVector(NDArray rhs);

   NDArray rdivRowVector(NDArray rhs);

   NDArray rdivi(NDArray rhs);

   NDArray rdiviColumnVector(NDArray rhs);

   NDArray rdiviRowVector(NDArray rhs);

   NDArray rsub(NDArray rhs);

   NDArray rsubColumnVector(NDArray rhs);

   NDArray rsubRowVector(NDArray rhs);

   NDArray rsubi(NDArray rhs);

   NDArray rsubiColumnVector(NDArray rhs);

   NDArray rsubiRowVector(NDArray rhs);

   default double scalar() {
      return get(0);
   }

   void set(long i, double value);

   void set(int row, int col, double value);

   void set(int channel, int row, int col, double value);

   void set(int kernel, int channel, int row, int col, double value);

   Shape shape();

   NDArray slice(int slice);

   NDArray sub(NDArray rhs);

   NDArray subColumnVector(NDArray rhs);

   NDArray subRowVector(NDArray rhs);

   NDArray subi(NDArray rhs);

   NDArray subiColumnVector(NDArray rhs);

   NDArray subiRowVector(NDArray rhs);

   DoubleMatrix[] toDoubleMatrix();

   default NDArray zero() {
      return fill(0d);
   }

   NDArray zeroLike();

   double sum();

   NDArray sliceSums();

   NDArray rowSums();

   NDArray columnSums();

   default double mean() {
      return sum() / (shape().matrixLength * shape().sliceLength);
   }

   NDArray sliceMeans();

   NDArray div(double value);

   NDArray divi(double value);

   NDArray rdiv(double value);

   NDArray rdivi(double value);

   NDArray mul(double value);

   NDArray muli(double value);

   NDArray sub(double value);

   NDArray subi(double value);

   NDArray rsub(double value);

   NDArray rsubi(double value);


   default NDArray rowMeans() {
      return rowSums().divi(shape().columns());
   }

   default NDArray columnMeans() {
      return columnSums().divi(shape().rows());
   }

   double min();

   NDArray sliceMins();

   NDArray rowMins();

   NDArray columnMins();

   double max();

   NDArray sliceMaxs();

   NDArray rowMaxs();

   NDArray columnMaxs();

   double argmin();

   NDArray sliceArgmins();

   NDArray rowArgmins();

   NDArray columnArgmins();

   double argmax();

   NDArray sliceArgmaxs();

   NDArray rowArgmaxs();

   NDArray columnArgmaxs();

   NDArray diag();

   long size();

   long length();

   NDArray mmul(NDArray rhs);

}//END OF NDArray
