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

import com.gengoai.Validation;
import com.gengoai.apollo.linear.Shape;
import com.gengoai.math.Operator;

import java.util.function.DoubleBinaryOperator;

/**
 * @author David B. Bracewell
 */
public abstract class Matrix implements NDArray {
   protected final Shape shape;

   protected Matrix(Shape shape) {
      Validation.notNull(shape);
      this.shape = shape.copy();
   }

   protected static void checkLength(Shape s1, Shape s2) {
      if (s2.sliceLength > 1 || s1.sliceLength > 1) {
         throw new IllegalArgumentException("Slice Mismatch: " + s1.sliceLength + " != " + s2.sliceLength);
      }
      if (s1.matrixLength != s2.matrixLength) {
         throw new IllegalArgumentException("Length Mismatch: " + s1 + " != " + s2);
      }
   }

   protected static void checkLength(int dim, Shape s2) {
      if (s2.sliceLength > 1) {
         throw new IllegalArgumentException("Slice Mismatch: " + s2.sliceLength + " != 1");
      }
      if (dim != s2.matrixLength) {
         throw new IllegalArgumentException("Length Mismatch: " + s2.matrixLength + " != " + dim);
      }
   }

   @Override
   public NDArray add(NDArray rhs) {
      return map(rhs, Operator::add);
   }

   @Override
   public NDArray add(double value) {
      if (value == 0) {
         return copy();
      }
      return map(value, Operator::add);
   }

   @Override
   public NDArray addColumnVector(NDArray rhs) {
      return mapColumn(rhs, Operator::add);
   }

   @Override
   public NDArray addRowVector(NDArray rhs) {
      return mapRow(rhs, Operator::add);
   }

   @Override
   public NDArray addi(NDArray rhs) {
      return mapi(rhs, Operator::add);
   }

   @Override
   public NDArray addi(double value) {
      if (value == 0) {
         return this;
      }
      return mapi(value, Operator::add);
   }

   @Override
   public NDArray addiColumnVector(NDArray rhs) {
      return mapiColumn(rhs, Operator::add);
   }

   @Override
   public NDArray addiRowVector(NDArray rhs) {
      return mapiRow(rhs, Operator::add);
   }

   @Override
   public NDArray div(NDArray rhs) {
      return map(rhs, Operator::divide);
   }

   @Override
   public NDArray divColumnVector(NDArray rhs) {
      return mapColumn(rhs, Operator::divide);
   }

   @Override
   public NDArray divRowVector(NDArray rhs) {
      return mapRow(rhs, Operator::divide);
   }

   @Override
   public NDArray divi(NDArray rhs) {
      return mapi(rhs, Operator::divide);
   }

   @Override
   public NDArray diviColumnVector(NDArray rhs) {
      return mapiColumn(rhs, Operator::divide);
   }

   @Override
   public NDArray diviRowVector(NDArray rhs) {
      return mapiRow(rhs, Operator::divide);
   }

   @Override
   public NDArray fill(double value) {
      for (int i = 0; i < shape.matrixLength; i++) {
         set(i, value);
      }
      return this;
   }

   @Override
   public double get(int channel, int row, int col) {
      if (channel == 0) {
         return get(row, col);
      }
      throw new IndexOutOfBoundsException();
   }

   @Override
   public double get(int kernel, int channel, int row, int col) {
      if (channel == 0 && kernel == 0) {
         return get(row, col);
      }
      throw new IndexOutOfBoundsException();
   }

   @Override
   public NDArray mapi(double value, DoubleBinaryOperator operator) {
      for (int i = 0; i < shape.matrixLength; i++) {
         set(i, operator.applyAsDouble(get(i), value));
      }
      return this;
   }

   @Override
   public NDArray map(double value, DoubleBinaryOperator operator) {
      NDArray out = zeroLike();
      for (int i = 0; i < shape.matrixLength; i++) {
         out.set(i, operator.applyAsDouble(get(i), value));
      }
      return out;
   }

   @Override
   public NDArray map(NDArray rhs, DoubleBinaryOperator operator) {
      if (rhs.shape().isScalar()) {
         return map(rhs.scalar(), operator);
      }
      checkLength(shape, rhs.shape());
      NDArray out = zeroLike();
      for (int i = 0; i < shape.matrixLength; i++) {
         out.set(i, operator.applyAsDouble(get(i), rhs.get(i)));
      }
      return out;
   }

   @Override
   public NDArray mapColumn(NDArray rhs, final DoubleBinaryOperator operator) {
      if (rhs.shape().isScalar()) {
         return map(rhs.scalar(), operator);
      }
      checkLength(shape.rows(), rhs.shape());
      NDArray out = zeroLike();
      for (int column = 0; column < shape.columns(); column++) {
         for (int row = 0; row < shape.rows(); row++) {
            out.set(row, column, operator.applyAsDouble(get(row, column), rhs.get(row)));
         }
      }
      return out;
   }

   @Override
   public NDArray mapRow(NDArray rhs, DoubleBinaryOperator operator) {
      if (rhs.shape().isScalar()) {
         return map(rhs.scalar(), operator);
      }
      checkLength(shape.columns(), rhs.shape());
      NDArray out = zeroLike();
      for (int column = 0; column < shape.columns(); column++) {
         for (int row = 0; row < shape.rows(); row++) {
            out.set(row, column, operator.applyAsDouble(get(row, column), rhs.get(column)));
         }
      }
      return out;
   }

   @Override
   public NDArray mapi(NDArray rhs, DoubleBinaryOperator operator) {
      if (rhs.shape().isScalar()) {
         return mapi(rhs.scalar(), operator);
      }
      checkLength(shape, rhs.shape());
      return mapi(rhs, operator);
   }

   @Override
   public NDArray mapiColumn(NDArray rhs, final DoubleBinaryOperator operator) {
      if (rhs.shape().isScalar()) {
         return mapi(rhs.scalar(), operator);
      }
      checkLength(shape.rows(), rhs.shape());
      for (int column = 0; column < shape.columns(); column++) {
         for (int row = 0; row < shape.rows(); row++) {
            set(row, column, operator.applyAsDouble(get(row, column), rhs.get(row)));
         }
      }
      return this;
   }

   @Override
   public NDArray mapiRow(NDArray rhs, DoubleBinaryOperator operator) {
      if (rhs.shape().isScalar()) {
         return mapi(rhs.scalar(), operator);
      }
      checkLength(shape.columns(), rhs.shape());
      for (int column = 0; column < shape.columns(); column++) {
         for (int row = 0; row < shape.rows(); row++) {
            set(row, column, operator.applyAsDouble(get(row, column), rhs.get(column)));
         }
      }
      return this;
   }

   @Override
   public NDArray mul(NDArray rhs) {
      return map(rhs, Operator::multiply);
   }

   @Override
   public NDArray mulColumnVector(NDArray rhs) {
      return mapColumn(rhs, Operator::multiply);
   }

   @Override
   public NDArray mulRowVector(NDArray rhs) {
      return mapRow(rhs, Operator::multiply);
   }

   @Override
   public NDArray muli(NDArray rhs) {
      return mapi(rhs, Operator::multiply);
   }

   @Override
   public NDArray muliColumnVector(NDArray rhs) {
      return mapiColumn(rhs, Operator::multiply);
   }

   @Override
   public NDArray muliRowVector(NDArray rhs) {
      return mapiRow(rhs, Operator::multiply);
   }

   @Override
   public NDArray rdiv(NDArray lhs) {
      return map(lhs, (v1, v2) -> v2 / v1);
   }

   @Override
   public NDArray rdivColumnVector(NDArray rhs) {
      return mapColumn(rhs, (v1, v2) -> v2 / v1);
   }

   @Override
   public NDArray rdivRowVector(NDArray rhs) {
      return mapRow(rhs, (v1, v2) -> v2 / v1);
   }

   @Override
   public NDArray rdivi(NDArray lhs) {
      return mapi(lhs, (v1, v2) -> v2 / v1);
   }

   @Override
   public NDArray rdiviColumnVector(NDArray rhs) {
      return mapiColumn(rhs, (v1, v2) -> v2 / v1);
   }

   @Override
   public NDArray rdiviRowVector(NDArray rhs) {
      return mapiRow(rhs, (v1, v2) -> v2 / v1);
   }

   @Override
   public NDArray rsub(NDArray lhs) {
      return map(lhs, (v1, v2) -> v2 - v1);
   }

   @Override
   public NDArray rsubColumnVector(NDArray rhs) {
      return mapColumn(rhs, (v1, v2) -> v2 - v1);
   }

   @Override
   public NDArray rsubRowVector(NDArray rhs) {
      return mapRow(rhs, (v1, v2) -> v2 - v1);
   }

   @Override
   public NDArray rsubi(NDArray lhs) {
      return mapi(lhs, (v1, v2) -> v2 - v1);
   }

   @Override
   public NDArray rsubiColumnVector(NDArray rhs) {
      return mapiColumn(rhs, (v1, v2) -> v2 - v1);
   }

   @Override
   public NDArray rsubiRowVector(NDArray rhs) {
      return mapiRow(rhs, (v1, v2) -> v2 - v1);
   }

   @Override
   public void set(int channel, int row, int col, double value) {
      if (channel == 0) {
         set(row, col, value);
      }
      throw new IndexOutOfBoundsException();
   }

   @Override
   public void set(int kernel, int channel, int row, int col, double value) {
      if (channel == 0 && kernel == 0) {
         set(row, col, value);
      }
      throw new IndexOutOfBoundsException();
   }

   @Override
   public Shape shape() {
      return shape;
   }

   @Override
   public NDArray slice(int slice) {
      return this;
   }

   @Override
   public NDArray sub(NDArray rhs) {
      return map(rhs, Operator::subtract);
   }

   @Override
   public NDArray subColumnVector(NDArray rhs) {
      return mapColumn(rhs, Operator::subtract);
   }

   @Override
   public NDArray subRowVector(NDArray rhs) {
      return mapRow(rhs, Operator::subtract);
   }

   @Override
   public NDArray subi(NDArray rhs) {
      return mapi(rhs, Operator::subtract);
   }

   @Override
   public NDArray subiColumnVector(NDArray rhs) {
      return mapiColumn(rhs, Operator::subtract);
   }

   @Override
   public NDArray subiRowVector(NDArray rhs) {
      return mapiRow(rhs, Operator::subtract);
   }


}//END OF NDArray
