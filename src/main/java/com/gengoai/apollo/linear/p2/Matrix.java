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
      checkLength(shape, rhs.shape());
      NDArray out = zeroLike();
      forEach((index, value) -> out.set(index, value + rhs.get(index)));
      return out;
   }

   @Override
   public NDArray add(double value) {
      if (value == 0) {
         return copy();
      }
      NDArray out = zeroLike();
      forEach((index, v) -> out.set(index, v + value));
      return out;
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
      checkLength(shape, rhs.shape());
      rhs.forEachSparse((index, value) -> set(index, value + get(index)));
      return this;
   }

   @Override
   public NDArray addi(double value) {
      if (value == 0) {
         return this;
      }
      forEach((index, v) -> set(index, v + value));
      return this;
   }

   @Override
   public NDArray addiColumnVector(NDArray rhs) {
      checkLength(shape.rows(), rhs.shape());
      for (int column = 0; column < shape.columns(); column++) {
         final int c = column;
         rhs.forEachSparse((row, value) -> set((int) row, c, get((int) row, c) + value));
      }
      return this;
   }

   @Override
   public NDArray addiRowVector(NDArray rhs) {
      checkLength(shape.columns(), rhs.shape());
      rhs.forEachSparse((column, value) -> {
         for (int row = 0; row < shape.rows(); row++) {
            set(row, (int) column, get(row, (int) column) + rhs.get(column));
         }
      });
      return this;
   }

   public abstract DenseMatrix asDense();

   public void checkLength(int l) {
      if (shape.matrixLength != l) {
         throw new IllegalArgumentException("Length Mismatch: " + shape.matrixLength + " != " + l);
      }
   }

   @Override
   public NDArray div(NDArray rhs) {
      checkLength(shape, rhs.shape());
      NDArray out = zeroLike();
      forEach((index, value) -> out.set(index, value / rhs.get(index)));
      return out;
   }

   @Override
   public NDArray divColumnVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray divRowVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray divi(NDArray rhs) {
      checkLength(shape, rhs.shape());
      forEach((index, value) -> set(index, value / rhs.get(index)));
      return this;
   }

   @Override
   public NDArray diviColumnVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray diviRowVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray mapColumn(NDArray rhs, final DoubleBinaryOperator operator) {
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
   public NDArray mul(NDArray rhs) {
      checkLength(shape, rhs.shape());
      NDArray out = zeroLike();
      forEach((index, value) -> out.set(index, value * rhs.get(index)));
      return out;
   }

   @Override
   public NDArray mulColumnVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray mulRowVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray muli(NDArray rhs) {
      checkLength(shape, rhs.shape());
      forEachSparse((index, value) -> set(index, value * rhs.get(index)));
      return this;
   }

   @Override
   public NDArray muliColumnVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray muliRowVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray rdiv(NDArray lhs) {
      checkLength(shape, lhs.shape());
      NDArray out = zeroLike();
      forEach((index, value) -> out.set(index, lhs.get(index) / value));
      return out;
   }

   @Override
   public NDArray rdivColumnVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray rdivRowVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray rdivi(NDArray lhs) {
      checkLength(shape, lhs.shape());
      forEach((index, value) -> set(index, lhs.get(index) / value));
      return this;
   }

   @Override
   public NDArray rdiviColumnVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray rdiviRowVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray rsub(NDArray lhs) {
      checkLength(shape, lhs.shape());
      NDArray out = zeroLike();
      lhs.forEach((index, value) -> out.set(index, value - get(index)));
      return out;
   }

   @Override
   public NDArray rsubColumnVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray rsubRowVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray rsubi(NDArray lhs) {
      checkLength(shape, lhs.shape());
      lhs.forEach((index, value) -> set(index, value - get(index)));
      return this;
   }

   @Override
   public NDArray rsubiColumnVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray rsubiRowVector(NDArray rhs) {
      return null;
   }

   @Override
   public Shape shape() {
      return shape;
   }

   @Override
   public NDArray slice(int slice) {
      return null;
   }

   @Override
   public NDArray sub(NDArray rhs) {
      checkLength(shape, rhs.shape());
      NDArray out = zeroLike();
      forEach((index, value) -> out.set(index, value - rhs.get(index)));
      return out;
   }

   @Override
   public NDArray subColumnVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray subRowVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray subi(NDArray rhs) {
      checkLength(shape, rhs.shape());
      rhs.forEachSparse((index, value) -> set(index, get(index) - value));
      return this;
   }

   @Override
   public NDArray subiColumnVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray subiRowVector(NDArray rhs) {
      return null;
   }

   @Override
   public NDArray zeroLike() {
      return null;
   }


}//END OF NDArray
