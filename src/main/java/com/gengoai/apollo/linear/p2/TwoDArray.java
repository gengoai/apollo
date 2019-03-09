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

/**
 * @author David B. Bracewell
 */
public abstract class TwoDArray implements NDArray {
   protected final Shape shape;

   protected TwoDArray(Shape shape) {
      Validation.notNull(shape);
      this.shape = shape.copy();
   }

   public NDArray add(NDArray rhs) {
      Validation.checkArgument(rhs.shape().sliceLength == 1, () -> sizeMismatch(rhs.shape()));
      if (shape.isSameLength(rhs.shape())) {
         return addi(rhs, zeroLike());
      } else if (shape.isRowBroadcastable(rhs.shape())) {
         return addRowVector(rhs);
      } else if (shape.isColumnBroadcastable(rhs.shape())) {
         return addColumnVector(rhs);
      } else if (rhs.shape().isScalar()) {
         return add(rhs.scalar());
      } else if (shape.isMultiBroadcastable(rhs.shape())) {
         //??
      }
      throw new IllegalArgumentException(sizeMismatch(rhs.shape()));
   }

   public NDArray addi(NDArray rhs) {
      Validation.checkArgument(rhs.shape().sliceLength == 1, () -> sizeMismatch(rhs.shape()));
      if (shape.isSameLength(rhs.shape())) {
         return addi(rhs, this);
      } else if (shape.isRowBroadcastable(rhs.shape())) {
         return addRowVector(rhs);
      } else if (shape.isColumnBroadcastable(rhs.shape())) {
         return addColumnVector(rhs);
      } else if (rhs.shape().isScalar()) {
         return add(rhs.scalar());
      } else if (shape.isMultiBroadcastable(rhs.shape())) {
         //??
      }
      throw new IllegalArgumentException(sizeMismatch(rhs.shape()));
   }

   protected NDArray addi(NDArray rhs, NDArray out) {
      rhs.forEachSparse((index, value) -> out.set(index, value + get(index)));
      return out;
   }




   public abstract DenseTwoDArray asDense();

   @Override
   public Shape shape() {
      return shape;
   }

   public String sizeMismatch(Shape rhs) {
      return "Size Mismatch: " + shape + " != " + rhs;
   }
}//END OF NDArray
