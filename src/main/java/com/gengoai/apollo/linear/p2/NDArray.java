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

import com.gengoai.apollo.linear.Shape;

/**
 * @author David B. Bracewell
 */
public abstract class NDArray {
   protected final Shape shape;

   @FunctionalInterface
   public interface EntryConsumer {

      void apply(int index, double value);

   }

   protected NDArray(Shape shape) {
      this.shape = shape;
   }

   public abstract NDArray zeroLike();

   public NDArray add(NDArray rhs) {
      return addi(rhs, zeroLike());
   }

   public abstract NDArray addRowVector(NDArray rhs);

   public abstract NDArray addColumnVector(NDArray rhs);

   public NDArray addi(NDArray rhs) {
      return addi(rhs, this);
   }

   public abstract NDArray addiRowVector(NDArray rhs);

   public abstract NDArray addiColumnVector(NDArray rhs);

   protected NDArray addi(NDArray rhs, NDArray out) {
      rhs.forEachSparse((index, value) -> out.set(index, value + get(index)));
      return out;
   }

   public abstract double get(int i);

   public abstract double get(int row, int col);

   public abstract void set(int i, double value);

   public abstract void set(int row, int col, double value);

   protected abstract void forEach(EntryConsumer consumer);

   protected abstract void forEachSparse(EntryConsumer consumer);


   public abstract boolean isDense();

   public abstract DenseNDArray asDense();

}//END OF NDArray
