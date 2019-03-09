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
public interface NDArray {

   double scalar();

   NDArray add(double value);

   NDArray addi(double value);

   NDArray add(NDArray rhs);

   NDArray addColumnVector(NDArray rhs);

   NDArray addRowVector(NDArray rhs);

   NDArray addi(NDArray rhs);

   NDArray addiColumnVector(NDArray rhs);

   NDArray addiRowVector(NDArray rhs);

   NDArray div(NDArray rhs);

   NDArray divColumnVector(NDArray rhs);

   NDArray divRowVector(NDArray rhs);

   NDArray divi(NDArray rhs);

   NDArray diviColumnVector(NDArray rhs);

   NDArray diviRowVector(NDArray rhs);

   void forEach(EntryConsumer consumer);

   void forEachSparse(EntryConsumer consumer);

   double get(long i);

   double get(int row, int col);

   boolean isDense();

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

   void set(long i, double value);

   void set(int row, int col, double value);

   Shape shape();

   NDArray slice(int slice);

   NDArray sub(NDArray rhs);

   NDArray subColumnVector(NDArray rhs);

   NDArray subRowVector(NDArray rhs);

   NDArray subi(NDArray rhs);

   NDArray subiColumnVector(NDArray rhs);

   NDArray subiRowVector(NDArray rhs);

   NDArray zeroLike();

   @FunctionalInterface
   interface EntryConsumer {

      void apply(long index, double value);

   }

}//END OF NDArray
