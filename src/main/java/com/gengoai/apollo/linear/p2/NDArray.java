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

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public abstract class NDArray implements Serializable {
   protected final Shape shape;

   public NDArray(Shape shape) {
      this.shape = shape.copy();
   }

   public abstract NDArray add(NDArray rhs);

   public abstract NDArray add(double rhs);

   public abstract NDArray addColumnVector(NDArray rhs);

   public abstract NDArray addRowVector(NDArray rhs);

   public abstract NDArray addi(NDArray rhs);

   public abstract NDArray addi(double rhs);

   public abstract NDArray addiColumnVector(NDArray rhs);

   public abstract NDArray addiRowVector(NDArray rhs);

   public abstract NDArray div(NDArray rhs);

   public abstract NDArray div(double rhs);

   public abstract NDArray divColumnVector(NDArray rhs);

   public abstract NDArray divRowVector(NDArray rhs);

   public abstract NDArray divi(NDArray rhs);

   public abstract NDArray divi(double rhs);

   public abstract NDArray diviColumnVector(NDArray rhs);

   public abstract NDArray diviRowVector(NDArray rhs);

   public abstract NDArray get(int i);

   public abstract NDArray get(int i, int j);

   public abstract NDArray get(int i, int j, int k);

   public abstract NDArray get(int i, int j, int k, int l);

   public abstract NDArray mul(NDArray rhs);

   public abstract NDArray mul(double rhs);

   public abstract NDArray mulColumnVector(NDArray rhs);

   public abstract NDArray mulRowVector(NDArray rhs);

   public abstract NDArray muli(NDArray rhs);

   public abstract NDArray muli(double rhs);

   public abstract NDArray muliColumnVector(NDArray rhs);

   public abstract NDArray muliRowVector(NDArray rhs);

   public abstract NDArray rdiv(NDArray rhs);

   public abstract NDArray rdiv(double rhs);

   public abstract NDArray rdivColumnVector(NDArray rhs);

   public abstract NDArray rdivRowVector(NDArray rhs);

   public abstract NDArray rdivi(NDArray rhs);

   public abstract NDArray rdivi(double rhs);

   public abstract NDArray rdiviColumnVector(NDArray rhs);

   public abstract NDArray rdiviRowVector(NDArray rhs);

   public abstract NDArray rsub(NDArray rhs);

   public abstract NDArray rsub(double rhs);

   public abstract NDArray rsubColumnVector(NDArray rhs);

   public abstract NDArray rsubRowVector(NDArray rhs);

   public abstract NDArray rsubi(NDArray rhs);

   public abstract NDArray rsubi(double rhs);

   public abstract NDArray rsubiColumnVector(NDArray rhs);

   public abstract NDArray rsubiRowVector(NDArray rhs);

   public abstract NDArray set(int i, double value);

   public abstract NDArray set(int i, int j, double value);

   public abstract NDArray set(int i, int j, int k, double value);

   public abstract NDArray set(int i, int j, int k, int l, double value);

   public abstract NDArray sub(NDArray rhs);

   public abstract NDArray sub(double rhs);

   public abstract NDArray subColumnVector(NDArray rhs);

   public abstract NDArray subRowVector(NDArray rhs);

   public abstract NDArray subi(NDArray rhs);

   public abstract NDArray subi(double rhs);

   public abstract NDArray subiColumnVector(NDArray rhs);

   public abstract NDArray subiRowVector(NDArray rhs);

}//END OF NDArray
