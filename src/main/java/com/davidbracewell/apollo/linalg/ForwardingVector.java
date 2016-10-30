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
 */

package com.davidbracewell.apollo.linalg;

import com.davidbracewell.EnhancedDoubleStatistics;

import java.io.Serializable;
import java.util.Iterator;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

/**
 * The type Forwarding vector.
 *
 * @author David B. Bracewell
 */
public abstract class ForwardingVector implements Vector, Serializable {

   private static final long serialVersionUID = 1L;

   @Override
   public Vector addSelf(Vector rhs) {
      delegate().addSelf(rhs);
      return this;
   }

   @Override
   public Vector compress() {
      delegate().compress();
      return this;
   }

   @Override
   public Vector copy() {
      return delegate().copy();
   }

   @Override
   public Vector decrement(int index) {
      delegate().decrement(index);
      return this;
   }

   @Override
   public Vector decrement(int index, double amount) {
      delegate().decrement(index, amount);
      return this;
   }

   /**
    * Delegate vector.
    *
    * @return the vector
    */
   protected abstract Vector delegate();

   @Override
   public int dimension() {
      return delegate().dimension();
   }

   @Override
   public Vector divideSelf(Vector rhs) {
      delegate().divideSelf(rhs);
      return this;
   }

   @Override
   public boolean equals(Object o) {
      return delegate().equals(o);
   }

   @Override
   public boolean isSparse() {
      return delegate().isSparse();
   }

   @Override
   public double get(int index) {
      return delegate().get(index);
   }

   @Override
   public int hashCode() {
      return delegate().hashCode();
   }

   @Override
   public Vector increment(int index) {
      delegate().increment(index);
      return this;
   }

   @Override
   public Vector increment(int index, double amount) {
      delegate().increment(index, amount);
      return this;
   }

   @Override
   public boolean isDense() {
      return delegate().isDense();
   }

   @Override
   public Iterator<Vector.Entry> iterator() {
      return delegate().iterator();
   }

   @Override
   public Vector mapAddSelf(double amount) {
      delegate().mapAddSelf(amount);
      return this;
   }

   @Override
   public Vector mapDivideSelf(double amount) {
      delegate().mapDivideSelf(amount);
      return this;
   }

   @Override
   public Vector mapMultiplySelf(double amount) {
      delegate().mapMultiplySelf(amount);
      return this;
   }

   @Override
   public Vector mapSelf(DoubleUnaryOperator function) {
      delegate().mapSelf(function);
      return this;
   }

   @Override
   public Vector mapSelf(Vector v, DoubleBinaryOperator function) {
      delegate().mapSelf(v, function);
      return this;
   }

   @Override
   public Vector mapSubtractSelf(double amount) {
      delegate().mapSubtractSelf(amount);
      return this;
   }

   @Override
   public Vector multiplySelf(Vector rhs) {
      delegate().multiplySelf(rhs);
      return this;
   }

   @Override
   public Iterator<Vector.Entry> nonZeroIterator() {
      return delegate().nonZeroIterator();
   }

   @Override
   public Iterator<Vector.Entry> orderedNonZeroIterator() {
      return delegate().orderedNonZeroIterator();
   }

   @Override
   public Vector scale(int index, double amount) {
      delegate().scale(index, amount);
      return this;
   }

   @Override
   public Vector set(int index, double value) {
      delegate().set(index, value);
      return this;
   }

   @Override
   public int size() {
      return delegate().size();
   }

   @Override
   public Vector slice(int from, int to) {
      return delegate().slice(from, to);
   }

   @Override
   public EnhancedDoubleStatistics statistics() {
      return delegate().statistics();
   }

   @Override
   public Vector subtractSelf(Vector rhs) {
      delegate().subtractSelf(rhs);
      return this;
   }

   @Override
   public double sum() {
      return delegate().sum();
   }

   @Override
   public double[] toArray() {
      return delegate().toArray();
   }

   @Override
   public String toString() {
      return delegate().toString();
   }

   @Override
   public Vector zero() {
      return delegate().zero();
   }

   @Override
   public Vector redim(int newDimension) {
      return delegate().redim(newDimension);
   }

}//END OF ForwardingVector
