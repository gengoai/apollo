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

import com.davidbracewell.collection.EnhancedDoubleStatistics;

import java.io.Serializable;
import java.util.Iterator;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

/**
 * @author David B. Bracewell
 */
public abstract class ForwardingVector implements Vector, Serializable {

  private static final long serialVersionUID = 1L;

  @Override
  public Vector add(Vector rhs) {
    return delegate().add(rhs);
  }

  @Override
  public Vector addSelf(Vector rhs) {
    return delegate().addSelf(rhs);
  }

  @Override
  public void compress() {
    delegate().compress();
  }

  @Override
  public Vector copy() {
    return delegate().copy();
  }

  @Override
  public Vector decrement(int index) {
    return delegate().decrement(index);
  }

  @Override
  public Vector decrement(int index, double amount) {
    return delegate().decrement(index, amount);
  }

  protected abstract Vector delegate();

  @Override
  public int dimension() {
    return delegate().dimension();
  }

  @Override
  public Vector divide(Vector rhs) {
    return delegate().divide(rhs);
  }

  @Override
  public Vector divideSelf(Vector rhs) {
    return delegate().divideSelf(rhs);
  }

  @Override
  public double dot(Vector rhs) {
    return delegate().dot(rhs);
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
    return delegate().increment(index);
  }

  @Override
  public Vector increment(int index, double amount) {
    return delegate().increment(index, amount);
  }

  @Override
  public boolean isDense() {
    return delegate().isDense();
  }

  @Override
  public boolean isFinite() {
    return delegate().isFinite();
  }

  @Override
  public boolean isInfinite() {
    return delegate().isInfinite();
  }

  @Override
  public boolean isNaN() {
    return delegate().isNaN();
  }

  @Override
  public Iterator<Vector.Entry> iterator() {
    return delegate().iterator();
  }

  @Override
  public double l1Norm() {
    return delegate().l1Norm();
  }

  @Override
  public double lInfNorm() {
    return delegate().lInfNorm();
  }

  @Override
  public double magnitude() {
    return delegate().magnitude();
  }

  @Override
  public Vector map(DoubleUnaryOperator function) {
    return delegate().map(function);
  }

  @Override
  public Vector map(Vector v, DoubleBinaryOperator function) {
    return delegate().map(v, function);
  }

  @Override
  public Vector mapAdd(double amount) {
    return delegate().mapAdd(amount);
  }

  @Override
  public Vector mapAddSelf(double amount) {
    return delegate().mapAddSelf(amount);
  }

  @Override
  public Vector mapDivide(double amount) {
    return delegate().mapDivide(amount);
  }

  @Override
  public Vector mapDivideSelf(double amount) {
    return delegate().mapDivideSelf(amount);
  }

  @Override
  public Vector mapMultiply(double amount) {
    return delegate().mapMultiply(amount);
  }

  @Override
  public Vector mapMultiplySelf(double amount) {
    return delegate().mapMultiplySelf(amount);
  }

  @Override
  public Vector mapSelf(DoubleUnaryOperator function) {
    return delegate().mapSelf(function);
  }

  @Override
  public Vector mapSelf(Vector v, DoubleBinaryOperator function) {
    return delegate().mapSelf(v, function);
  }

  @Override
  public Vector mapSubtract(double amount) {
    return delegate().mapSubtract(amount);
  }

  @Override
  public Vector mapSubtractSelf(double amount) {
    return delegate().mapSubtractSelf(amount);
  }

  @Override
  public double max() {
    return delegate().max();
  }

  @Override
  public double min() {
    return delegate().min();
  }

  @Override
  public Vector multiply(Vector rhs) {
    return delegate().multiply(rhs);
  }

  @Override
  public Vector multiplySelf(Vector rhs) {
    return delegate().multiplySelf(rhs);
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
    return delegate().scale(index, amount);
  }

  @Override
  public Vector set(int index, double value) {
    return delegate().set(index, value);
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
  public Vector subtract(Vector rhs) {
    return delegate().subtract(rhs);
  }

  @Override
  public Vector subtractSelf(Vector rhs) {
    return delegate().subtractSelf(rhs);
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
