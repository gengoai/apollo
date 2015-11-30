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

import com.google.common.base.Preconditions;

import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

/**
 * @author David B. Bracewell
 */
public class UnmodifiableVector extends ForwardingVector {

  private static final long serialVersionUID = 1L;
  private final Vector delegate;

  public UnmodifiableVector(Vector delegate) {
    this.delegate = Preconditions.checkNotNull(delegate);
  }

  @Override
  protected Vector delegate() {
    return delegate;
  }

  @Override
  public Vector addSelf(Vector rhs) {
    throw new UnsupportedOperationException();
  }

  @Override
  public Vector divideSelf(Vector rhs) {
    throw new UnsupportedOperationException();
  }

  @Override
  public Vector decrement(int index, double amount) {
    throw new UnsupportedOperationException();
  }

  @Override
  public Vector decrement(int index) {
    throw new UnsupportedOperationException();
  }

  @Override
  public Vector increment(int index) {
    throw new UnsupportedOperationException();
  }

  @Override
  public Vector increment(int index, double amount) {
    throw new UnsupportedOperationException();
  }

  @Override
  public Vector mapAddSelf(double amount) {
    throw new UnsupportedOperationException();
  }

  @Override
  public Vector mapDivideSelf(double amount) {
    throw new UnsupportedOperationException();
  }

  @Override
  public Vector mapMultiplySelf(double amount) {
    throw new UnsupportedOperationException();
  }

  @Override
  public Vector mapSelf(Vector v, DoubleBinaryOperator function) {
    throw new UnsupportedOperationException();
  }

  @Override
  public Vector mapSubtractSelf(double amount) {
    throw new UnsupportedOperationException();
  }

  @Override
  public Vector multiplySelf(Vector rhs) {
    throw new UnsupportedOperationException();
  }

  @Override
  public Vector scale(int index, double amount) {
    throw new UnsupportedOperationException();
  }

  @Override
  public Vector set(int index, double value) {
    throw new UnsupportedOperationException();
  }

  @Override
  public Vector subtractSelf(Vector rhs) {
    throw new UnsupportedOperationException();
  }

  @Override
  public Vector mapSelf(DoubleUnaryOperator function) {
    throw new UnsupportedOperationException();
  }
}//END OF UnmodifiableVector
