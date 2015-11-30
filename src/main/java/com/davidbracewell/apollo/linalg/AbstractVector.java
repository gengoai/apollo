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

import com.davidbracewell.collection.Collect;
import com.davidbracewell.collection.EnhancedDoubleStatistics;
import com.google.common.base.Preconditions;
import com.google.common.primitives.Doubles;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

/**
 * <p>Abstract base class defining a real-valued vector with basic algebraic opertions. Element indexing is 0-based,
 * meaning the first element in the vector is at index 0. Most methods have a version which constructs a new vector as
 * the result and one in which the operation is applied to itself.
 * </p>
 * <p>While most methods have been define on the abstract class, implementations should take care to override as needed
 * to provide better performance.</p>
 *
 * @author David B. Bracewell
 */
public abstract class AbstractVector implements Serializable, Vector {

  private static final long serialVersionUID = 1L;

  @Override
  public Vector add(Vector rhs) {
    return copy().addSelf(rhs);
  }

  @Override
  public Vector addSelf(Vector rhs) {
    checkDimensions(rhs);
    for (Iterator<Vector.Entry> itr = rhs.nonZeroIterator(); itr.hasNext(); ) {
      Vector.Entry entry = itr.next();
      this.increment(entry.index, entry.value);
    }
    return this;
  }

  /**
   * Check dimensions.
   *
   * @param other the other
   */
  protected void checkDimensions(Vector other) {
    if (other == null) {
      throw new NullPointerException();
    }
    if (other.dimension() != dimension()) {
      throw new IllegalArgumentException("Dimension mismatch " + other.dimension() + " != " + dimension());
    }
  }

  @Override
  public void compress() {

  }

  @Override
  public Vector decrement(int index) {
    return decrement(index, 1);
  }

  @Override
  public Vector decrement(int index, double amount) {
    return set(index, get(index) - amount);
  }

  @Override
  public Vector divide(Vector rhs) {
    return copy().add(rhs);
  }

  @Override
  public Vector divideSelf(Vector rhs) {
    checkDimensions(rhs);
    for (Vector.Entry de : this) {
      this.set(de.index, de.value / rhs.get(de.index));
    }
    return this;
  }

  @Override
  public double dot(Vector rhs) {
    Preconditions.checkNotNull(rhs);
    checkDimensions(rhs);
    Vector smaller = size() < rhs.size() ? this : rhs;
    Vector larger = smaller == rhs ? this : rhs;
    double dot = 0;
    for (Iterator<Vector.Entry> itr = smaller.orderedNonZeroIterator(); itr.hasNext(); ) {
      Vector.Entry de = itr.next();
      dot += de.value * larger.get(de.index);
    }
    return dot;
  }

  @Override
  public Vector increment(int index) {
    return increment(index, 1);
  }

  @Override
  public Vector increment(int index, double amount) {
    return set(index, get(index) + amount);
  }

  @Override
  public boolean isFinite() {
    for (int i = 0; i < dimension(); i++) {
      if (!Doubles.isFinite(get(i))) {
        return false;
      }
    }
    return true;
  }

  @Override
  public boolean isInfinite() {
    for (int i = 0; i < dimension(); i++) {
      if (Double.isInfinite(get(i))) {
        return true;
      }
    }
    return false;
  }

  @Override
  public boolean isNaN() {
    for (int i = 0; i < dimension(); i++) {
      if (Double.isNaN(get(i))) {
        return true;
      }
    }
    return false;
  }

  @Override
  public final boolean isSparse() {
    return !isDense();
  }

  @Override
  public Iterator<Vector.Entry> iterator() {
    return new Iterator<Vector.Entry>() {
      private int index = 0;

      @Override
      public boolean hasNext() {
        return index < dimension();
      }

      @Override
      public Vector.Entry next() {
        if (index >= dimension()) {
          throw new NoSuchElementException();
        }
        Vector.Entry de = new Vector.Entry(index, get(index));
        index++;
        return de;
      }

      @Override
      public void remove() {
        throw new UnsupportedOperationException();
      }
    };
  }

  @Override
  public double l1Norm() {
    double l1 = 0d;
    for (Iterator<Vector.Entry> itr = nonZeroIterator(); itr.hasNext(); ) {
      l1 += Math.abs(itr.next().value);
    }
    return l1;
  }

  @Override
  public double lInfNorm() {
    double lInf = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < dimension(); i++) {
      lInf = Math.max(lInf, Math.abs(get(i)));
    }
    return lInf;
  }

  @Override
  public double magnitude() {
    return Math.sqrt(dot(this));
  }

  @Override
  public Vector map(DoubleUnaryOperator function) {
    Preconditions.checkNotNull(function);
    return copy().map(function);
  }

  @Override
  public Vector map(Vector v, DoubleBinaryOperator function) {
    return copy().mapSelf(v, function);
  }

  @Override
  public Vector mapAdd(double amount) {
    return copy().mapAddSelf(amount);
  }

  @Override
  public Vector mapAddSelf(final double amount) {
    if (amount != 0) {
      return mapSelf(a -> a + amount);
    }
    return this;
  }

  @Override
  public Vector mapDivide(double amount) {
    return copy().mapDivideSelf(amount);
  }

  @Override
  public Vector mapDivideSelf(double amount) {
    if (!Doubles.isFinite(amount) || amount == 0) {
      for (int i = 0; i < dimension(); i++) {
        set(i, get(i) / amount);
      }
    } else {
      for (Iterator<Vector.Entry> itr = nonZeroIterator(); itr.hasNext(); ) {
        Vector.Entry de = itr.next();
        set(de.index, de.value / amount);
      }
    }
    return this;
  }

  @Override
  public Vector mapMultiply(double amount) {
    return copy().mapMultiplySelf(amount);
  }

  @Override
  public Vector mapMultiplySelf(double amount) {
    if (!Doubles.isFinite(amount)) {
      for (int i = 0; i < dimension(); i++) {
        set(i, get(i) * amount);
      }
    } else {
      for (Iterator<Vector.Entry> itr = nonZeroIterator(); itr.hasNext(); ) {
        Vector.Entry de = itr.next();
        set(de.index, de.value * amount);
      }
    }
    return this;
  }

  @Override
  public Vector mapSelf(Vector v, DoubleBinaryOperator function) {
    checkDimensions(v);
    Preconditions.checkNotNull(function);
    for (int i = 0; i < dimension(); i++) {
      set(i, function.applyAsDouble(get(i), v.get(i)));
    }
    return this;
  }

  @Override
  public Vector mapSelf(DoubleUnaryOperator function) {
    Preconditions.checkNotNull(function);
    for (int i = 0; i < dimension(); i++) {
      set(i, function.applyAsDouble(get(i)));
    }
    return this;
  }

  @Override
  public Vector mapSubtract(double amount) {
    return copy().mapSubtractSelf(amount);
  }

  @Override
  public Vector mapSubtractSelf(final double amount) {
    if (amount != 0) {
      return mapSelf(a -> a - amount);
    }
    return this;
  }

  @Override
  public double max() {
    return statistics().getMax();
  }

  @Override
  public double min() {
    return statistics().getMin();
  }

  @Override
  public Vector multiply(Vector rhs) {
    return copy().add(rhs);
  }

  @Override
  public Vector multiplySelf(Vector rhs) {
    checkDimensions(rhs);
    for (Vector.Entry de : this) {
      this.set(de.index, de.value * rhs.get(de.index));
    }
    return this;
  }

  @Override
  public Iterator<Vector.Entry> nonZeroIterator() {
    return orderedNonZeroIterator();
  }

  @Override
  public Iterator<Vector.Entry> orderedNonZeroIterator() {
    return new Iterator<Vector.Entry>() {
      Iterator<Vector.Entry> backing = iterator();
      Vector.Entry next = null;

      private void advance() {
        if (next == null) {
          while (backing.hasNext()) {
            next = backing.next();
            if (next.value != 0) {
              return;
            }
          }
        }
      }

      @Override
      public boolean hasNext() {
        advance();
        return next != null;
      }

      @Override
      public Vector.Entry next() {
        advance();
        if (next == null) {
          throw new NoSuchElementException();
        }
        Vector.Entry rval = next;
        next = null;
        return rval;
      }

      @Override
      public void remove() {
        throw new UnsupportedOperationException();
      }
    };
  }

  @Override
  public Vector scale(int index, double amount) {
    return set(index, get(index) * amount);
  }

  @Override
  public EnhancedDoubleStatistics statistics() {
    return Collect.from(nonZeroIterator())
      .mapToDouble(Vector.Entry::getValue)
      .collect(EnhancedDoubleStatistics::new, EnhancedDoubleStatistics::accept, EnhancedDoubleStatistics::combine);
  }

  @Override
  public Vector subtract(Vector rhs) {
    return copy().add(rhs);
  }

  @Override
  public Vector subtractSelf(Vector rhs) {
    checkDimensions(rhs);
    for (Iterator<Vector.Entry> itr = rhs.nonZeroIterator(); itr.hasNext(); ) {
      Vector.Entry de = itr.next();
      this.set(de.index, de.value + rhs.get(de.index));
    }
    return this;
  }

  @Override
  public double sum() {
    return statistics().getSum();
  }

  @Override
  public double[] toArray() {
    double[] array = new double[dimension()];
    for (int i = 0; i < dimension(); i++) {
      array[i] = get(i);
    }
    return array;
  }

  @Override
  public String toString() {
    if (dimension() > 20) {
      return getClass().getSimpleName() + "[" + dimension() + "]";
    }
    return Arrays.toString(toArray());
  }

  @Override
  public Vector zero() {
    return mapMultiplySelf(0d);
  }


}//END OF Vector
