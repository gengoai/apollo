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

import org.junit.Test;

import java.util.Iterator;

import static com.davidbracewell.tuple.Tuples.$;
import static org.junit.Assert.*;

/**
 * @author David B. Bracewell
 */
public abstract class AbstractVectorTest {

   Vector v;

   @Test
   public void add() throws Exception {
      assertEquals(10, v.add(DenseVector.zeros(10)).sum(), 0);
      assertEquals(20, v.add(SparseVector.ones(10)).sum(), 0);
   }

   @Test(expected = IllegalArgumentException.class)
   public void addError() throws Exception {
      assertEquals(10, v.add(DenseVector.zeros(100)).sum(), 0);
   }

   @Test
   public void addSelf() throws Exception {
      assertEquals(10, v.addSelf(DenseVector.zeros(10)).sum(), 0);
      assertEquals(20, v.addSelf(SparseVector.ones(10)).sum(), 0);
   }

   @Test(expected = IllegalArgumentException.class)
   public void addSelfError() throws Exception {
      assertEquals(10, v.addSelf(DenseVector.zeros(100)).sum(), 0);
   }

   @Test
   public void getLabel() throws Exception {
      assertNull(v.getLabel());
   }

   @Test
   public void withLabel() throws Exception {
      assertEquals("test", v.setLabel("test").getLabel());
   }

   @Test
   public void toMatrix() throws Exception {
      assertEquals($(1, 10), $(v.toMatrix().rows, v.toMatrix().columns));
   }

   @Test
   public void transpose() throws Exception {
      assertEquals($(10, 1), $(v.transpose().rows, v.transpose().columns));
   }

   @Test
   public void compress() throws Exception {
      v.compress();
   }

   @Test
   public void decrement() throws Exception {
      v.decrement(0);
      assertEquals(0, v.get(0), 0);
      v.decrement(1, 100);
      assertEquals(-99, v.get(1), 0);
   }


   @Test
   public void dimension() throws Exception {
      assertEquals(10, v.dimension(), 0);
   }

   @Test
   public void divide() throws Exception {
      assertEquals(5, v.divide(new DenseVector(10).mapSelf(d -> 2)).sum(), 0);
      assertEquals(10, v.divide(SparseVector.ones(10)).sum(), 0);
   }

   @Test(expected = IllegalArgumentException.class)
   public void divideError() throws Exception {
      assertEquals(10, v.divide(SparseVector.ones(1)).sum(), 0);
   }

   @Test
   public void divideSelf() throws Exception {
      assertEquals(5, v.divideSelf(new DenseVector(10).mapSelf(d -> 2)).sum(), 0);
      assertEquals(5, v.divideSelf(SparseVector.ones(10)).sum(), 0);
   }

   @Test(expected = IllegalArgumentException.class)
   public void divideSelfError() throws Exception {
      assertEquals(10, v.divideSelf(SparseVector.ones(1)).sum(), 0);
   }

   @Test
   public void dot() throws Exception {
      assertEquals(0.0, v.dot(DenseVector.zeros(10)), 0);
      assertEquals(10.0, v.dot(DenseVector.ones(10)), 0);
   }

   @Test(expected = IllegalArgumentException.class)
   public void dotError() throws Exception {
      assertEquals(0.0, v.dot(DenseVector.zeros(100)), 0);
   }

   @Test
   public void get() throws Exception {
      assertEquals(1, v.get(0), 0);
   }

   @Test
   public void increment() throws Exception {
      v.increment(0);
      assertEquals(2, v.get(0), 0);
      v.increment(1, 100);
      assertEquals(101, v.get(1), 0);
   }


   @Test
   public void isFinite() throws Exception {
      assertTrue(v.isFinite());
      v.set(0, Double.NaN);
      assertFalse(v.isFinite());
   }

   @Test
   public void isInfinite() throws Exception {
      assertFalse(v.isInfinite());
      v.set(0, Double.POSITIVE_INFINITY);
      assertTrue(v.isInfinite());
   }

   @Test
   public void isNaN() throws Exception {
      assertFalse(v.isNaN());
      v.set(0, Double.NaN);
      assertTrue(v.isNaN());
   }

   @Test
   public void l1Norm() throws Exception {
      assertEquals(10, v.l1Norm(), 0);
   }

   @Test
   public void lInfNorm() throws Exception {
      assertEquals(1, v.lInfNorm(), 0);
   }

   @Test
   public void magnitude() throws Exception {
      assertEquals(3.16, v.magnitude(), 0.01);
   }

   @Test
   public void map() throws Exception {
      assertEquals(0, v.map(d -> 0).sum(), 0);
      assertEquals(0, v.map(DenseVector.zeros(10), (d1, d2) -> d1 * d2).sum(), 0);
   }

   @Test(expected = IllegalArgumentException.class)
   public void mapError() throws Exception {
      assertEquals(0, v.map(DenseVector.zeros(5), (d1, d2) -> d1 * d2).sum(), 0);
   }

   @Test
   public void mapAdd() throws Exception {
      assertEquals(10, v.mapAdd(0).sum(), 0);
   }

   @Test
   public void mapAddSelf() throws Exception {
      assertEquals(10, v.mapAddSelf(0).sum(), 0);
   }

   @Test
   public void mapDivide() throws Exception {
      assertEquals(10, v.mapDivide(1).sum(), 0);
   }

   @Test
   public void mapDivideSelf() throws Exception {
      assertEquals(10, v.mapDivideSelf(1).sum(), 0);
   }

   @Test
   public void mapMultiply() throws Exception {
      assertEquals(10, v.mapMultiply(1).sum(), 0);
   }

   @Test
   public void mapMultiplySelf() throws Exception {
      assertEquals(10, v.mapMultiplySelf(1).sum(), 0);
   }

   @Test
   public void mapSelf() throws Exception {
      assertEquals(0, v.mapSelf(d -> 0).sum(), 0);
      assertEquals(0, v.mapSelf(DenseVector.zeros(10), (d1, d2) -> d1 * d2).sum(), 0);
   }

   @Test(expected = IllegalArgumentException.class)
   public void mapSelfError() throws Exception {
      assertEquals(0, v.mapSelf(DenseVector.zeros(5), (d1, d2) -> d1 * d2).sum(), 0);
   }

   @Test
   public void mapSubtract() throws Exception {
      assertEquals(0, v.mapSubtract(1).sum(), 0);
   }

   @Test
   public void mapSubtractSelf() throws Exception {
      assertEquals(0, v.mapSubtractSelf(1).sum(), 0);
   }

   @Test
   public void max() throws Exception {
      assertEquals(1, v.max(), 0);
   }

   @Test
   public void min() throws Exception {
      assertEquals(1, v.min(), 0);
   }

   @Test
   public void multiply() throws Exception {
      assertEquals(10, v.multiply(DenseVector.ones(10)).sum(), 0);
   }

   @Test
   public void multiplySelf() throws Exception {
      assertEquals(10, v.multiplySelf(DenseVector.ones(10)).sum(), 0);
   }

   @Test(expected = IllegalArgumentException.class)
   public void multiplyError() throws Exception {
      assertEquals(10, v.multiply(DenseVector.ones(1)).sum(), 0);
   }

   @Test(expected = IllegalArgumentException.class)
   public void multiplySelfError() throws Exception {
      assertEquals(10, v.multiplySelf(DenseVector.ones(1)).sum(), 0);
   }

   @Test
   public void iterator() throws Exception {
      double sum = 0;
      for (Vector.Entry aV : v) {
         sum += aV.value;
      }
      assertEquals(10, sum, 0);
   }

   @Test
   public void nonZeroIterator() throws Exception {
      double sum = 0;
      for (Iterator<Vector.Entry> itr = v.nonZeroIterator(); itr.hasNext(); ) {
         sum += itr.next().value;
      }
      assertEquals(10, sum, 0);
   }

   @Test
   public void orderedNonZeroIterator() throws Exception {
      int last = 0;
      for (Iterator<Vector.Entry> itr = v.orderedNonZeroIterator(); itr.hasNext(); ) {
         assertEquals(last, itr.next().getIndex());
         last++;
      }
   }

   @Test
   public void scale() throws Exception {
      v.scale(0, 100);
      assertEquals(100, v.get(0), 0);
   }

   @Test
   public void set() throws Exception {
      v.set(0, 100);
      assertEquals(100, v.get(0), 0);
   }

   @Test
   public void size() throws Exception {
      assertEquals(10, v.size(), 0);
   }

   @Test
   public void slice() throws Exception {
      Vector slice = v.slice(0, 5);
      assertEquals(5, slice.sum(), 0);
   }

   @Test
   public void statistics() throws Exception {
      assertEquals(1, v.statistics().getAverage(), 0);
   }

   @Test
   public void subtract() throws Exception {
      assertEquals(0, v.subtract(DenseVector.ones(10)).sum(), 0);
   }

   @Test(expected = IllegalArgumentException.class)
   public void subtractError() throws Exception {
      assertEquals(0, v.subtract(DenseVector.ones(1)).sum(), 0);
   }

   @Test
   public void subtractSelf() throws Exception {
      assertEquals(0, v.subtractSelf(DenseVector.ones(10)).sum(), 0);
   }

   @Test(expected = IllegalArgumentException.class)
   public void subtractSelfError() throws Exception {
      assertEquals(0, v.subtractSelf(DenseVector.ones(1)).sum(), 0);
   }

   @Test
   public void sum() throws Exception {
      assertEquals(10, v.sum(), 0);
   }

   @Test
   public void toArray() throws Exception {
      assertArrayEquals(new double[]{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, v.toArray(), 0);
   }

   @Test
   public void zero() throws Exception {
      v = v.zero();
      assertEquals(0, v.sum(), 0);
   }

   @Test
   public void redim() throws Exception {
      v = v.redim(20);
      assertEquals(20, v.dimension(), 0);
      assertEquals(10, v.sum(), 0);
   }

   @Test
   public void forEachSparse() throws Exception {
      double[] values = new double[10];
      v.forEachSparse(e -> values[e.index] = e.value);
      assertArrayEquals(new double[]{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, values, 0);
   }

   @Test
   public void forEachOrderedSparse() throws Exception {
      double[] values = new double[10];
      v.forEachSparse(e -> values[e.index] = e.value);
      assertArrayEquals(new double[]{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, values, 0);
   }

   @Test
   public void corr() throws Exception {
      Vector other = SparseVector.zeros(10);
      assertEquals(1, v.corr(other), 0);
   }

}