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
      assertEquals("test", v.withLabel("test").getLabel());
   }

   @Test
   public void toMatrix() throws Exception {
      assertEquals($(1, 10), v.toMatrix().shape());
   }

   @Test
   public void transpose() throws Exception {
      assertEquals($(10, 1), v.transpose().shape());
   }

   @Test
   public void compress() throws Exception {

   }

   @Test
   public void decrement() throws Exception {

   }


   @Test
   public void dimension() throws Exception {
      assertEquals(10, v.dimension(), 0);
   }

   @Test
   public void divide() throws Exception {

   }

   @Test
   public void divideSelf() throws Exception {

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

   }

   @Test
   public void mapAddSelf() throws Exception {

   }

   @Test
   public void mapDivide() throws Exception {

   }

   @Test
   public void mapDivideSelf() throws Exception {

   }

   @Test
   public void mapMultiply() throws Exception {

   }

   @Test
   public void mapMultiplySelf() throws Exception {

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

   }

   @Test
   public void mapSubtractSelf() throws Exception {

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

   }

   @Test
   public void multiplySelf() throws Exception {

   }

   @Test
   public void iterator() throws Exception {

   }

   @Test
   public void nonZeroIterator() throws Exception {

   }

   @Test
   public void orderedNonZeroIterator() throws Exception {

   }

   @Test
   public void scale() throws Exception {

   }

   @Test
   public void set() throws Exception {

   }

   @Test
   public void size() throws Exception {

   }

   @Test
   public void slice() throws Exception {

   }

   @Test
   public void statistics() throws Exception {

   }

   @Test
   public void subtract() throws Exception {

   }

   @Test
   public void subtractSelf() throws Exception {

   }

   @Test
   public void sum() throws Exception {

   }

   @Test
   public void toArray() throws Exception {

   }

   @Test
   public void zero() throws Exception {

   }

   @Test
   public void redim() throws Exception {

   }

   @Test
   public void forEachSparse() throws Exception {

   }

   @Test
   public void forEachOrderedSparse() throws Exception {

   }

   @Test
   public void corr() throws Exception {

   }

}