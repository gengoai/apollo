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

import java.util.Arrays;
import java.util.Iterator;

import static org.junit.Assert.*;

/**
 * @author David B. Bracewell
 */
public abstract class AbstractMatrixTest {

   Matrix m;

   @Test
   public void toDense() throws Exception {
      DenseMatrix dense = m.toDense();
      assertNotNull(dense);
      assertEquals(m, dense);
   }

   @Test
   public void diag() throws Exception {
      Matrix diag = m.diag();
      assertEquals(2, diag.numberOfColumns());
      assertEquals(2, diag.numberOfRows());
      assertEquals(1, diag.get(0, 0), 0);
      assertEquals(1, diag.get(1, 1), 0);
      assertEquals(0, diag.get(0, 1), 0);
      assertEquals(0, diag.get(1, 0), 0);
   }

   @Test
   public void column() throws Exception {
      assertEquals(DenseVector.wrap(1, 1), m.column(0));
      m.column(1).set(0, 0);
      assertEquals(DenseVector.wrap(0, 1), m.column(1));
   }

   @Test
   public void row() throws Exception {
      assertEquals(DenseVector.wrap(1, 1), m.row(0));
      m.row(1).set(0, 0);
      assertEquals(DenseVector.wrap(0, 1), m.row(1));
   }

   @Test
   public void setColumn() throws Exception {
      m.setColumn(0, DenseVector.wrap(2, 2));
      assertEquals(DenseVector.wrap(2, 2), m.column(0));
   }

   @Test(expected = IllegalArgumentException.class)
   public void setColumnError() throws Exception {
      m.setColumn(0, DenseVector.wrap(2, 2, 2));
      assertEquals(DenseVector.wrap(2, 2), m.column(0));
   }

   @Test
   public void setRow() throws Exception {
      m.setRow(0, DenseVector.wrap(2, 2));
      assertEquals(DenseVector.wrap(2, 2), m.row(0));
   }

   @Test(expected = IllegalArgumentException.class)
   public void setRowError() throws Exception {
      m.setRow(0, DenseVector.wrap(2, 2, 3));
      assertEquals(DenseVector.wrap(2, 2), m.row(0));
   }

   @Test
   public void addSelf() throws Exception {
      assertEquals(new DenseMatrix(new double[][]{
                      new double[]{2, 2},
                      new double[]{2, 2}
                   }),
                   m.addSelf(DenseMatrix.ones(2, 2)));
   }

   @Test(expected = IllegalArgumentException.class)
   public void addSelfError() throws Exception {
      assertEquals(DenseMatrix.ones(4, 4), m.addSelf(DenseMatrix.ones(4, 2)));
   }

   @Test
   public void subtractSelf() throws Exception {
      assertEquals(SparseMatrix.zeroes(2, 2), m.subtractSelf(DenseMatrix.ones(2, 2)));
   }

   @Test(expected = IllegalArgumentException.class)
   public void subtractSelfError() throws Exception {
      assertEquals(SparseMatrix.zeroes(4, 2), m.subtractSelf(DenseMatrix.ones(4, 2)));
   }

   @Test
   public void scaleSelf() throws Exception {
      assertEquals(new DenseMatrix(new double[][]{
                      new double[]{3, 3},
                      new double[]{3, 3}
                   }),
                   m.scaleSelf(3));
      assertEquals(new DenseMatrix(new double[][]{
                      new double[]{3, 3},
                      new double[]{3, 3}
                   }),
                   m.scaleSelf(SparseMatrix.ones(2, 2)));
      assertEquals(new DenseMatrix(new double[][]{
                      new double[]{3, 3},
                      new double[]{3, 3}
                   }),
                   m.scaleSelf(DenseMatrix.ones(2, 2)));
   }

   @Test(expected = IllegalArgumentException.class)
   public void scaleSelfError() throws Exception {
      assertEquals(new DenseMatrix(new double[][]{
                      new double[]{1, 1},
                      new double[]{1, 1}
                   }),
                   m.scaleSelf(DenseMatrix.ones(2, 4)));
   }

   @Test
   public void incrementSelf() throws Exception {
      assertEquals(new DenseMatrix(new double[][]{
                      new double[]{2, 2},
                      new double[]{2, 2}
                   }),
                   m.incrementSelf(1));
   }

   @Test
   public void increment() throws Exception {
      assertEquals(new DenseMatrix(new double[][]{
                      new double[]{2, 2},
                      new double[]{2, 2}
                   }),
                   m.increment(1));
   }

   @Test
   public void columnIterator() throws Exception {
      int n = 0;
      double sum = 0;
      for (Iterator<Vector> v = m.columnIterator(); v.hasNext(); ) {
         n++;
         sum += v.next().sum();
      }
      assertEquals(2, n);
      assertEquals(4, sum, 0);
   }

   @Test
   public void rowIterator() throws Exception {
      int n = 0;
      double sum = 0;
      for (Iterator<Vector> v = m.rowIterator(); v.hasNext(); ) {
         n++;
         sum += v.next().sum();
      }
      assertEquals(2, n);
      assertEquals(4, sum, 0);
   }

   @Test
   public void multiply() throws Exception {
      assertEquals(new DenseMatrix(new double[][]{
                      new double[]{1, 1},
                      new double[]{1, 1}
                   }),
                   m.scaleSelf(DenseMatrix.ones(2, 2)));
   }

   @Test(expected = IllegalArgumentException.class)
   public void multiplyError() throws Exception {
      assertEquals(new DenseMatrix(new double[][]{
                      new double[]{1, 1},
                      new double[]{1, 1}
                   }),
                   m.scaleSelf(DenseMatrix.ones(4, 2)));
   }

   @Test
   public void transpose() throws Exception {
      assertEquals(m, m.transpose());
   }

   @Test
   public void toArray() throws Exception {
      double[][] d = m.toArray();
      assertTrue(Arrays.equals(new double[]{1, 1}, d[0]));
      assertTrue(Arrays.equals(new double[]{1, 1}, d[1]));
   }

   @Test
   public void scale() throws Exception {
      assertEquals(new DenseMatrix(new double[][]{
                      new double[]{3, 3},
                      new double[]{3, 3}
                   }),
                   m.scale(3));
   }

   @Test
   public void iterator() throws Exception {
      double sum = 0;
      for (Iterator<Matrix.Entry> itr = m.iterator(); itr.hasNext(); ) {
         sum += itr.next().value;
      }
      assertEquals(4, sum, 0);
   }


   @Test
   public void nonZero() throws Exception {
      double sum = 0;
      for (Iterator<Matrix.Entry> itr = m.nonZeroIterator(); itr.hasNext(); ) {
         sum += itr.next().value;
      }
      assertEquals(4, sum, 0);
   }

   @Test
   public void orderedSparse() throws Exception {
      double sum = 0;
      int lastC = 0;
      int lastR = 0;
      for (Iterator<Matrix.Entry> itr = m.orderedNonZeroIterator(); itr.hasNext(); ) {
         Matrix.Entry e = itr.next();
         sum += e.value;
         assertEquals(lastC, e.column);
         assertEquals(lastR, e.row);

         if (e.column == 0) {
            lastC = 1;
         } else if (e.column == 1) {
            lastC = 0;
            lastR = 1;
         }
      }
      assertEquals(4, sum, 0);
   }


   @Test
   public void dot() throws Exception {
      assertEquals(DenseVector.wrap(2, 2), m.dot(DenseVector.ones(2)));
   }

   @Test(expected = IllegalArgumentException.class)
   public void dotError() throws Exception {
      assertEquals(DenseVector.wrap(2, 2), m.dot(DenseVector.ones(4)));
   }

   @Test
   public void valueChange() throws Exception {
      m.increment(1, 1);
      m.increment(0, 1, 1);
      m.decrement(1, 1);
      m.decrement(0, 1, 1);
      assertEquals(new DenseMatrix(new double[][]{
                      new double[]{1, 1},
                      new double[]{1, 1}
                   }),
                   m);
   }


   @Test
   public void sum() throws Exception {
      assertEquals(4, m.sum(), 0);
   }

   @Test
   public void diagVector() throws Exception {
      assertEquals(DenseVector.ones(2), m.diagVector());
   }
}