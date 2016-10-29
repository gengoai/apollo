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

package com.davidbracewell.apollo.linalg.store;

import com.davidbracewell.apollo.affinity.Measure;
import com.davidbracewell.apollo.affinity.Similarity;
import com.davidbracewell.apollo.linalg.Vector;

import java.util.Arrays;
import java.util.Random;

/**
 * @author David B. Bracewell
 */
public class MinHashSignature implements SignatureFunction {
  private static final long serialVersionUID = 1L;
  private final int signatureSize;
  private final int dimension;
  private final long[][] coefficents;

  public MinHashSignature(double error, int dimension) {
    this((int) (1 / (error * error)), dimension);
  }

  public MinHashSignature(int signatureSize, int dimension) {
    this.signatureSize = signatureSize;
    this.dimension = dimension;
    Random rnd = new Random();
    this.coefficents = new long[signatureSize][2];
    for (int i = 0; i < signatureSize; i++) {
      this.coefficents[i][0] = rnd.nextInt(dimension);
      this.coefficents[i][1] = rnd.nextInt(dimension);
    }
  }

  @Override
  public int[] signature(Vector vector) {
    int[] sig = new int[signatureSize];
    Arrays.fill(sig, Integer.MAX_VALUE);
    vector.nonZeroIterator().forEachRemaining(entry -> {
      if (entry.getValue() > 0) {
        for (int i = 0; i < signatureSize; i++) {
          sig[i] = Math.min(sig[i], h(i, entry.getIndex()));
        }
      }
    });
    return sig;
  }

  private int h(int i, long x) {
    return (int) (coefficents[i][0] * x + coefficents[i][1]) % dimension;
  }


  @Override
  public boolean isBinary() {
    return false;
  }

  @Override
  public int getDimension() {
    return dimension;
  }

  @Override
  public int getSignatureSize() {
    return signatureSize;
  }

  @Override
  public Measure getMeasure() {
    return Similarity.Jaccard;
  }
}// END OF MinHashSignature
