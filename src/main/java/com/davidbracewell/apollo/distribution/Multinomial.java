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

package com.davidbracewell.apollo.distribution;

import java.io.Serializable;
import java.util.Arrays;

/**
 * @author David B. Bracewell
 */
public class Multinomial implements Distribution, Serializable {
  private static final long serialVersionUID = 1L;
  private final long[] values;
  private final double alpha;
  private final double alphaTimesV;
  private long sum = 0;

  public Multinomial(int size) {
    this(size, 0);
  }

  public Multinomial(int size, double alpha) {
    this.values = new long[size];
    this.alpha = alpha;
    this.alphaTimesV = alpha * size;
  }

  public Multinomial increment(int index) {
    return increment(index, 1);
  }

  public Multinomial decrement(int index) {
    return increment(index, -1);
  }

  public Multinomial decrement(int index, long amount) {
    return increment(index, -amount);
  }

  public Multinomial increment(int index, long amount) {
    this.values[index] += amount;
    sum += amount;
    return this;
  }

  @Override
  public double p(double value) {
    int index = (int) value;
    return (values[index] + alpha) / (sum + alphaTimesV);
  }

  @Override
  public double sample() {
    double rnd = Math.random() * sum;
    double sum = 0;
    for (int i = 0; i < values.length; i++) {
      double p = values[i];
      if (rnd < (sum + p)) {
        return i;
      }
      sum += p;
    }
    return values.length - 1;
  }

  @Override
  public String toString() {
    return Arrays.toString(values);
  }

}//END OF Multinomial
