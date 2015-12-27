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

import com.google.common.base.Preconditions;
import lombok.NonNull;
import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

/**
 * The type Multinomial.
 *
 * @author David B. Bracewell
 */
public class Multinomial implements DiscreteDistribution<Multinomial>, Serializable {
  private static final long serialVersionUID = 1L;
  private final long[] values;
  private final double alpha;
  private final double alphaTimesV;
  private final Random random;
  private long sum = 0;

  /**
   * Instantiates a new Multinomial.
   *
   * @param size   the size
   * @param alpha  the alpha
   * @param random the random
   */
  public Multinomial(int size, double alpha, @NonNull Random random) {
    Preconditions.checkArgument(size > 0, "Size must be > 0");
    Preconditions.checkArgument(alpha >= 0, "Alpha must be >= 0");
    this.values = new long[size];
    this.alpha = alpha;
    this.alphaTimesV = alpha * size;
    this.random = random;
  }

  /**
   * Instantiates a new Multinomial.
   *
   * @param size the size
   */
  public Multinomial(int size) {
    this(size, 0, new Random());
  }

  /**
   * Instantiates a new Multinomial.
   *
   * @param size  the size
   * @param alpha the alpha
   */
  public Multinomial(int size, double alpha) {
    this(size, alpha, new Random());
  }


  @Override
  public double unnormalizedProbability(int index) {
    if (index < 0 || index >= values.length) {
      return 0.0;
    }
    return (values[index] + alpha);
  }

  @Override
  public Multinomial increment(int index, long amount) {
    this.values[index] += amount;
    sum += amount;
    return this;
  }

  @Override
  public double probability(int index) {
    if (index < 0 || index >= values.length) {
      return 0.0;
    }
    return (values[index] + alpha) / (sum + alphaTimesV);
  }

  @Override
  public int sample() {
    double rnd = random.nextDouble() * sum;
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

  @Override
  public double cumulativeProbability(int x) {
    return new EnumeratedIntegerDistribution(
      IntStream.range(0, values.length).toArray(),
      IntStream.range(0, values.length).mapToDouble(this::probability).toArray()
    ).cumulativeProbability(x);
  }

  @Override
  public double cumulativeProbability(int lowerBound, int higherBound) {
    return new EnumeratedIntegerDistribution(
      IntStream.range(0, values.length).toArray(),
      IntStream.range(0, values.length).mapToDouble(this::probability).toArray()
    ).cumulativeProbability(lowerBound, higherBound);
  }

}//END OF Multinomial
