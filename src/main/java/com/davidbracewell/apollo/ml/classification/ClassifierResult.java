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

package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.collection.Counter;
import com.davidbracewell.collection.Counters;
import com.davidbracewell.string.StringUtils;
import lombok.EqualsAndHashCode;

import java.io.Serializable;
import java.util.Map;
import java.util.Set;

/**
 * The type Classifier result.
 *
 * @author David B. Bracewell
 */
@EqualsAndHashCode
public class ClassifierResult implements Serializable {
  private static final long serialVersionUID = 1L;
  private final Counter<String> distribution;
  private final String result;

  /**
   * Instantiates a new Classifier result.
   *
   * @param distribution the distribution
   */
  public ClassifierResult(Counter<String> distribution) {
    this.distribution = distribution;
    this.result = distribution.max() == null ? StringUtils.EMPTY : distribution.max();

  }

  /**
   * Distribution set.
   *
   * @return the set
   */
  public Set<Map.Entry<String, Double>> distribution() {
    return distribution.entries();
  }

  /**
   * As counter counter.
   *
   * @return the counter
   */
  public Counter<String> asCounter() {
    return Counters.newHashMapCounter(distribution);
  }

  /**
   * Result string.
   *
   * @return the string
   */
  public String getResult() {
    return result;
  }


  @Override
  public String toString() {
    return distribution.toString();
  }


  /**
   * Gets confidence.
   *
   * @return the confidence
   */
  public double getConfidence() {
    return distribution.get(result);
  }

  /**
   * Gets confidence.
   *
   * @param label the label
   * @return the confidence
   */
  public double getConfidence(String label) {
    return distribution.get(label);
  }

  /**
   * Gets labels.
   *
   * @return the labels
   */
  public Set<String> getLabels() {
    return distribution.items();
  }


}//END OF ClassifierResult
