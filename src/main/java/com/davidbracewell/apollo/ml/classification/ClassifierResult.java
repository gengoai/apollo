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

import com.davidbracewell.apollo.ml.Encoder;
import com.davidbracewell.conversion.Cast;
import lombok.EqualsAndHashCode;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

/**
 * The type Classifier result.
 *
 * @author David B. Bracewell
 */
@EqualsAndHashCode
public class ClassifierResult implements Serializable {
  private static final long serialVersionUID = 1L;
  private final double[] distribution;
  private final int resultIndex;
  private final Encoder labelEncoder;

  /**
   * Instantiates a new Classifier result.
   *
   * @param distribution the distribution
   */
  public ClassifierResult(double[] distribution, Encoder labelEncoder) {
    this.distribution = distribution;
    double max = distribution[0];
    int maxI = 0;
    for (int i = 1; i < distribution.length; i++) {
      if (distribution[i] > max) {
        max = distribution[i];
        maxI = i;
      }
    }
    this.resultIndex = maxI;
    this.labelEncoder = labelEncoder;
  }

  /**
   * Distribution set.
   *
   * @return the set
   */
  public double[] distribution() {
    return distribution;
  }

  /**
   * Result string.
   *
   * @return the string
   */
  public String getResult() {
    return labelEncoder.decode(resultIndex).toString();
  }

  public boolean resultIs(Object gold) {
    if (gold == null) {
      return false;
    }
    return getResult().equals(gold.toString());
  }


  @Override
  public String toString() {
    return Arrays.toString(distribution);
  }


  /**
   * Gets confidence.
   *
   * @return the confidence
   */
  public double getConfidence() {
    return distribution[resultIndex];
  }

  /**
   * Gets confidence.
   *
   * @param label the label
   * @return the confidence
   */
  public double getConfidence(String label) {
    return distribution[(int) labelEncoder.encode(label)];
  }

  public String getLabel(int index) {
    return labelEncoder.decode(index).toString();
  }


  public int getEncodedResult() {
    return resultIndex;
  }

  /**
   * Gets labels.
   *
   * @return the labels
   */
  public List<String> getLabels() {
    return Cast.cast(labelEncoder.values());
  }


}//END OF ClassifierResult
