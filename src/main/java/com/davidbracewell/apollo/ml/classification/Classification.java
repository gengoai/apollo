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
import com.davidbracewell.collection.counter.Counter;
import com.davidbracewell.collection.counter.Counters;
import com.davidbracewell.conversion.Cast;
import lombok.EqualsAndHashCode;
import lombok.NonNull;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

/**
 * Encapsulates the result of a classifier model applied to an instance.
 *
 * @author David B. Bracewell
 */
@EqualsAndHashCode
public class Classification implements Serializable {
   private static final long serialVersionUID = 1L;
   private final double[] distribution;
   private final int resultIndex;
   private final Encoder labelEncoder;

   /**
    * Instantiates a new Classifier result.
    *
    * @param distribution the distribution across the labels
    * @param labelEncoder the encoder for converting indexes into labels
    */
   public Classification(@NonNull double[] distribution, @NonNull Encoder labelEncoder) {
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
    * Returns the distribution across the possible labels
    *
    * @return a double array of scores, or probabilities, for each of the labels
    */
   public double[] distribution() {
      return distribution;
   }

   /**
    * Gets the String value of the label with best score
    *
    * @return the label with the best score
    */
   public String getResult() {
      return labelEncoder.decode(resultIndex).toString();
   }

   /**
    * Determines if the result of this classification equals the given gold value
    *
    * @param gold the gold value
    * @return True if the gold value is not null and equals the best result of this classification, False otherwise
    */
   public boolean resultIs(Object gold) {
      return gold != null && getResult().equals(gold.toString());
   }

   @Override
   public String toString() {
      return Arrays.toString(distribution);
   }


   /**
    * Gets the confidence of the best result
    *
    * @return the confidence of the best result, i.e. predicted value
    */
   public double getConfidence() {
      return distribution[resultIndex];
   }

   /**
    * Gets the confidence of the given label.
    *
    * @param label the label whose confidence we want
    * @return the confidence of the label or <code>Double.NEGATIVE_INFINITY</code> if the label is invalid
    */
   public double getConfidence(String label) {
      int index = labelEncoder.index(label);
      if (index == -1) {
         return Double.NEGATIVE_INFINITY;
      }
      return distribution[index];
   }

   /**
    * Gets the label for the given label index.
    *
    * @param index the index of the label
    * @return the label
    */
   public String getLabel(int index) {
      Object lbl = labelEncoder.decode(index);
      return lbl == null ? null : lbl.toString();
   }


   /**
    * Gets the result of the classification as an encoded label
    *
    * @return the encoded result
    */
   public double getEncodedResult() {
      return resultIndex;
   }

   /**
    * Gets all labels.
    *
    * @return all labels in the classification
    */
   public List<String> getLabels() {
      return Cast.cast(labelEncoder.values());
   }


   /**
    * Gets the classification result as a counter
    *
    * @return the counter of item (label) and value (confidence)
    */
   public Counter<String> asCounter() {
      Counter<String> counter = Counters.newCounter();
      for (int ci = 0; ci < distribution.length; ci++) {
         counter.set(labelEncoder.decode(ci).toString(), distribution[ci]);
      }
      return counter;
   }


}//END OF Classification
