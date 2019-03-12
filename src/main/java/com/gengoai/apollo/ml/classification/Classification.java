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
 *
 */

package com.gengoai.apollo.ml.classification;

import com.gengoai.apollo.linear.p2.NDArray;
import com.gengoai.apollo.ml.vectorizer.DiscreteVectorizer;
import com.gengoai.collection.counter.Counter;
import com.gengoai.collection.counter.Counters;

import java.io.Serializable;

/**
 * Encapsulates the result of a classifier model applied to an instance.
 *
 * @author David B. Bracewell
 */
public class Classification implements Serializable {
   private static final long serialVersionUID = 1L;
   private final String argMax;
   private final NDArray distribution;
   private DiscreteVectorizer vectorizer;

   /**
    * Instantiates a new Classification with a vectorizer to facilitate label id to label mapping.
    *
    * @param distribution the distribution
    * @param vectorizer   the vectorizer
    */
   public Classification(NDArray distribution, DiscreteVectorizer vectorizer) {
      this.distribution = distribution.shape().isColumnVector() ? distribution.T() : distribution.copy();
      this.argMax = vectorizer.getString(this.distribution.argmax());
      this.vectorizer = vectorizer;
   }


   /**
    * Gets the classification object as a Counter. Will convert to label ids to names if a vectorizer is present,
    * otherwise will use string representation of label ids.
    *
    * @return the counter
    */
   public Counter<String> asCounter() {
      Counter<String> counter = Counters.newCounter();
      for (long i = 0; i < distribution.length(); i++) {
         counter.set(vectorizer.getString(i), distribution.get((int) i));
      }
      return counter;
   }

   /**
    * Gets the underlying distribution of scores.
    *
    * @return the NDArray representing the distribution.
    */
   public NDArray distribution() {
      return distribution;
   }


   /**
    * Gets the score for a label.
    *
    * @param label the label
    * @return the score
    */
   public double getScore(String label) {
      return distribution.get(vectorizer.indexOf(label));
   }

   /**
    * Gets the argMax as a string either converting the id using the supplied vectorizer or using
    * <code>Integer.toString</code>
    *
    * @return the result
    */
   public String getResult() {
      return argMax;
   }

   @Override
   public String toString() {
      return "Classification" + distribution;

   }
}//END OF Classification
