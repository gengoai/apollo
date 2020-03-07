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

package com.gengoai.apollo.ml.sequence;

import com.gengoai.jcrfsuite.util.Pair;

import java.io.Serializable;
import java.util.List;

/**
 * <p>Represents the predicted labels and their scores for a sequence.</p>
 *
 * @author David B. Bracewell
 */
public class Labeling implements Serializable {
   /**
    * The Labels.
    */
   public final String[] labels;
   /**
    * The Scores.
    */
   public final double[] scores;


   /**
    * Instantiates a new Labeling using the output of CRFSuite.
    *
    * @param crfSuiteTags the crf suite tags
    */
   public Labeling(List<Pair<String, Double>> crfSuiteTags) {
      this.labels = new String[crfSuiteTags.size()];
      this.scores = new double[crfSuiteTags.size()];
      for (int i = 0; i < crfSuiteTags.size(); i++) {
         this.labels[i] = crfSuiteTags.get(i).first;
         this.scores[i] = crfSuiteTags.get(i).second;
      }
   }


   /**
    * Instantiates a new Labeling with a given set of labels and scores.
    *
    * @param labels the labels
    * @param scores the scores
    */
   public Labeling(String[] labels, double[] scores) {
      this.labels = labels;
      this.scores = scores;
   }


   /**
    * Instantiates a new Labeling of a given length
    *
    * @param sequenceLength the sequence length
    */
   public Labeling(int sequenceLength) {
      this.labels = new String[sequenceLength];
      this.scores = new double[sequenceLength];
   }


   /**
    * Gets the label for the item in the sequence at the given index.
    *
    * @param index the index of the item in the sequence
    * @return the label
    */
   public String getLabel(int index) {
      return labels[index];
   }

   /**
    * Gets the score for the item in the sequence at the given index.
    *
    * @param index the index of the item in the sequence
    * @return the score
    */
   public double getScore(int index) {
      return scores[index];
   }


}//END OF Labeling
