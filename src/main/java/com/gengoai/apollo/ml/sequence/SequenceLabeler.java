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

import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.Model;

/**
 * <p>Labels each example in a sequence of examples, which may represent points in time, tokens in a sentence, etc.
 * </p>
 *
 * @author David B. Bracewell
 */
public abstract class SequenceLabeler extends Model {
   private static final long serialVersionUID = 1L;
   private final SequencePipeline modelParameters;

   /**
    * Instantiates a new Sequence labeler.
    *
    * @param modelParameters the model parameters
    */
   public SequenceLabeler(SequencePipeline modelParameters) {
      this.modelParameters = modelParameters.copy();
   }


   /**
    * Gets sequence validator.
    *
    * @return the sequence validator
    */
   public SequenceValidator getSequenceValidator() {
      return modelParameters.sequenceValidator;
   }

   /**
    * Sets sequence validator.
    *
    * @param sequenceValidator the sequence validator
    */
   public void setSequenceValidator(SequenceValidator sequenceValidator) {
      modelParameters.sequenceValidator = sequenceValidator;
   }

   /**
    * Is valid transition boolean.
    *
    * @param current  the current
    * @param previous the previous
    * @param example  the example
    * @return the boolean
    */
   protected boolean isValidTransition(String current, String previous, Example example) {
      return getSequenceValidator().isValid(current, previous, example);
   }

   /**
    * Is valid transition boolean.
    *
    * @param current  the current
    * @param previous the previous
    * @param example  the example
    * @return the boolean
    */
   protected boolean isValidTransition(int current, int previous, Example example) {
      return getSequenceValidator().isValid(modelParameters.labelVectorizer.getString(current),
                                            modelParameters.labelVectorizer.getString(previous),
                                            example);
   }


   @Override
   public int getNumberOfLabels() {
      return modelParameters.labelVectorizer.size();
   }

   /**
    * Specialized transform to predict the labels for a sequence.
    *
    * @param example the example sequence to label
    * @return the labeling
    */
   public abstract Labeling label(Example example);


   @Override
   public SequencePipeline getPipeline() {
      return modelParameters;
   }
}//END OF SequenceLabeler
