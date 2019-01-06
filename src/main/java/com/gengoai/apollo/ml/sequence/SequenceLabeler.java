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
import com.gengoai.apollo.ml.ModelParameters;
import com.gengoai.apollo.ml.vectorizer.Vectorizer;
import com.gengoai.conversion.Cast;

/**
 * <p>Labels each example in a sequence of examples, which may represent points in time, tokens in a sentence, etc.
 * </p>
 *
 * @author David B. Bracewell
 */
public abstract class SequenceLabeler extends Model<SequenceLabeler> {
   private static final long serialVersionUID = 1L;
   private SequenceValidator sequenceValidator;

   public SequenceLabeler(ModelParameters modelParameters) {
      super(modelParameters);
      this.sequenceValidator = modelParameters.sequenceValidator;
   }

   @Override
   @SuppressWarnings("unchecked")
   public Vectorizer<String> getLabelVectorizer() {
      return Cast.as(super.getLabelVectorizer());
   }

   public SequenceValidator getSequenceValidator() {
      return sequenceValidator;
   }

   public void setSequenceValidator(SequenceValidator sequenceValidator) {
      this.sequenceValidator = sequenceValidator;
   }

   protected boolean isValidTransition(String current, String previous, Example example) {
      return sequenceValidator.isValid(current, previous, example);
   }

   protected boolean isValidTransition(int current, int previous, Example example) {
      return sequenceValidator.isValid(getLabelVectorizer().decode(current),
                                       getLabelVectorizer().decode(previous),
                                       example);
   }

   /**
    * Specialized transform to predict the labels for a sequence.
    *
    * @param example the example sequence to label
    */
   public abstract Labeling label(Example example);

}//END OF SequenceLabeler
