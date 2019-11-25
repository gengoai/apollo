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

import com.gengoai.apollo.ml.Pipeline;
import com.gengoai.apollo.ml.vectorizer.DiscreteVectorizer;
import com.gengoai.apollo.ml.vectorizer.MultiLabelBinarizer;
import com.gengoai.conversion.Cast;

/**
 * The type Sequence pipeline.
 *
 * @author David B. Bracewell
 */
public class SequencePipeline extends Pipeline<DiscreteVectorizer, SequencePipeline> {
   private static final long serialVersionUID = 1L;
   /**
    * The Sequence validator.
    */
   public SequenceValidator sequenceValidator = SequenceValidator.ALWAYS_TRUE;


   /**
    * Instantiates a new Sequence pipeline.
    *
    * @param labelVectorizer the label vectorizer
    */
   protected SequencePipeline(DiscreteVectorizer labelVectorizer) {
      super(labelVectorizer);
   }

   @Override
   public SequencePipeline copy() {
      return Cast.as(super.copy());
   }


   /**
    * Create sequence pipeline.
    *
    * @return the sequence pipeline
    */
   public static SequencePipeline create() {
      return new SequencePipeline(new MultiLabelBinarizer());
   }


   /**
    * Create sequence pipeline.
    *
    * @param labelVectorizer the label vectorizer
    * @return the sequence pipeline
    */
   public static SequencePipeline create(DiscreteVectorizer labelVectorizer) {
      return new SequencePipeline(labelVectorizer);
   }


}//END OF SequenceModelParameters
