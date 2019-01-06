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

import com.gengoai.Validation;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.classification.MultiClassEvaluation;
import com.gengoai.conversion.Cast;
import com.gengoai.stream.MStream;

import java.io.PrintStream;
import java.io.Serializable;

/**
 * <p>Sequence labeling evaluation that evaluates each item in the sequence (i.e. instance) independently.</p>
 *
 * @author David B. Bracewell
 */
public class PerInstanceEvaluation implements SequenceLabelerEvaluation, Serializable {
   private static final long serialVersionUID = 1L;
   private final MultiClassEvaluation eval;


   /**
    * Instantiates a new Per instance evaluation.
    */
   public PerInstanceEvaluation() {
      this.eval = new MultiClassEvaluation();
   }

   @Override
   public void evaluate(SequenceLabeler model, MStream<Example> dataset) {
      dataset.forEach(sequence -> {
         Labeling result = model.label(sequence);
         for (int i = 0; i < sequence.size(); i++) {
            eval.entry(sequence.getExample(i).getLabelAsString(), result.getLabel(i));
         }
      });
   }

   @Override
   public void merge(SequenceLabelerEvaluation evaluation) {
      Validation.checkArgument(evaluation instanceof PerInstanceEvaluation);
      eval.merge(Cast.<PerInstanceEvaluation>as(evaluation).eval);
   }

   @Override
   public void output(PrintStream printStream, boolean printConfusionMatrix) {
      eval.output(printStream, printConfusionMatrix);
   }

}//END OF PerInstanceEvaluation
