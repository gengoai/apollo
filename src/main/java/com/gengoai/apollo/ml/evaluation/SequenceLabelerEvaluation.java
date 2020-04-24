/*
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

package com.gengoai.apollo.ml.evaluation;

import com.gengoai.apollo.ml.DataSet;
import com.gengoai.apollo.ml.Datum;
import com.gengoai.apollo.ml.model.Model;
import com.gengoai.stream.MStream;
import lombok.NonNull;

import java.io.PrintStream;

/**
 * @author David B. Bracewell
 */
public interface SequenceLabelerEvaluation {
   /**
    * Evaluate the given model using the given dataset
    *
    * @param model   the model to evaluate
    * @param dataset the dataset to evaluate over
    */
   default void evaluate(@NonNull Model model, @NonNull DataSet dataset) {
      evaluate(model, dataset.stream());
   }

   /**
    * Evaluate the given model using the given set of examples
    *
    * @param model   the model to evaluate
    * @param dataset the dataset to evaluate over
    */
   void evaluate(@NonNull Model model, @NonNull MStream<Datum> dataset);

   /**
    * Merge this evaluation with another combining the results.
    *
    * @param evaluation the other evaluation to combine
    */
   void merge(@NonNull SequenceLabelerEvaluation evaluation);

   /**
    * Outputs the results of the classification to the given <code>PrintStream</code>
    *
    * @param printStream          the print stream to write to
    * @param printConfusionMatrix True print the confusion matrix, False do not print the confusion matrix.
    */
   void output(@NonNull PrintStream printStream, boolean printConfusionMatrix);

   /**
    * Outputs the evaluation results to standard out.
    *
    * @param printConfusionMatrix True print the confusion matrix, False do not print the confusion matrix.
    */
   default void output(boolean printConfusionMatrix) {
      output(System.out, printConfusionMatrix);
   }

   /**
    * Outputs the evaluation results to standard out.
    */
   default void output() {
      output(System.out, false);
   }

}//END OF SequenceLabelerEvaluation
