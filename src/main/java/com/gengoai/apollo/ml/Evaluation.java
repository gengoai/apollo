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

package com.gengoai.apollo.ml;

import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.io.resource.Resource;
import com.gengoai.stream.MStream;

import java.io.IOException;
import java.io.PrintStream;

/**
 * <p>Generic interface for evaluating models.</p>
 *
 * @author David B. Bracewell
 */
public interface Evaluation<M extends Model> {

   /**
    * Evaluate the given model using the given dataset
    *
    * @param model   the model to evaluate
    * @param dataset the dataset to evaluate over
    */
   default void evaluate(M model, Dataset dataset) {
      evaluate(model, dataset.stream());
   }

   /**
    * Evaluate the given model using the given set of examples
    *
    * @param model   the model to evaluate
    * @param dataset the dataset to evaluate over
    */
   void evaluate(M model, MStream<Example> dataset);

   /**
    * Merge this evaluation with another combining the results.
    *
    * @param evaluation the other evaluation to combine
    */
   void merge(Evaluation evaluation);

   /**
    * Outputs the evaluation results to the given print stream.
    *
    * @param printStream the print stream to write to
    */
   void output(PrintStream printStream);

   /**
    * Outputs the evaluation results to the given resource.
    *
    * @param resource the resource to write to
    * @throws IOException the io exception
    */
   default void output(Resource resource) throws IOException {
      try (PrintStream ps = new PrintStream(resource.outputStream())) {
         output(ps);
      }
   }

   /**
    * Outputs the evaluation results to standard out.
    */
   default void output() {
      output(System.out);
   }

}//END OF Evaluation
