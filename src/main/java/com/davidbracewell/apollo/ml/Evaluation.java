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

package com.davidbracewell.apollo.ml;

import com.davidbracewell.apollo.ml.data.Dataset;

import java.io.PrintStream;
import java.util.Collection;

/**
 * <p>Generic interface for evaluating models.</p>
 *
 * @param <T> the type of example
 * @param <M> the type of model
 * @author David B. Bracewell
 */
public interface Evaluation<T extends Example, M extends Model> {

   /**
    * Evaluate the given model using the given dataset
    *
    * @param model   the model to evaluate
    * @param dataset the dataset to evaluate over
    */
   void evaluate(M model, Dataset<T> dataset);

   /**
    * Evaluate the given model using the given set of examples
    *
    * @param model   the model to evaluate
    * @param dataset the dataset to evaluate over
    */
   void evaluate(M model, Collection<T> dataset);

   /**
    * Merge this evaluation with another combining the results.
    *
    * @param evaluation the other evaluation to combine
    */
   void merge(Evaluation<T, M> evaluation);

   /**
    * Outputs the evaluation results to the given print stream.
    *
    * @param printStream the print stream to write to
    */
   void output(PrintStream printStream);

}//END OF Evaluation
