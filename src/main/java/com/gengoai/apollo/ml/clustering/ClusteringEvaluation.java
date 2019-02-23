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

package com.gengoai.apollo.ml.clustering;

import java.io.PrintStream;

/**
 * <p>
 *    Specialized evaluation for evaluating the result of {@link Clusterer}s.
 * </p>
 *
 * @author David B. Bracewell
 */
public interface ClusteringEvaluation  {

   /**
    * Evaluates the given clustering.
    *
    * @param clustering the clustering to evaluate.
    */
   void evaluate(Clusterer clustering);

   /**
    * Outputs the evaluation results to the given print stream.
    *
    * @param printStream the print stream to write to
    */
   void output(PrintStream printStream);

   /**
    * Outputs the evaluation results to standard out.
    */
   default void output() {
      output(System.out);
   }

}//END OF ClusteringEvaluation