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

package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import de.bwaldvogel.liblinear.*;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;

import java.util.Iterator;

/**
 * A learner that uses LibLinear to train a logistic regression or SVM model
 *
 * @author David B. Bracewell
 */
public class LibLinearLearner extends ClassifierLearner {
   private static final long serialVersionUID = 6185877887927537722L;
   @Getter
   private SolverType solver = SolverType.L2R_LR;
   @Getter
   @Setter
   private double C = 1;
   @Getter
   @Setter
   private double eps = 0.0001;
   @Getter
   @Setter
   private boolean bias = false;
   @Getter
   @Setter
   private boolean verbose = false;

   @Override
   public void resetLearnerParameters() {

   }

   /**
    * Sets the type of solver to use
    *
    * @param solver The solve to use
    */
   public void setSolver(@NonNull SolverType solver) {
      this.solver = solver;
   }

   @Override
   protected LibLinearModel trainImpl(Dataset<Instance> dataset) {
      LibLinearModel model = new LibLinearModel(this);

      Problem problem = new Problem();
      problem.l = dataset.size();
      problem.x = new Feature[problem.l][];
      problem.y = new double[problem.l];
      problem.bias = bias ? 0 : -1;

      int biasIndex = (bias ? model.numberOfFeatures() + 1 : -1);

      int index = 0;
      for (Iterator<Instance> iitr = dataset.iterator(); iitr.hasNext(); index++) {
         NDArray vector = iitr.next().toVector(dataset.getEncoderPair());
         problem.x[index] = LibLinearModel.toFeature(vector, biasIndex);
         problem.y[index] = vector.getLabel();
      }
      problem.n = model.numberOfFeatures() + 1;

      if (verbose) {
         Linear.enableDebugOutput();
      } else {
         Linear.disableDebugOutput();
      }

      model.biasIndex = biasIndex;
      model.model = Linear.train(problem, new Parameter(solver, C, eps));
      return model;
   }

}//END OF LibLinearLearner
