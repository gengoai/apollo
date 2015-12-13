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

package com.davidbracewell.apollo.ml.classification.linear;

import com.davidbracewell.apollo.ml.Dataset;
import com.davidbracewell.apollo.ml.FeatureVector;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.ClassifierLearner;
import de.bwaldvogel.liblinear.*;
import lombok.Data;

import java.util.Iterator;

/**
 * @author David B. Bracewell
 */
@Data
public class LibLinearLearner extends ClassifierLearner {
  private static final long serialVersionUID = 6185877887927537722L;
  private SolverType solver = SolverType.L2R_LR;
  private double C = 1;
  private double eps = 0.0001;
  private boolean verbose = false;

  @Override
  protected LibLinearModel trainImpl(Dataset<Instance> dataset) {
    LibLinearModel model = new LibLinearModel(
      dataset.getLabelEncoder(),
      dataset.getFeatureEncoder(),
      dataset.getPreprocessors().getModelProcessors()
    );

    Problem problem = new Problem();
    problem.l = dataset.size();
    problem.x = new Feature[problem.l][];
    problem.y = new double[problem.l];
    int index = 0;
    for (Iterator<Instance> iitr = dataset.iterator(); iitr.hasNext(); index++) {
      FeatureVector vector = iitr.next().toVector(dataset.getFeatureEncoder(),dataset.getLabelEncoder());
      problem.x[index] = LibLinearModel.toFeature(vector);
      problem.y[index] = vector.getLabel();
    }
    problem.n = model.getFeatureEncoder().size() + 1;

    if (isVerbose()) {
      Linear.enableDebugOutput();
    } else {
      Linear.disableDebugOutput();
    }

    model.model = Linear.train(problem, new Parameter(solver, C, eps));
    return model;
  }

  @Override
  public void reset() {

  }

}//END OF LibLinearLearner
