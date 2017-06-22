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

package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.LibLinearLearner;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.io.QuietIO;

import java.util.Map;

/**
 * <p>Trains a Maximum Entropy Markov Model using LibLinear.</p>
 *
 * @author David B. Bracewell
 */
public class MEMMLearner extends SequenceLabelerLearner {
   private static final long serialVersionUID = 1L;
   private LibLinearLearner learner = new LibLinearLearner();

   @Override
   public Object getParameter(String name) {
      return learner.getParameter(name);
   }

   @Override
   public Map<String, ?> getParameters() {
      return learner.getParameters();
   }

   @Override
   public void resetLearnerParameters() {
   }

   @Override
   public MEMMLearner setParameter(String name, Object value) {
      learner.setParameter(name, value);
      return this;
   }

   @Override
   public MEMMLearner setParameters(Map<String, Object> parameters) {
      learner.setParameters(parameters);
      return Cast.as(this);
   }

   @Override
   protected SequenceLabeler trainImpl(Dataset<Sequence> dataset) {
      MEMM model = new MEMM(dataset.getLabelEncoder(),
                            dataset.getFeatureEncoder(),
                            dataset.getPreprocessors().getModelProcessors(),
                            getTransitionFeatures(),
                            getValidator()
      );
      Dataset<Instance> nd = Dataset.classification()
                                    .source(dataset.stream().flatMap(s -> s.asInstances().stream()));
      QuietIO.closeQuietly(dataset);
      model.model = Cast.as(learner.train(nd));
      return model;
   }
}// END OF MEMMLearner
