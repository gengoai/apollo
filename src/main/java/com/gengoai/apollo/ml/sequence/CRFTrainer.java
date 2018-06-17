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

package com.gengoai.apollo.ml.sequence;

import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.io.Resources;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import third_party.org.chokkan.crfsuite.*;

/**
 * <p>Trains a CRF model using CRFSuite.</p>
 *
 * @author David B. Bracewell
 */
public class CRFTrainer extends SequenceLabelerLearner {
   private static final long serialVersionUID = 1L;
   @Getter
   private Solver solver = Solver.LBFGS;
   @Getter
   @Setter
   private int maxIterations = 100;
   @Getter
   @Setter
   private double c2 = 1.0;

   @Override
   public void resetLearnerParameters() {

   }

   /**
    * Sets the type of solver to use for optimization.
    *
    * @param solver the solver
    */
   public void setSolver(@NonNull Solver solver) {
      this.solver = solver;
   }

   @Override
   protected SequenceLabeler trainImpl(Dataset<Sequence> dataset) {
      setTransitionFeatures(TransitionFeature.NO_OPT);
      LibraryLoader.INSTANCE.load();
      Trainer trainer = new Trainer();
      dataset.forEach(sequence -> {
         ItemSequence seq = new ItemSequence();
         StringList labels = new StringList();
         for (Instance instance : sequence.asInstances()) {
            Item item = new Item();
            instance.forEach(f -> item.add(new Attribute(f.getFeatureName(), f.getValue())));
            labels.add(instance.getLabel().toString());
            seq.add(item);
         }
         trainer.append(seq, labels, 0);
      });
      trainer.select(solver.parameterSetting, "crf1d");
      trainer.set("max_iterations", Integer.toString(maxIterations));
      trainer.set("c2", Double.toString(c2));
      String tempFile = Resources.temporaryFile().asFile().get().getAbsolutePath();
      trainer.train(tempFile, -1);
      trainer.clear();
      return new CRFTagger(this, tempFile);
   }

}// END OF CRFTrainer
