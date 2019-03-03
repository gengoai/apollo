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

package com.gengoai.apollo.ml.classification;

import cc.mallet.classify.C45;
import cc.mallet.classify.C45Trainer;
import cc.mallet.classify.ClassifierTrainer;
import cc.mallet.types.GainRatio;
import cc.mallet.util.MalletLogger;
import com.gengoai.apollo.ml.params.BoolParam;
import com.gengoai.apollo.ml.params.IntParam;
import com.gengoai.apollo.ml.params.ParamMap;
import com.gengoai.apollo.ml.preprocess.Preprocessor;

import java.util.logging.Level;

/**
 * <p>
 * A classifier wrapper around Mallet's C4.5 decision tree. The C4.5 algorithm constructs a decision tree by selecting
 * the feature that best splits the dataset into sub-samples that support the labels using information gain at each node
 * in the tree. The feature with the highest normalized gain is chosen as the decision for that node.
 * </p>
 *
 * @author David B. Bracewell
 */
public class C45Classifier extends MalletClassifier {
   public static final BoolParam depthLimited = new BoolParam("depthLimited",
                                                              "True - limit the depth of the tree, False let the tree get as deep as needed.");

   public static final BoolParam doPruning = new BoolParam("doPruning",
                                                           "True - prune the tree, False no pruning");

   public static final IntParam minInstances = new IntParam("minInstances",
                                                            "The minimum number of instances a leaf in the tree must contain",
                                                            i -> i > 0);
   public static final IntParam maxDepth = new IntParam("maxDepth",
                                                        "The maximum depth the tree can grow if depth limited.",
                                                        i -> i > 0);
   private static final long serialVersionUID = 1L;

   /**
    * Instantiates a new C45 classifier.
    *
    * @param preprocessors the preprocessors
    */
   public C45Classifier(Preprocessor... preprocessors) {
      super(preprocessors);
   }

   @Override
   public ParamMap getDefaultFitParameters() {
      return new ParamMap(verbose.set(false),
                          depthLimited.set(false),
                          doPruning.set(true),
                          minInstances.set(2),
                          maxDepth.set(4));
   }

   @Override
   protected ClassifierTrainer<?> getTrainer(ParamMap parameters) {
      C45Trainer trainer = new C45Trainer();
      if (parameters.getOrDefault(verbose, false)) {
         MalletLogger.getLogger(C45Trainer.class.getName()).setLevel(Level.INFO);
         MalletLogger.getLogger(GainRatio.class.getName()).setLevel(Level.INFO);
         MalletLogger.getLogger(C45.class.getName()).setLevel(Level.INFO);
      } else {
         MalletLogger.getLogger(C45Trainer.class.getName()).setLevel(Level.OFF);
         MalletLogger.getLogger(GainRatio.class.getName()).setLevel(Level.OFF);
         MalletLogger.getLogger(C45.class.getName()).setLevel(Level.OFF);
      }
      trainer.setDepthLimited(parameters.get(depthLimited));
      trainer.setDoPruning(parameters.get(doPruning));
      trainer.setMaxDepth(parameters.get(maxDepth));
      trainer.setMinNumInsts(parameters.get(minInstances));
      return trainer;
   }

}//END OF C45Classifier
