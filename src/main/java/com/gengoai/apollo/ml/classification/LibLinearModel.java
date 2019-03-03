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

import com.gengoai.apollo.ml.DiscretePipeline;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.LibLinear;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.params.BoolParam;
import com.gengoai.apollo.ml.params.DoubleParam;
import com.gengoai.apollo.ml.params.Param;
import com.gengoai.apollo.ml.params.ParamMap;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.logging.Loggable;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.SolverType;

/**
 * <p>
 * A classifier wrapper around LibLinear.
 * </p>
 *
 * @author David B. Bracewell
 */
public class LibLinearModel extends Classifier implements Loggable {
   private static final long serialVersionUID = 1L;

   public static final DoubleParam C = new DoubleParam("C",
                                                       "The cost parameter.",
                                                       d -> d >= 0);

   public static final BoolParam bias = new BoolParam("bias",
                                                      "Use a bias feature or not.");

   public static final DoubleParam eps = new DoubleParam("eps",
                                                         "The tolerance for termination.",
                                                         d -> d >= 0);

   public static final DoubleParam p = new DoubleParam("p",
                                                       "The epsilon in loss function of epsilon-SVR.",
                                                       d -> d >= 0);

   public static final Param<SolverType> solver = new Param<>("solver", SolverType.class,
                                                              "The Solver to use.");

   private int biasIndex = -1;
   private Model model;


   /**
    * Instantiates a new Lib linear model.
    *
    * @param preprocessors the preprocessors
    */
   public LibLinearModel(Preprocessor... preprocessors) {
      super(preprocessors);
   }

   /**
    * Instantiates a new Lib linear model.
    *
    * @param modelParameters the model parameters
    */
   public LibLinearModel(DiscretePipeline modelParameters) {
      super(modelParameters);
   }


   @Override
   protected void fitPreprocessed(Dataset preprocessed, ParamMap parameters) {
      biasIndex = (parameters.get(bias) ? getNumberOfFeatures() + 1 : -1);
      model = LibLinear.fit(() -> preprocessed.asVectorStream(getPipeline()),
                            new Parameter(parameters.get(solver),
                                          parameters.get(C),
                                          parameters.get(eps),
                                          parameters.get(maxIterations),
                                          parameters.get(p)),
                            parameters.get(verbose),
                            getNumberOfFeatures(),
                            biasIndex
                           );
   }

   @Override
   public ParamMap getDefaultFitParameters() {
      return new ParamMap(
         maxIterations.set(1000),
         bias.set(false),
         C.set(1.0),
         eps.set(1e-4),
         p.set(0.1),
         solver.set(SolverType.L2R_LR),
         verbose.set(false)
      );
   }

   @Override
   public Classification predict(Example example) {
      return new Classification(LibLinear.estimate(model,
                                                   example.preprocessAndTransform(getPipeline()),
                                                   biasIndex),
                                getPipeline().labelVectorizer);
   }


}//END OF LibLinearModel
