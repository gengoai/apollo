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

import com.gengoai.ParameterDef;
import com.gengoai.apollo.ml.*;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.conversion.Cast;
import com.gengoai.logging.Loggable;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.SolverType;

import static com.gengoai.Validation.notNull;

/**
 * <p>
 * A classifier wrapper around LibLinear.
 * </p>
 *
 * @author David B. Bracewell
 */
public class LibLinearModel extends Classifier implements Loggable {
   public static final ParameterDef<Double> C = ParameterDef.doubleParam("C");
   public static final ParameterDef<Boolean> bias = ParameterDef.boolParam("bias");
   public static final ParameterDef<Double> eps = ParameterDef.doubleParam("eps");
   public static final ParameterDef<SolverType> solver = ParameterDef.param("solver", SolverType.class);
   private static final long serialVersionUID = 1L;
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
   protected void fitPreprocessed(Dataset preprocessed, FitParameters parameters) {
      Parameters fitParameters = notNull(Cast.as(parameters, Parameters.class));
      biasIndex = (fitParameters.bias.value() ? getNumberOfFeatures() + 1 : -1);
      model = LibLinear.fit(() -> preprocessed.asVectorStream(getPipeline()),
                            new Parameter(fitParameters.solver.value(),
                                          fitParameters.C.value(),
                                          fitParameters.tolerance.value(),
                                          fitParameters.maxIterations.value(),
                                          fitParameters.eps.value()),
                            fitParameters.verbose.value(),
                            getNumberOfFeatures(),
                            biasIndex
                           );
   }

   @Override
   public Parameters getFitParameters() {
      return new Parameters();
   }

   @Override
   public Classification predict(Example example) {
      return new Classification(LibLinear.estimate(model,
                                                   example.preprocessAndTransform(getPipeline()),
                                                   biasIndex),
                                getPipeline().labelVectorizer);
   }

   /**
    * Custom fit parameters for LibLinear
    */
   public static class Parameters extends FitParameters<Parameters> {
      private static final long serialVersionUID = 1L;
      /**
       * The cost parameter (default 1.0)
       */
      public final Parameter<Double> C = parameter(LibLinearModel.C, 1.0);
      /**
       * Use a bias feature or not. (default false)
       */
      public final Parameter<Boolean> bias = parameter(LibLinearModel.bias, false);
      /**
       * The epsilon in loss function of epsilon-SVR (default 0.1)
       */
      public final Parameter<Double> eps = parameter(LibLinearModel.eps, 0.1);
      /**
       * The maximum number of iterations to run the trainer (Default 1000)
       */
      public final Parameter<Integer> maxIterations = parameter(Params.Optimizable.maxIterations, 1000);
      /**
       * The Solver to use. (default L2R_LR)
       */
      public final Parameter<SolverType> solver = parameter(LibLinearModel.solver, SolverType.L2R_LR);
      /**
       * The tolerance for termination.(default 0.0001)
       */
      public final Parameter<Double> tolerance = parameter(Params.Optimizable.tolerance, 1e-4);
   }
}//END OF LibLinearModel
