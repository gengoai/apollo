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

package com.gengoai.apollo.ml.regression;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.LibLinear;
import com.gengoai.apollo.ml.ModelParameters;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.conversion.Cast;
import com.gengoai.function.SerializableSupplier;
import com.gengoai.stream.MStream;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.SolverType;

import static com.gengoai.Validation.notNull;

/**
 * <p>Regression model using LibLinear</p>
 *
 * @author David B. Bracewell
 */
public class LibLinearLinearRegression extends Regression {
   private static final long serialVersionUID = 1L;
   private Model model;
   private int biasIndex;


   /**
    * Instantiates a new Lib linear linear regression.
    */
   public LibLinearLinearRegression(Preprocessor... preprocessors) {
      super(preprocessors);
   }

   public LibLinearLinearRegression(ModelParameters modelParameters) {
      super(modelParameters);
   }

   @Override
   public double estimate(Example data) {
      return LibLinear.regress(model, encodeAndPreprocess(data), biasIndex);
   }

   private void fit(SerializableSupplier<MStream<NDArray>> dataSupplier, Parameters fitParameters) {
      biasIndex = (fitParameters.bias ? getNumberOfFeatures() + 1 : -1);
      model = LibLinear.fit(dataSupplier,
                            new Parameter(SolverType.L2R_L2LOSS_SVR,
                                          fitParameters.C,
                                          fitParameters.eps,
                                          fitParameters.maxIterations,
                                          fitParameters.p),
                            fitParameters.verbose,
                            getNumberOfFeatures(),
                            biasIndex
                           );
   }

   @Override
   public Regression fitPreprocessed(Dataset dataSupplier, FitParameters fitParameters) {
      fit(() -> dataSupplier.stream().map(this::encode), notNull(Cast.as(fitParameters, Parameters.class)));
      return this;
   }

   @Override
   public Parameters getDefaultFitParameters() {
      return new Parameters();
   }

   /**
    * Custom fit parameters for LibLinear
    */
   public static class Parameters extends FitParameters {
      private static final long serialVersionUID = 1L;
      /**
       * The cost parameter (default 1.0)
       */
      public double C = 1.0;
      /**
       * Use a bias feature or not. (default false)
       */
      public boolean bias = false;
      /**
       * The tolerance for termination.(default 0.0001)
       */
      public double eps = 0.0001;
      /**
       * The maximum number of iterations to run the trainer (Default 1000)
       */
      public int maxIterations = 1000;

      /**
       * The epsilon in loss function of epsilon-SVR (default 0.1)
       */
      public double p = 0.1;

   }

}//END OF MultivariateLinearRegression
