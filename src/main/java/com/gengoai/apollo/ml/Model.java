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

package com.gengoai.apollo.ml;

import com.gengoai.Stopwatch;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.params.*;
import com.gengoai.apollo.optimization.WeightUpdate;
import com.gengoai.apollo.optimization.activation.Activation;
import com.gengoai.apollo.optimization.loss.LossFunction;
import com.gengoai.io.resource.Resource;
import com.gengoai.logging.Logger;

import java.io.Serializable;

import static com.gengoai.Validation.notNull;

/**
 * <p>
 * A machine learning model learns a function from an input object to an output by fitting the function to a given
 * dataset. The input and output of the function are dependent on the type of model/function being learned. The model
 * interface defines the methodology for fitting the function using the various <code>fit</code> methods. Varying models
 * have different parameters for fitting, which are defined using {@link FitParameters}. Each model will have its own
 * subclass of {@link FitParameters} defining its specific parameters. A model maybe used to directly transform future
 * unseen input objects into an output (e.g. regression, classification, and sequence labeling) or the result of fitting
 * the model may generate a usable non-model result (e.g. clustering and word embedding).
 * </p>
 * <p>
 * Subclasses of <code>Model</code> define the function for transforming a given input into an output, where the type of
 * output is dependent on the type model.
 * </p>
 *
 * @author David B. Bracewell
 */
public abstract class Model implements Serializable {
   public static final BoolParam verbose = new BoolParam("verbose",
                                                         "Determines if information about the fitting of the model is displayed during training.");

   public static final IntParam maxIterations = new IntParam("maxIterations",
                                                             "The maximum number of iterations to perform training",
                                                             i -> i > 0);

   public static final Param<Activation> activation = new Param<>("activation",
                                                                  Activation.class,
                                                                  "The activation function to use");
   public static final Param<LossFunction> lossFunction = new Param<>("lossFunction",
                                                                      LossFunction.class,
                                                                      "The loss function function to use");
   public static final Param<WeightUpdate> weightUpdater = new Param<>("weightUpdater",
                                                                       WeightUpdate.class,
                                                                       "The weight update function to use.");
   public static final IntParam batchSize = new IntParam("batchSize",
                                                         "The number of examples per batch.",
                                                         i -> i > 0);
   public static final IntParam reportInterval = new IntParam("reportInterval",
                                                              "The number of iterations between reporting fit updates.",
                                                              i -> i >= 0);
   public static final IntParam historySize = new IntParam("history",
                                                           "The number of consecutive iterations to terminate early when the criteria hasn't changed within the given tolerance.",
                                                           i -> i >= 0);
   public static final DoubleParam tolerance = new DoubleParam("tolerance",
                                                               "The amount of change in criteria between iterations to be considered converged",
                                                               i -> i >= 0);

   public static final BoolParam cacheData = new BoolParam("cacheData",
                                                           "Whether to cache the data during fitting.");


   public static ParamValuePair<Boolean> verbose(boolean isVerbose) {
      return verbose.set(isVerbose);
   }

   private static final Logger log = Logger.getLogger(Model.class);
   private static final long serialVersionUID = 1L;

   /**
    * Reads a Classifier from the given resource
    *
    * @param <T>      the type parameter
    * @param resource the resource containing the saved model
    * @return the deserialized (loaded) model
    * @throws Exception Something went wrong reading in the model
    */
   public static <T extends Model> T read(Resource resource) throws Exception {
      return resource.readObject();
   }

   /**
    * Fits the model on the given {@link Dataset} using the given {@link FitParameters}.
    *
    * @param dataset       the dataset
    * @param fitParameters the fit parameters
    */
   public final void fit(Dataset dataset, ParamValuePair... fitParameters) {
      notNull(dataset);
      ParamMap parameters = getDefaultFitParameters();
      parameters.update(fitParameters);
      Stopwatch sw = Stopwatch.createStarted();
      boolean isVerbose = parameters.getOrDefault(verbose, false);

      if (isVerbose) {
         log.info("Preprocessing...");
      }
      Dataset preprocessed = getPipeline().fitAndPreprocess(dataset).cache();
      sw.stop();

      if (isVerbose) {
         log.info("Preprocessing completed. ({0})", sw);
      }
      fitPreprocessed(preprocessed, parameters);
   }

   /**
    * Training implementation over preprocessed dataset
    *
    * @param preprocessed  the preprocessed dataset
    * @param fitParameters the fit parameters
    */
   protected abstract void fitPreprocessed(Dataset preprocessed, ParamMap fitParameters);

   /**
    * Gets default fit parameters for the model.
    *
    * @return the default fit parameters
    */
   public abstract ParamMap getDefaultFitParameters();

   /**
    * Gets the pipeline associated with the model. Subclasses should override the return type to match the requirements
    * of the model.
    *
    * @return shapeless.the model parameters
    */
   public abstract Pipeline getPipeline();

   /**
    * Gets the number of features in the model.
    *
    * @return the number of features
    */
   public int getNumberOfFeatures() {
      return getPipeline().featureVectorizer.size();
   }

   /**
    * Gets the number of labels in the model.
    *
    * @return the number of labels
    */
   public abstract int getNumberOfLabels();

   /**
    * Writes the model to the given resource.
    *
    * @param resource the resource to write the model to
    * @throws Exception Something went wrong writing the model
    */
   public void write(Resource resource) throws Exception {
      resource.setIsCompressed(true).writeObject(this);
   }
}//END OF Model
