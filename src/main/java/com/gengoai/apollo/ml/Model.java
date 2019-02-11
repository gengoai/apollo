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
import com.gengoai.io.resource.Resource;
import com.gengoai.logging.Logger;

import java.io.Serializable;
import java.util.function.Consumer;

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
 * @param <T> the return type parameter of the fit methods
 * @author David B. Bracewell
 */
public abstract class Model<T> implements Serializable {
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
    * Fits the model on the given {@link Dataset} using the model's default {@link FitParameters}.
    *
    * @param dataset the dataset to fit the model on
    * @return the result of fitting
    */
   public final T fit(Dataset dataset) {
      return fit(dataset, getDefaultFitParameters());
   }

   /**
    * Fits the model on the given {@link Dataset} using the given consumer to modify the model's default {@link
    * FitParameters}.
    *
    * @param dataset          the dataset
    * @param parameterUpdater the consumer to use to update the fit parameters
    * @return the result of fitting
    */
   public final T fit(Dataset dataset, Consumer<? extends FitParameters> parameterUpdater) {
      return fit(dataset, getDefaultFitParameters().update(parameterUpdater));
   }

   /**
    * Fits the model on the given {@link Dataset} using the given {@link FitParameters}.
    *
    * @param dataset       the dataset
    * @param fitParameters the fit parameters
    * @return the result of fitting
    */
   public final T fit(Dataset dataset, FitParameters fitParameters) {
      Stopwatch sw = Stopwatch.createStarted();
      if (fitParameters.verbose) {
         log.info("Preprocessing...");
      }
      Dataset preprocessed = getPipeline().fitAndPreprocess(dataset).cache();
      sw.stop();
      if (fitParameters.verbose) {
         log.info("Preprocessing completed. ({0})", sw);
      }
      return fitPreprocessed(preprocessed, fitParameters);
   }

   /**
    * Training implementation over preprocessed dataset
    *
    * @param preprocessed  the preprocessed dataset
    * @param fitParameters the fit parameters
    * @return the result of fitting
    */
   protected abstract T fitPreprocessed(Dataset preprocessed, FitParameters fitParameters);

   /**
    * Gets default fit parameters for the model.
    *
    * @return the default fit parameters
    */
   public abstract FitParameters getDefaultFitParameters();


   /**
    * Gets the pipeline associated with the model. Subclasses should override the return type to match the requirements
    * of the model.
    *
    * @return the model parameters
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
