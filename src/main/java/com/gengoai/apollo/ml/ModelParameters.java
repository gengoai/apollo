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

import com.gengoai.Parameters;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.ml.preprocess.PreprocessorList;
import com.gengoai.apollo.ml.sequence.SequenceValidator;
import com.gengoai.apollo.ml.vectorizer.BinaryLabelVectorizer;
import com.gengoai.apollo.ml.vectorizer.DoubleVectorizer;
import com.gengoai.apollo.ml.vectorizer.IndexVectorizer;
import com.gengoai.apollo.ml.vectorizer.Vectorizer;

import java.io.Serializable;
import java.util.Collections;
import java.util.function.Consumer;

/**
 * <p>Parameters for defining the basic components of a Machine Learning Model.</p>
 *
 * @author David B. Bracewell
 */
public class ModelParameters implements Parameters<ModelParameters>, Serializable {
   private static final long serialVersionUID = 1L;
   /**
    * The preprocessors used for filtering and transforming the data
    */
   public final PreprocessorList preprocessors = new PreprocessorList();
   private final Vectorizer<?> labelVectorizer;
   /**
    * The vectorizer to use for encoding/decoding {@link Feature} and transforming {@link Example} into {@link
    * com.gengoai.apollo.linear.NDArray}****
    */
   public Vectorizer<String> featureVectorizer = IndexVectorizer.featureVectorizer();
   /**
    * The validator to use for verifying correct transitions in sequence labeling problems
    */
   public SequenceValidator sequenceValidator = SequenceValidator.ALWAYS_TRUE;


   /**
    * Instantiates a new Model parameters.
    *
    * @param labelVectorizer the label vectorizer
    */
   protected ModelParameters(Vectorizer<?> labelVectorizer) {
      this.labelVectorizer = labelVectorizer;
   }

   /**
    * Creates a <code>ModelParameters</code> for classification using either a {@link BinaryLabelVectorizer} or an
    * {@link IndexVectorizer} as the label vectorizer.
    *
    * @param isBinary True - if the problem is binary and a {@link BinaryLabelVectorizer} should be use, False otherwise
    *                 having an {@link IndexVectorizer} used as the label vectorizer
    * @return the model parameters
    */
   public static ModelParameters classification(boolean isBinary) {
      if (isBinary) {
         return new ModelParameters(new BinaryLabelVectorizer());
      }
      return new ModelParameters(IndexVectorizer.labelVectorizer());
   }

   /**
    * Creates a <code>ModelParameters</code> for classification using either a {@link BinaryLabelVectorizer} or an
    * {@link IndexVectorizer} as the label vectorizer.
    *
    * @param isBinary True - if the problem is binary and a {@link BinaryLabelVectorizer} should be use, False otherwise
    *                 having an {@link IndexVectorizer} used as the label vectorizer
    * @param updater  the updater
    * @return the model parameters
    */
   public static ModelParameters classification(boolean isBinary, Consumer<ModelParameters> updater) {
      return classification(isBinary).update(updater);
   }


   /**
    * Creates a new <code>ModelParameters</code> with the given label vectorizer..
    *
    * @param labelVectorizer the label vectorizer
    * @return the model parameters
    */
   public static ModelParameters create(Vectorizer<?> labelVectorizer) {
      return new ModelParameters(labelVectorizer);
   }

   /**
    * Creates a new <code>ModelParameters</code> with the given label vectorizer..
    *
    * @param labelVectorizer the label vectorizer
    * @param updater         the updater
    * @return the model parameters
    */
   public static ModelParameters create(Vectorizer<?> labelVectorizer, Consumer<ModelParameters> updater) {
      return new ModelParameters(labelVectorizer).update(updater);
   }


   /**
    * Creates a new <code>ModelParameters</code> object with an {@link IndexVectorizer} for the label vectorizer.
    *
    * @return the model parameters
    */
   public static ModelParameters indexedLabelVectorizer() {
      return new ModelParameters(IndexVectorizer.labelVectorizer());
   }

   /**
    * Creates a new <code>ModelParameters</code> object with an {@link IndexVectorizer} for the label vectorizer.
    *
    * @param updater the updater
    * @return the model parameters
    */
   public static ModelParameters indexedLabelVectorizer(Consumer<ModelParameters> updater) {
      return new ModelParameters(IndexVectorizer.labelVectorizer()).update(updater);
   }

   /**
    * Creates a new <code>ModelParameters</code> with a {@link DoubleVectorizer} as the label vectorizer for regression
    * problems.
    *
    * @return the model parameters
    */
   public static ModelParameters regression() {
      return new ModelParameters(new DoubleVectorizer());
   }

   /**
    * Creates a new <code>ModelParameters</code> with a {@link DoubleVectorizer} as the label vectorizer for regression
    * problems.
    *
    * @param updater the updater
    * @return the model parameters
    */
   public static ModelParameters regression(Consumer<ModelParameters> updater) {
      return new ModelParameters(new DoubleVectorizer()).update(updater);
   }

   /**
    * Gets the label vectorizer.
    *
    * @return the vectorizer to use for encoding/decoding labels
    */
   public Vectorizer<?> labelVectorizer() {
      return labelVectorizer;
   }

   /**
    * Convenience method for adding multiple {@link Preprocessor}s.
    *
    * @param preprocessors the preprocessors
    * @return the model parameters
    */
   public ModelParameters preprocessors(Preprocessor... preprocessors) {
      this.preprocessors.clear();
      Collections.addAll(this.preprocessors, preprocessors);
      return this;
   }

   /**
    * Convenience method for adding multiple {@link Preprocessor}s.
    *
    * @param preprocessors the preprocessors
    * @return the model parameters
    */
   public ModelParameters preprocessors(PreprocessorList preprocessors) {
      this.preprocessors.clear();
      this.preprocessors.addAll(preprocessors);
      return this;
   }

}//END OF ModelParameters
